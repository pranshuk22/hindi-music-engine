"""
pipeline/processor.py
──────────────────────
Master pipeline orchestrator.

Extraction sequence (FEATURES.md steps 1–18, 45s smart-trim variant):

  1.  Download audio (yt-dlp)
  2.  Trim to 45s hook clip (smart RMS scan, or forced clip_start)
  3.  DELETE raw download
  4.  demucs stem separation → vocals + instrumental
      └─ on failure: fallback to full clip, set stem_separation_failed=True
  5.  Extract base spectral features (full clip)
  6.  Extract Indian-specific features (vocal stem / full clip / instr stem)
  7.  CLAP embedding + lyrics fetch IN PARALLEL
  8.  NLP lyric embedding (zero vector if lyrics missing)
  9.  DELETE all audio (clip + stems)
  10. Build weighted fused vector (~997d) via embedder.build_fused_vector()
  11. PCA compress → 256d if pca_model.pkl exists;
      else save raw to embeddings_raw/ for build_index.py to compress later
  12. Save embedding, features JSON, NLP vector
  13. Insert to SQLite with all quality flags

Thread safety:
  CLAP and demucs models are loaded once at module level.
  ThreadPoolExecutor workers share them safely (torch CPU is GIL-safe).
  Do NOT run process_song() across multiple *processes*.
"""

import os
import shutil
import logging
import numpy as np
import joblib
import concurrent.futures

from pipeline.downloader       import download_audio, search_and_download
from pipeline.trimmer          import trim_audio, cleanup
from pipeline.stemmer          import separate_stems
from pipeline.feature_extractor import extract_features, extract_vocal_mfcc, save_features, get_tempo_bucket
from pipeline.indian_features  import extract_all_indian_features
from pipeline.embedder         import (
    get_clap_embedding,
    build_fused_vector,
    compress_with_pca,
    save_embedding,
)
from pipeline.lyrics_extractor import fetch_lyrics
from pipeline.nlp_embedder     import get_text_embedding, save_embedding as save_nlp_embedding
from utils.db                  import insert_song

logger = logging.getLogger(__name__)

# ── Directory layout ──────────────────────────────────────────────────────────
EMBEDDINGS_DIR     = "data/embeddings"       # 256d PCA-compressed (final)
EMBEDDINGS_RAW_DIR = "data/embeddings_raw"   # ~997d raw fused (pilot, pre-PCA)
FEATURES_DIR       = "data/features"
NLP_DIR            = "data/nlp"
TEMP_DIR           = "/tmp/hindi_music_temp"
PCA_PATH           = "index/pca_model.pkl"

for _d in (EMBEDDINGS_DIR, EMBEDDINGS_RAW_DIR, FEATURES_DIR, NLP_DIR, TEMP_DIR):
    os.makedirs(_d, exist_ok=True)


# ── PCA model singleton ───────────────────────────────────────────────────────
_pca_model = None

def _get_pca():
    """
    Load the pre-fitted PCA model on first call.
    Returns None during the pilot batch (before build_index.py --fit-pca).
    """
    global _pca_model
    if _pca_model is not None:
        return _pca_model
    if os.path.isfile(PCA_PATH):
        logger.info("Loading PCA model from %s", PCA_PATH)
        _pca_model = joblib.load(PCA_PATH)
    else:
        logger.warning(
            "PCA model not found at %s — raw fused vectors will be saved to "
            "%s. Run build_index.py --fit-pca after processing the pilot batch.",
            PCA_PATH, EMBEDDINGS_RAW_DIR,
        )
    return _pca_model


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_song_id(title: str, artist: str) -> str:
    raw = f"{title}_{artist}".lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in raw)[:50]


def _to_list(v) -> list:
    """Convert ndarray or list to a JSON-serialisable plain list."""
    return v.tolist() if isinstance(v, np.ndarray) else list(v)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_song(
    title:       str,
    artist:      str,
    youtube_url: str = None,
    clip_start:  int = None,     # None → smart RMS scan; int → forced offset
    category:    str = "unknown",
) -> str:
    """
    Full ingestion pipeline for a single song.

    Args:
        title:       Song title.
        artist:      Artist name.
        youtube_url: Direct YouTube URL. Searched by title+artist if None.
        clip_start:  Manual start offset in seconds (from songs.csv).
                     Pass None — not 0 — to trigger the smart RMS scan.
                     (0 is a valid forced start; None means "auto-detect".)
        category:    Category label for evaluation (ghazal, sufi, party, …).

    Returns:
        song_id — the unique key used for all file paths and DB records.
    """
    song_id = _make_song_id(title, artist)
    logger.info("\n%s\nProcessing: %s — %s  (id: %s)", "─" * 60, title, artist, song_id)

    raw_path               = None
    clip_path              = None
    stems                  = None
    stem_separation_failed = False

    try:
        # ── Step 1: Download ─────────────────────────────────────────────────
        logger.info("[1] Downloading audio…")
        raw_path = (
            download_audio(youtube_url, TEMP_DIR)
            if youtube_url
            else search_and_download(title, artist, TEMP_DIR)
        )

        # ── Step 2: Trim to 45s hook ─────────────────────────────────────────
        # IMPORTANT: use `clip_start is not None`, not `if clip_start`.
        # clip_start=0 is a valid forced start; the falsy check would lose it.
        logger.info("[2] Trimming to 45s clip (%s)…",
                    f"forced start={clip_start}s" if clip_start is not None
                    else "smart RMS scan")
        clip_path    = os.path.join(TEMP_DIR, f"{song_id}_clip.wav")
        actual_start = trim_audio(
            raw_path, clip_path,
            duration_sec    = 45,
            force_start_sec = clip_start,   # None is intentional when not set
        )
        logger.info("   Clip starts at %.1fs", actual_start)

        # ── Step 3: Delete raw download ──────────────────────────────────────
        cleanup(raw_path);  raw_path = None

        # ── Step 4: Stem separation ──────────────────────────────────────────
        logger.info("[3] Stem separation (htdemucs_light)…")
        try:
            # FIX 1: Pass only one argument and unpack the tuple
            vocals_path, instr_path = separate_stems(clip_path)
            # Recreate the dictionary structure so the cleanup() block at Step 10 works
            stems = {"vocals": vocals_path, "instrumental": instr_path}
            logger.info("   Stems ready: %s | %s", vocals_path, instr_path)
        except Exception as stem_exc:
            logger.warning(
                "Stem separation failed (%s) — falling back to full clip for "
                "both stems. vocal_energy_ratio, murki_index, and "
                "onset_skewness will be degraded. stem_separation_failed=1.",
                stem_exc,
            )
            stem_separation_failed = True
            vocals_path = clip_path
            instr_path  = clip_path
            stems       = None

        # ── Steps 5–6: Feature extraction ────────────────────────────────────
        logger.info("[4] Base spectral features (full clip)…")
        base_feat = extract_features(clip_path)
        # Provides: tempo, spectral_centroid_mean, spectral_rolloff_mean,
        #           spectral_bandwidth_mean, zcr_mean, rms_mean, tonnetz_mean(6d)

        tempo_bucket_arr = get_tempo_bucket(float(base_feat.get("tempo", 100.0)))

        logger.info("[5] Indian-specific features…")
        indian_feat = extract_all_indian_features(
            clip_path   = clip_path,
            vocals_path = vocals_path,
            instr_path  = instr_path,
        )
        # Provides: vocal_mfcc(20d), vocal_energy_ratio, murki_index,
        #           microtonal_pcp(36d), meend_variance, raga_probability(30d),
        #           onset_skewness, tempo_bucket(4d), hnr_mean, raga_missing

        logger.info("   Extracting vocal MFCC (from vocal stem)…")
        vocal_mfcc_dict = extract_vocal_mfcc(vocals_path)

        raga_missing = bool(indian_feat.get("raga_missing", False))

        # ── Steps 7–8: CLAP + lyrics IN PARALLEL ─────────────────────────────
        # CLAP is compute-bound (~10s); Genius is I/O-bound (~2–4s).
        # Running concurrently recovers the full Genius round-trip for free.
        logger.info("[6] CLAP embedding + lyrics fetch (parallel)…")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            clap_future   = pool.submit(get_clap_embedding, clip_path)
            lyrics_future = pool.submit(fetch_lyrics, title, artist)
            
            clap_emb = clap_future.result()    # ndarray (512,)
            
            # FIX 2: Handle the single string return safely
            raw_lyrics = lyrics_future.result()
            if not isinstance(raw_lyrics, str):
                raw_lyrics = ""
                
            lyrics = raw_lyrics
            lyrics_missing = (lyrics.strip() == "")

        if lyrics_missing:
            logger.info("   Lyrics not found — zero NLP vector, weights redistributed.")

        # ── Step 9: NLP embedding ─────────────────────────────────────────────
        logger.info("[7] NLP lyric embedding…")
        nlp_emb  = get_text_embedding(lyrics)   # (384,) — zeros if lyrics is None
        nlp_path = os.path.join(NLP_DIR, f"{song_id}.npy")
        save_nlp_embedding(nlp_emb, nlp_path)

        # ── Step 10: Delete all audio ─────────────────────────────────────────
        logger.info("[8] Deleting clip and stems…")
        cleanup(clip_path);  clip_path = None
        if stems:
            # Normal path: separate named stem files + nested demucs output dir
            cleanup(stems.get("vocals"))
            cleanup(stems.get("instrumental"))
            shutil.rmtree(stems.get("demucs_out_dir", ""), ignore_errors=True)
            stems = None
        # Fallback path: vocals_path == instr_path == clip_path (already deleted above)

        # ── Steps 11–12: Fuse + PCA ───────────────────────────────────────────
        logger.info("[9] Building fused vector…")

        # tonnetz_mean (6d) lives in base_feat; default to zeros if absent
        tonnetz = np.array(
            base_feat.get("tonnetz_mean", np.zeros(6)), dtype=np.float32
        )

        fused_vec = build_fused_vector(
            clap_emb               = clap_emb,
            nlp_emb                = nlp_emb,
            vocal_mfcc             = np.array(vocal_mfcc_dict["vocal_mfcc_mean"], dtype=np.float32),
            vocal_energy_ratio     = float(indian_feat["vocal_energy_ratio"]),
            murki_index            = float(indian_feat["murki_index"]),
            microtonal_pcp         = np.array(indian_feat["microtonal_pcp"],     dtype=np.float32),
            meend_variance         = float(indian_feat["meend_variance"]),
            raga_probability       = np.array(indian_feat["raga_probability"],   dtype=np.float32),
            onset_skewness         = float(indian_feat["onset_skewness"]),
            tempo_bucket           = tempo_bucket_arr,
            hnr_mean               = float(indian_feat["hnr_mean"]),
            tonnetz_mean           = tonnetz,                     # 6d — new
            lyrics_missing         = lyrics_missing,
            stem_separation_failed = stem_separation_failed,      # new
        )

        pca = _get_pca()
        if pca is not None:
            logger.info("[10] PCA compress → 256d…")
            final_vec      = compress_with_pca(fused_vec, pca)
            embedding_path = os.path.join(EMBEDDINGS_DIR, f"{song_id}.npy")
        else:
            logger.warning("[10] No PCA model — storing raw ~997d vector.")
            final_vec      = fused_vec
            embedding_path = os.path.join(EMBEDDINGS_RAW_DIR, f"{song_id}.npy")

        # ── Step 13a: Save embedding ──────────────────────────────────────────
        save_embedding(final_vec, embedding_path)

        # ── Step 13b: Save features JSON ──────────────────────────────────────
        features_path = os.path.join(FEATURES_DIR, f"{song_id}.json")
        full_features = {
            # Base spectral (serialise any ndarrays to lists)
            **{k: _to_list(v) if isinstance(v, np.ndarray) else v
               for k, v in base_feat.items()},
            # Indian features
            # FIX: Explicitly save mean and std separately
            "vocal_mfcc_mean":     _to_list(vocal_mfcc_dict["vocal_mfcc_mean"]),
            "vocal_mfcc_std":      _to_list(vocal_mfcc_dict["vocal_mfcc_std"]),

            "vocal_energy_ratio":  float(indian_feat["vocal_energy_ratio"]),
            "murki_index":         float(indian_feat["murki_index"]),
            "microtonal_pcp":      _to_list(indian_feat["microtonal_pcp"]),
            "meend_variance":      float(indian_feat["meend_variance"]),
            "raga_probability":    _to_list(indian_feat["raga_probability"]),
            "onset_skewness":      float(indian_feat["onset_skewness"]),
            "tempo_bucket":        _to_list(tempo_bucket_arr),
            "hnr_mean":            float(indian_feat["hnr_mean"]),
            # Quality flags
            "lyrics_missing":         int(lyrics_missing),
            "raga_missing":           int(raga_missing),
            "stem_separation_failed": int(stem_separation_failed),
            # Metadata
            "category":           category,
            "actual_clip_start":  actual_start,
        }
        save_features(full_features, features_path)

        # ── Step 13c: Insert to SQLite ────────────────────────────────────────
        insert_song(
            song_id                = song_id,
            title                  = title,
            artist                 = artist,
            youtube_url            = youtube_url or "",
            category               = category,
            tempo                  = float(base_feat.get("tempo", 0.0)),
            embedding_path         = embedding_path,
            features_path          = features_path,
            nlp_path               = nlp_path,
            lyrics_missing         = int(lyrics_missing),
            raga_missing           = int(raga_missing),
            stem_separation_failed = int(stem_separation_failed),
        )

        logger.info(
            "✓ Done: %s  (lyrics_missing=%s | raga_missing=%s | stem_failed=%s)\n",
            song_id, lyrics_missing, raga_missing, stem_separation_failed,
        )
        return song_id

    except Exception as exc:
        logger.error(
            "Pipeline failed for '%s — %s': %s", title, artist, exc, exc_info=True
        )
        # Emergency cleanup — never leave audio on disk
        for path in filter(None, [raw_path, clip_path]):
            cleanup(path)
        if stems:
            cleanup(stems.get("vocals"))
            cleanup(stems.get("instrumental"))
            shutil.rmtree(stems.get("demucs_out_dir", ""), ignore_errors=True)
        raise