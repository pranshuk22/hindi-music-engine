"""
scripts/recover_clap.py
─────────────────────────
Recover real CLAP embeddings for songs whose raw fused vector was
destroyed by the pre-fix PCA bug (see plan.md Phase 0 / experiment_log.md).

This is a LEAN recovery, not a full pipeline re-run: Demucs stem separation,
handcrafted feature extraction, and lyrics/NLP embedding are all already
correct and intact in data/features/*.json and data/nlp/*.npy — only the
CLAP component needs real audio again. So per song this does:

  download (yt-dlp) → trim to the ORIGINAL clip window (actual_clip_start
  from the features JSON, so the CLAP embedding covers the same audio the
  other features were computed from) → CLAP-embed → delete audio →
  rebuild the fused vector with the real clap_emb (clap_missing=False) →
  overwrite data/embeddings_raw/<song_id>.npy → mark clap_recovered=1 in
  the features JSON (for resumability — reruns skip already-recovered songs
  unless --force).

No Demucs, no lyrics re-fetch — those are unchanged and don't need redoing.

After all songs are processed, re-finalizes embeddings (data/embeddings/)
and rebuilds the FAISS index, same as index/build_index.py --from-features
would, but starting from the now-real-CLAP raw vectors instead of zeros.

Usage:
  python scripts/recover_clap.py
  python scripts/recover_clap.py --force        # redo even already-recovered songs
  python scripts/recover_clap.py --limit 5       # smoke-test on a few songs first
"""

import sys
import os
import json
import time
import argparse
import logging

sys.path.insert(0, ".")

from utils.db import init_db, get_all_songs, update_embedding_path
from pipeline.downloader import download_audio, search_and_download
from pipeline.trimmer import trim_audio, cleanup
from pipeline.embedder import get_clap_embedding, build_fused_vector, save_embedding
from pipeline.feature_extractor import get_tempo_bucket
from index.build_index import finalize_embeddings, build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TEMP_DIR = "/tmp/hindi_music_clap_recovery"
os.makedirs(TEMP_DIR, exist_ok=True)


def _load_features(features_path: str) -> dict:
    with open(features_path) as f:
        return json.load(f)


def _save_features(features: dict, features_path: str) -> None:
    with open(features_path, "w") as f:
        json.dump(features, f, indent=2)


RAW_DIR = "data/embeddings_raw"


def recover_one(row) -> tuple[bool, str]:
    """Returns (success, message)."""
    song_id = row["id"]
    features_path = row["features_path"]
    nlp_path = row["nlp_path"]

    # Always write recovered raw vectors to a dedicated data/embeddings_raw/
    # path, NEVER to row["embedding_path"] directly. Bug found 2026-09-15:
    # after a corpus has been finalized once (index/build_index.py's
    # finalize_embeddings), embedding_path points at data/embeddings/ (the
    # FINAL store), not the raw one — so writing there silently let the raw
    # archive go stale (still 1381d/zero-CLAP from before CLAP_DIM was
    # fixed) even though search correctness was unaffected. This defeats
    # the entire point of the Phase 0 fix that stopped raw vectors from
    # being destroyed. Recovery always targets the raw path explicitly;
    # finalize_embeddings() is then responsible for propagating it to the
    # final store, exactly as originally designed.
    embedding_path = os.path.join(RAW_DIR, f"{song_id}.npy")

    if not features_path or not os.path.exists(features_path):
        return False, "features JSON missing"

    feat = _load_features(features_path)
    clip_start = int(feat.get("actual_clip_start", 15))

    raw_path = None
    clip_path = os.path.join(TEMP_DIR, f"{song_id}_clip.wav")
    try:
        # ── Download (reuse youtube_url if present, else search) ────────────
        youtube_url = row["youtube_url"] or None
        if youtube_url:
            raw_path = download_audio(youtube_url, TEMP_DIR)
        else:
            raw_path = search_and_download(row["title"], row["artist"], TEMP_DIR)

        # ── Trim to the SAME window the other features were computed from ───
        trim_audio(raw_path, clip_path, duration_sec=45, force_start_sec=clip_start)
        cleanup(raw_path)
        raw_path = None

        # ── Real CLAP embedding ───────────────────────────────────────────────
        clap_emb = get_clap_embedding(clip_path)
        cleanup(clip_path)
        clip_path = None

        # ── Load existing (intact) NLP + handcrafted features ────────────────
        lyrics_missing = bool(feat.get("lyrics_missing", 0))
        if nlp_path and os.path.exists(nlp_path):
            import numpy as np
            nlp_emb = np.load(nlp_path).astype("float32")
        else:
            import numpy as np
            nlp_emb = np.zeros(768, dtype="float32")
            lyrics_missing = True

        import numpy as np
        vocal_mfcc = np.array(feat.get("vocal_mfcc_mean", [0.0] * 20), dtype=np.float32)
        vocal_energy_ratio = float(feat.get("vocal_energy_ratio", 0.5))
        murki_index = float(feat.get("murki_index", 0.0))
        microtonal_pcp = np.array(feat.get("microtonal_pcp", [0.0] * 36), dtype=np.float32)
        meend_variance = float(feat.get("meend_variance", 0.0))
        raga_probability = np.array(feat.get("raga_probability", [0.0] * 30), dtype=np.float32)
        onset_skewness = float(feat.get("onset_skewness", 0.0))
        hnr_mean = float(feat.get("hnr_mean", 0.5))
        tonnetz = np.array(feat.get("tonnetz_mean", [0.0] * 6), dtype=np.float32)
        stem_separation_failed = bool(feat.get("stem_separation_failed", 0))

        tb_raw = feat.get("tempo_bucket")
        if tb_raw is not None and len(tb_raw) == 4:
            tempo_bucket = np.array(tb_raw, dtype=np.float32)
        else:
            tempo_bucket = get_tempo_bucket(float(feat.get("tempo", 100.0)))

        # ── Rebuild fused vector WITH real CLAP ──────────────────────────────
        fused = build_fused_vector(
            clap_emb=clap_emb,
            nlp_emb=nlp_emb,
            vocal_mfcc=vocal_mfcc,
            vocal_energy_ratio=vocal_energy_ratio,
            murki_index=murki_index,
            microtonal_pcp=microtonal_pcp,
            meend_variance=meend_variance,
            raga_probability=raga_probability,
            onset_skewness=onset_skewness,
            tempo_bucket=tempo_bucket,
            hnr_mean=hnr_mean,
            tonnetz_mean=tonnetz,
            lyrics_missing=lyrics_missing,
            stem_separation_failed=stem_separation_failed,
            clap_missing=False,
        )

        save_embedding(fused, embedding_path)

        feat["clap_recovered"] = 1
        _save_features(feat, features_path)

        return True, f"{fused.shape[0]}d"

    except Exception as exc:
        return False, str(exc)
    finally:
        cleanup(raw_path)
        cleanup(clip_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Redo even already-recovered songs")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N songs (smoke test)")
    args = parser.parse_args()

    init_db()
    songs = get_all_songs()
    if args.limit:
        songs = songs[: args.limit]

    log.info(f"Found {len(songs)} songs.")

    recovered, skipped, failed = 0, 0, []

    for i, row in enumerate(songs, 1):
        song_id = row["id"]
        features_path = row["features_path"]

        if not args.force and features_path and os.path.exists(features_path):
            feat = _load_features(features_path)
            if feat.get("clap_recovered"):
                skipped += 1
                log.info(f"[{i}/{len(songs)}] SKIP (already recovered): {song_id}")
                continue

        log.info(f"[{i}/{len(songs)}] Recovering: {row['title']} — {row['artist']}")
        ok, msg = recover_one(row)
        if ok:
            recovered += 1
            log.info(f"  ✓ {song_id}  ({msg})")
        else:
            failed.append((song_id, msg))
            log.error(f"  ✗ {song_id}: {msg}")

        time.sleep(1)  # polite pause between yt-dlp requests

    log.info(f"\nRecovered: {recovered}  |  Skipped (already done): {skipped}  |  Failed: {len(failed)}")
    if failed:
        log.warning("Failed songs:")
        for sid, msg in failed:
            log.warning(f"  {sid}: {msg}")

    if recovered > 0:
        log.info("\nFinalizing embeddings and rebuilding FAISS index...")
        songs = get_all_songs()
        finalize_embeddings(songs, force_pca=False)
        songs = get_all_songs()
        build_index(songs, already_finalized=True)
        log.info("Done. Run scripts/evaluate_golden.py to check the effect.")


if __name__ == "__main__":
    main()
