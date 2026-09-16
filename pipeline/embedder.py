"""
pipeline/embedder.py

CLAP neural embedding + weighted feature fusion.

Implements the full fusion pipeline from FEATURES.md, updated per audit:

  Group 1  — CLAP neural embedding       (1024d, weight 0.30 base)
  NLP      — Lyric semantic embedding    (768d,  weight 0.15 base)
               NOTE: paraphrase-multilingual-mpnet-base-v2 outputs 768d,
               not the 384d some older docs/comments in this repo claim.
  Group 2  — Vocal / Gayaki              ( 22d,  weight 0.22 base)
               vocal_mfcc(20) + vocal_energy_ratio(1) + murki_index(1)
  Group 3  — Melodic                    ( 43d,  weight 0.20 base)
               microtonal_pcp(36) + meend_variance(1) + tonnetz_mean(6)
               NOTE: raga_probability(30) REMOVED 2026-09-17 — was cosine
               similarity against 30 hand-typed binary swara templates,
               never a real raga classifier, never validated. Cut rather
               than replaced with another unvalidated signal; see
               build_fused_vector()'s docstring and plan.md Phase 5 for
               real candidate replacements found but not yet integrated.
  Group 4  — Rhythmic / Taal            (  6d,  weight 0.13 base)
               onset_skewness(1) + tempo_bucket(4) + hnr_mean(1)
  ─────────────────────────────────────────────────────────────────
  Total raw: ~1863d (1024 CLAP + 768 NLP + 22 vocal + 43 melodic + 6 rhythmic).
  PCA compression is applied only once the corpus is large enough that
  fitting components isn't just memorising per-song idiosyncrasies — see
  MIN_SONGS_FOR_PCA in index/build_index.py. Below that threshold, the
  full-dimension L2-normalised vector is indexed directly.

Weight adjustments (per audit):
  - CLAP reduced 0.40 → 0.35  (over-dominates; misclassifies sufi as party)
  - Vocal increased 0.20 → 0.22  (murki + energy ratio are now discriminative)
  - Melodic increased 0.15 → 0.18  (raga PCP heuristic + tonnetz added)
  - NLP stays 0.15, Rhythmic stays 0.10

Flag-based redistributions:
  lyrics_missing=True  →  NLP weight redistributed (+0.08 CLAP, +0.04 Vocal, +0.03 Melodic)
  stem_separation_failed=True  →  Vocal/Rhythmic halved; freed weight moves to CLAP

AGENTS.md rules:
  - CLAP model loaded ONCE at module level (module-level singleton)
  - PCA model: transform() only — never fit() or fit_transform()
  - All groups L2-normalised independently before concatenation
"""

import os
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# msclap's CLAP(version="2023") outputs 1024-dim embeddings, not the 512d
# every doc/comment in this repo originally assumed (including this file's
# own module docstring below, now corrected). Verified empirically on
# 2026-09-15 during CLAP recovery — a real embedding was never actually
# produced before that point in this project's history, so the wrong
# assumption never got caught. Referenced from here rather than hardcoded
# elsewhere so this can't silently drift again.
CLAP_DIM = 1024


# ── Module-level CLAP singleton ───────────────────────────────────────────────
# AGENTS.md Rule 8: load once at module level; do NOT reload inside worker fns.
# torch CPU inference is GIL-safe → shared across ThreadPoolExecutor workers.

_clap_model = None
_torchaudio_patched = False


def _patch_torchaudio_load():
    """
    Replace torchaudio.load with a soundfile-based implementation.

    Newer torchaudio (2.x) defaults to a torchcodec backend for .load(),
    which dlopens an FFmpeg shared library by searching paths relative to
    the Python installation (Anaconda, in this environment) rather than
    wherever Homebrew actually put FFmpeg — a version/path mismatch that
    has nothing to do with this project's code and would likely recur
    differently on any other machine (Kaggle included).

    msclap's CLAPWrapper.read_audio() calls torchaudio.load(path) directly
    with no way to pass a backend argument, so the fix is applied here
    rather than in msclap. All audio this pipeline ever loads is plain WAV
    (yt-dlp + pydub output), which `soundfile` reads natively via libsndfile
    — no FFmpeg dependency at all, so this sidesteps the whole problem
    rather than chasing a correct dylib path on this specific machine.
    """
    global _torchaudio_patched
    if _torchaudio_patched:
        return
    import torchaudio
    import soundfile as sf

    def _sf_load(path, *args, **kwargs):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile: (frames, channels) → torchaudio contract: (channels, frames)
        waveform = torch.from_numpy(data.T.copy())
        return waveform, sr

    torchaudio.load = _sf_load
    _torchaudio_patched = True
    logger.info("Patched torchaudio.load → soundfile (bypasses torchcodec/FFmpeg dylib issue)")


def get_clap():
    """Return the cached CLAP model, loading it on first call."""
    global _clap_model
    if _clap_model is None:
        _patch_torchaudio_load()
        from msclap import CLAP
        logger.info("Loading CLAP model (first call — ~30s)…")
        _clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
        logger.info("CLAP model ready.")
    return _clap_model


# ── Group 1: CLAP audio embedding ────────────────────────────────────────────

def get_clap_embedding(audio_path: str) -> np.ndarray:
    """
    1024-dimensional CLAP embedding for an audio file.
    L2-normalised at output so inner product == cosine similarity downstream.
    """
    model      = get_clap()
    embeddings = model.get_audio_embeddings([audio_path])
    emb        = np.array(embeddings[0].detach().cpu(), dtype=np.float32)
    return _l2(emb)


def get_clap_text_query(text: str) -> np.ndarray:
    """
    1024-dimensional CLAP text embedding for conversational / vibe-based search.
    Used by the LLM orchestrator to translate natural-language prompts into
    the same vector space as the audio embeddings.
    """
    model      = get_clap()
    embeddings = model.get_text_embeddings([text])
    emb        = np.array(embeddings[0].detach().cpu(), dtype=np.float32)
    return _l2(emb)


# ── Fusion weight computation ─────────────────────────────────────────────────

# Base weights (audit-revised from FEATURES.md v1 defaults)
_BASE = dict(clap=0.30, nlp=0.15, vocal=0.22, melodic=0.20, rhythmic=0.13)

# Verify they sum to 1.0 at import time
assert abs(sum(_BASE.values()) - 1.0) < 1e-6, "Base weights must sum to 1.0"


def _redistribute(w: dict, zero_key: str) -> None:
    """
    Zero out w[zero_key] in place and redistribute its weight proportionally
    across every other group, preserving their relative ratios.

    Proportional (not fixed-constant) redistribution is what keeps this
    correct under composition: applying it for clap_missing and then again
    for lyrics_missing always sums to 1.0 regardless of which groups are
    already missing, because each step redistributes whatever weight
    currently exists rather than a value computed for one specific base case.
    An earlier version used fixed constants tuned for the "only NLP missing"
    case and silently under-redistributed by ~0.064 whenever CLAP was also
    missing (caught via the sanity-check warning below, which corrected the
    output but signalled the math itself was wrong).
    """
    freed = w[zero_key]
    w[zero_key] = 0.0
    targets = [k for k in w if k != zero_key]
    total = sum(w[k] for k in targets)
    if total <= 0:
        return
    for k in targets:
        w[k] += freed * (w[k] / total)


def _compute_weights(
    lyrics_missing: bool,
    stem_separation_failed: bool,
    clap_missing: bool = False,
) -> dict:
    """
    Return per-group scalar weights after applying flag-based redistributions.

    Redistribution rules (applied in order, each proportional — see
    _redistribute — so composition of any subset of flags still sums to 1.0):

    0. clap_missing=True
       CLAP embedding could not be recovered (e.g. rebuilding from a features
       JSON after the raw audio and raw fused vector are both gone — the CLAP
       component can only ever come from re-running audio through the model).
       Zero the CLAP weight, redistribute proportionally to the rest.

    1. stem_separation_failed=True
       Vocal and Rhythmic groups are based on separated stems. When separation
       failed, those features were extracted from the full mix and are
       degraded. Halve both weights; redistribute freed weight to CLAP (most
       reliable) — or to Vocal if CLAP is also missing, so the freed weight
       doesn't silently resurrect a zeroed-out group.

    2. lyrics_missing=True
       NLP weight cannot contribute; zero it and redistribute proportionally
       to whatever groups currently hold weight (reflects any adjustment
       already made by steps 0–1, unlike a fixed set of added constants).
    """
    w = dict(_BASE)   # copy

    # ── Step 0: missing CLAP ──────────────────────────────────────────────────
    if clap_missing:
        _redistribute(w, "clap")

    # ── Step 1: stem failure ──────────────────────────────────────────────────
    if stem_separation_failed:
        freed        = (w["vocal"] / 2) + (w["rhythmic"] / 2)
        w["vocal"]   = w["vocal"]   / 2
        w["rhythmic"] = w["rhythmic"] / 2
        target = "vocal" if clap_missing else "clap"
        w[target] += freed

    # ── Step 2: missing lyrics ────────────────────────────────────────────────
    if lyrics_missing:
        _redistribute(w, "nlp")

    # Sanity check (catches future bugs in redistribution arithmetic)
    total = sum(w.values())
    if abs(total - 1.0) > 1e-4:
        logger.warning("Weights sum to %.4f (expected 1.0) — normalising.", total)
        w = {k: v / total for k, v in w.items()}

    return w


# ── Primary fusion function ───────────────────────────────────────────────────

def build_fused_vector(
    clap_emb:              np.ndarray,        # (CLAP_DIM,) == (1024,)
    nlp_emb:               np.ndarray,        # (384,) — zeros if lyrics_missing
    vocal_mfcc:            np.ndarray,        # (20,)
    vocal_energy_ratio:    float,
    murki_index:           float,
    microtonal_pcp:        np.ndarray,        # (36,)
    meend_variance:        float,
    onset_skewness:        float,
    tempo_bucket:          np.ndarray,        # (4,)  one-hot
    hnr_mean:              float,
    tonnetz_mean:          np.ndarray = None, # (6,)  optional; zeros if not available
    lyrics_missing:        bool = False,
    stem_separation_failed: bool = False,
    clap_missing:          bool = False,
    raga_probability:      np.ndarray = None, # (30,) — DEPRECATED 2026-09-17, see note below; accepted but ignored, never required
) -> np.ndarray:
    """
    Assemble, weight, and L2-normalise the ~997d fused feature vector.

    The returned vector is L2-normalised and ready for either:
      (a) PCA compression via compress_with_pca()          [new songs, post-pilot]
      (b) Saving raw to data/embeddings_raw/ for PCA fit    [pilot batch]

    Args:
        clap_emb:               CLAP audio embedding, shape (CLAP_DIM,) == (1024,)
        nlp_emb:                Lyric NLP embedding, shape (384,); pass zero vector
                                when lyrics unavailable AND set lyrics_missing=True
        vocal_mfcc:             Mean MFCC over voiced frames, shape (20,)
        vocal_energy_ratio:     RMS(vocals) / RMS(instr)
        murki_index:            Band-power ratio of F0 derivative at 5–15 Hz
        microtonal_pcp:         36-bin pitch class profile, shape (36,)
        meend_variance:         Log-scaled mean absolute F0 derivative
        onset_skewness:         Skewness of onset strength envelope (instr stem)
        tempo_bucket:           One-hot tempo zone vector, shape (4,)
        hnr_mean:               Harmonic-to-noise ratio of instrumental stem
        tonnetz_mean:           Tonal centroid features, shape (6,); pass None
                                if not available (zeroed automatically)
        lyrics_missing:         True when Genius returned no lyrics
        stem_separation_failed: True when demucs failed; vocal/rhythmic features
                                were extracted from full mix (degraded quality)
        clap_missing:           True when no real CLAP embedding is available
                                (e.g. rebuilding from a features JSON after the
                                raw audio and raw fused vector are both gone).
                                Pass a zero (CLAP_DIM,) array for clap_emb in this
                                case — its weight is redistributed to 0 anyway.
        raga_probability:       DEPRECATED 2026-09-17 — accepted for backward
                                compatibility with existing callers but IGNORED,
                                not included in the fused vector. The original
                                implementation was cosine similarity against 30
                                hand-typed binary swara templates — never a real
                                raga classifier, never validated, and this was
                                real weight in the vector for a fabricated
                                signal. Cut rather than replaced: research into
                                real alternatives (2026-09-17) found a
                                pretrained Hindustani model (E2ERaga, real
                                weights, but no disclosed accuracy/raga-list/
                                license, old Python 3.6 pin — needs validation
                                before trusting) and real raga names embedded in
                                Wikipedia soundtrack-album prose (confirmed for
                                at least one song, but not a clean structured
                                Wikidata property — would need a real scraper,
                                not yet built). Neither was safe to ship without
                                validation; see plan.md Phase 5 for both leads.

    Returns:
        np.ndarray, shape (~1863,), dtype float32, L2-normalised
    """
    w = _compute_weights(lyrics_missing, stem_separation_failed, clap_missing)

    # ── Handle optional tonnetz ───────────────────────────────────────────────
    _tonnetz = (
        np.asarray(tonnetz_mean, dtype=np.float32)
        if tonnetz_mean is not None
        else np.zeros(6, dtype=np.float32)
    )

    # ── Assemble group sub-vectors ────────────────────────────────────────────

    # Group 1: CLAP (CLAP_DIM = 1024d)
    g_clap = _l2(np.asarray(clap_emb, dtype=np.float32)) * w["clap"]

    # NLP (384d) — zero vector already passed in when lyrics_missing
    g_nlp = _l2(np.asarray(nlp_emb, dtype=np.float32)) * w["nlp"]

    # Group 2: Vocal (22d)
    vocal_group = np.concatenate([
        np.asarray(vocal_mfcc,          dtype=np.float32),   # 20d
        np.array([vocal_energy_ratio],  dtype=np.float32),   #  1d
        np.array([murki_index],         dtype=np.float32),   #  1d
    ])
    g_vocal = _l2(vocal_group) * w["vocal"]

    # Group 3: Melodic + Tonnetz (36 + 1 + 6 = 43d). raga_probability
    # deliberately excluded — see the deprecation note in this function's
    # docstring. Group renamed conceptually from "Melodic/Raga" to just
    # "Melodic" but the dict key ("melodic") is kept as-is to avoid
    # touching _BASE/_compute_weights and every caller unnecessarily.
    melodic_group = np.concatenate([
        np.asarray(microtonal_pcp,    dtype=np.float32),     # 36d
        np.array([meend_variance],    dtype=np.float32),     #  1d
        _tonnetz,                                             #  6d
    ])
    g_melodic = _l2(melodic_group) * w["melodic"]

    # Group 4: Rhythmic (1 + 4 + 1 = 6d)
    rhythmic_group = np.concatenate([
        np.array([onset_skewness],    dtype=np.float32),     #  1d
        np.asarray(tempo_bucket,      dtype=np.float32),     #  4d
        np.array([hnr_mean],          dtype=np.float32),     #  1d
    ])
    g_rhythmic = _l2(rhythmic_group) * w["rhythmic"]

    # ── Concatenate → final L2-normalise ─────────────────────────────────────
    # Order must be consistent across all songs — never change without
    # deleting all embeddings and refitting PCA from scratch.
    fused = np.concatenate([
        g_clap,       # CLAP_DIM (1024d)
        g_nlp,        # 384d
        g_vocal,      #  22d
        g_melodic,    #  43d
        g_rhythmic,   #   6d
    ]).astype(np.float32)   # total: ~1863d

    return _l2(fused)


# ── PCA compression ───────────────────────────────────────────────────────────

def compress_with_pca(fused_vector: np.ndarray, pca_model) -> np.ndarray:
    """
    Compress a ~997d fused vector to 256d using the frozen PCA model.

    AGENTS.md Rule 4: ONLY pca_model.transform() — never fit() or fit_transform().
    Refitting shifts the entire embedding space and invalidates the FAISS index.

    Args:
        fused_vector: shape (~997,) output of build_fused_vector()
        pca_model:    sklearn PCA fitted on the pilot dataset

    Returns:
        np.ndarray shape (n_components,), L2-normalised
    """
    compressed = pca_model.transform(fused_vector.reshape(1, -1))[0]
    return _l2(compressed.astype(np.float32))


# ── Rocchio query adjustment ──────────────────────────────────────────────────

def rocchio_adjust(
    query_vec:     np.ndarray,
    approved_vecs: list,
    rejected_vecs: list,
    alpha: float = 1.0,
    beta:  float = 0.75,
    gamma: float = 0.25,
) -> np.ndarray:
    """
    Rocchio-style query vector adjustment for the Tick/Cross feedback loop.

        new_query = α·query + β·mean(approved) − γ·mean(rejected)

    All vectors must already be in the PCA-compressed space (256d).
    Returns an L2-normalised adjusted query vector.

    Args:
        query_vec:     Current query vector (256d)
        approved_vecs: List of ✅ song vectors from this session
        rejected_vecs: List of ❌ song vectors from this session
        alpha, beta, gamma: Rocchio coefficients (AGENTS.md defaults)

    AGENTS.md Rule 2: this is the ONLY permitted feedback mechanism.
    Do NOT retrain CLAP, fine-tune embeddings, or modify the FAISS index.
    """
    new_query = alpha * query_vec.copy().astype(np.float32)

    if approved_vecs:
        new_query += beta * np.mean(np.stack(approved_vecs), axis=0)

    if rejected_vecs:
        new_query -= gamma * np.mean(np.stack(rejected_vecs), axis=0)

    return _l2(new_query)


# ── Playlist centroid ─────────────────────────────────────────────────────────

def playlist_centroid(song_vectors: list) -> np.ndarray:
    """
    Mean embedding of a playlist for playlist expansion queries.
    All input vectors must be in the same PCA-compressed space (256d).
    Returns an L2-normalised centroid vector.
    """
    if not song_vectors:
        raise ValueError("playlist_centroid requires at least one song vector.")
    centroid = np.mean(np.stack(song_vectors), axis=0)
    return _l2(centroid.astype(np.float32))


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_embedding(embedding: np.ndarray, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    np.save(path, embedding)
    logger.debug("Embedding saved → %s", path)


def load_embedding(path: str) -> np.ndarray:
    return np.load(path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector. Returns the original if the norm is near zero."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v