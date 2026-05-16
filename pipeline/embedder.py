"""
pipeline/embedder.py

CLAP neural embedding + weighted feature fusion.

Implements the full fusion pipeline from FEATURES.md, updated per audit:

  Group 1  — CLAP neural embedding       (512d,  weight 0.35)
  NLP      — Lyric semantic embedding    (384d,  weight 0.15)
  Group 2  — Vocal / Gayaki              ( 22d,  weight 0.22)
               vocal_mfcc(20) + vocal_energy_ratio(1) + murki_index(1)
  Group 3  — Melodic / Raga             ( 73d,  weight 0.18)
               microtonal_pcp(36) + meend_variance(1) + raga_probability(30) + tonnetz_mean(6)
  Group 4  — Rhythmic / Taal            (  6d,  weight 0.10)
               onset_skewness(1) + tempo_bucket(4) + hnr_mean(1)
  ─────────────────────────────────────────────────────────────────
  Total raw: ~997d  →  PCA  →  256d (fit once in build_index.py)

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


# ── Module-level CLAP singleton ───────────────────────────────────────────────
# AGENTS.md Rule 8: load once at module level; do NOT reload inside worker fns.
# torch CPU inference is GIL-safe → shared across ThreadPoolExecutor workers.

_clap_model = None


def get_clap():
    """Return the cached CLAP model, loading it on first call."""
    global _clap_model
    if _clap_model is None:
        from msclap import CLAP
        logger.info("Loading CLAP model (first call — ~30s)…")
        _clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
        logger.info("CLAP model ready.")
    return _clap_model


# ── Group 1: CLAP audio embedding ────────────────────────────────────────────

def get_clap_embedding(audio_path: str) -> np.ndarray:
    """
    512-dimensional CLAP embedding for an audio file.
    L2-normalised at output so inner product == cosine similarity downstream.
    """
    model      = get_clap()
    embeddings = model.get_audio_embeddings([audio_path])
    emb        = np.array(embeddings[0], dtype=np.float32)
    return _l2(emb)


def get_clap_text_query(text: str) -> np.ndarray:
    """
    512-dimensional CLAP text embedding for conversational / vibe-based search.
    Used by the LLM orchestrator to translate natural-language prompts into
    the same vector space as the audio embeddings.
    """
    model      = get_clap()
    embeddings = model.get_text_embeddings([text])
    emb        = np.array(embeddings[0], dtype=np.float32)
    return _l2(emb)


# ── Fusion weight computation ─────────────────────────────────────────────────

# Base weights (audit-revised from FEATURES.md v1 defaults)
_BASE = dict(clap=0.30, nlp=0.15, vocal=0.22, melodic=0.20, rhythmic=0.13)

# Verify they sum to 1.0 at import time
assert abs(sum(_BASE.values()) - 1.0) < 1e-6, "Base weights must sum to 1.0"


def _compute_weights(lyrics_missing: bool, stem_separation_failed: bool) -> dict:
    """
    Return per-group scalar weights after applying flag-based redistributions.

    Redistribution rules (applied in order):

    1. stem_separation_failed=True
       Vocal and Rhythmic groups are based on separated stems.  When separation
       failed, those features were extracted from the full mix and are degraded.
       Halve both weights; redistribute freed weight to CLAP (most reliable).
         Vocal:    0.22 → 0.11   (−0.11)
         Rhythmic: 0.10 → 0.05   (−0.05)
         CLAP:     0.35 → 0.51   (+0.16)

    2. lyrics_missing=True
       NLP weight cannot contribute; redistribute to other groups.
       Applied on top of any stem adjustment already made.
         NLP:     W_nlp → 0.00
         CLAP:    += 0.08
         Vocal:   += 0.04
         Melodic: += 0.03

    Combined (both flags):
         CLAP=0.59, NLP=0.00, Vocal=0.15, Melodic=0.21, Rhythmic=0.05  (sums to 1.0)
    """
    w = dict(_BASE)   # copy

    # ── Step 1: stem failure ──────────────────────────────────────────────────
    if stem_separation_failed:
        freed       = (w["vocal"] / 2) + (w["rhythmic"] / 2)
        w["vocal"]   = w["vocal"]   / 2
        w["rhythmic"] = w["rhythmic"] / 2
        w["clap"]   += freed

    # ── Step 2: missing lyrics ────────────────────────────────────────────────
    if lyrics_missing:
        w["clap"]    += 0.08
        w["vocal"]   += 0.04
        w["melodic"] += 0.03
        w["nlp"]      = 0.0

    # Sanity check (catches future bugs in redistribution arithmetic)
    total = sum(w.values())
    if abs(total - 1.0) > 1e-4:
        logger.warning("Weights sum to %.4f (expected 1.0) — normalising.", total)
        w = {k: v / total for k, v in w.items()}

    return w


# ── Primary fusion function ───────────────────────────────────────────────────

def build_fused_vector(
    clap_emb:              np.ndarray,        # (512,)
    nlp_emb:               np.ndarray,        # (384,) — zeros if lyrics_missing
    vocal_mfcc:            np.ndarray,        # (20,)
    vocal_energy_ratio:    float,
    murki_index:           float,
    microtonal_pcp:        np.ndarray,        # (36,)
    meend_variance:        float,
    raga_probability:      np.ndarray,        # (30,)
    onset_skewness:        float,
    tempo_bucket:          np.ndarray,        # (4,)  one-hot
    hnr_mean:              float,
    tonnetz_mean:          np.ndarray = None, # (6,)  optional; zeros if not available
    lyrics_missing:        bool = False,
    stem_separation_failed: bool = False,
) -> np.ndarray:
    """
    Assemble, weight, and L2-normalise the ~997d fused feature vector.

    The returned vector is L2-normalised and ready for either:
      (a) PCA compression via compress_with_pca()          [new songs, post-pilot]
      (b) Saving raw to data/embeddings_raw/ for PCA fit    [pilot batch]

    Args:
        clap_emb:               CLAP audio embedding, shape (512,)
        nlp_emb:                Lyric NLP embedding, shape (384,); pass zero vector
                                when lyrics unavailable AND set lyrics_missing=True
        vocal_mfcc:             Mean MFCC over voiced frames, shape (20,)
        vocal_energy_ratio:     RMS(vocals) / RMS(instr)
        murki_index:            Band-power ratio of F0 derivative at 5–15 Hz
        microtonal_pcp:         36-bin pitch class profile, shape (36,)
        meend_variance:         Log-scaled mean absolute F0 derivative
        raga_probability:       30-class raga similarity vector, shape (30,)
        onset_skewness:         Skewness of onset strength envelope (instr stem)
        tempo_bucket:           One-hot tempo zone vector, shape (4,)
        hnr_mean:               Harmonic-to-noise ratio of instrumental stem
        tonnetz_mean:           Tonal centroid features, shape (6,); pass None
                                if not available (zeroed automatically)
        lyrics_missing:         True when Genius returned no lyrics
        stem_separation_failed: True when demucs failed; vocal/rhythmic features
                                were extracted from full mix (degraded quality)

    Returns:
        np.ndarray, shape (~997,), dtype float32, L2-normalised
    """
    w = _compute_weights(lyrics_missing, stem_separation_failed)

    # ── Handle optional tonnetz ───────────────────────────────────────────────
    _tonnetz = (
        np.asarray(tonnetz_mean, dtype=np.float32)
        if tonnetz_mean is not None
        else np.zeros(6, dtype=np.float32)
    )

    # ── Assemble group sub-vectors ────────────────────────────────────────────

    # Group 1: CLAP (512d)
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

    # Group 3: Melodic + Raga + Tonnetz (36 + 1 + 30 + 6 = 73d)
    melodic_group = np.concatenate([
        np.asarray(microtonal_pcp,    dtype=np.float32),     # 36d
        np.array([meend_variance],    dtype=np.float32),     #  1d
        np.asarray(raga_probability,  dtype=np.float32),     # 30d
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
        g_clap,       # 512d
        g_nlp,        # 384d
        g_vocal,      #  22d
        g_melodic,    #  73d
        g_rhythmic,   #   6d
    ]).astype(np.float32)   # total: ~997d

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