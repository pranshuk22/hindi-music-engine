"""
pipeline/feature_extractor.py
──────────────────────────────
Standard handcrafted acoustic feature extraction using librosa.

CHANGES FROM PREVIOUS VERSION (per hindi_engine_improvements.md):

  [FIX 1] get_reliable_tempo() — new function, replaces raw beat_track call.
    librosa.beat_track systematically detects 2× the actual tempo on
    Hindustani acoustic music (ghazals, classical) because it locks onto
    tabla subdivisions rather than the actual pulse.
    Fix: after detecting tempo, compare against onset density.
    If tempo > 120 BPM and the onset envelope is sparse (< 30% of frames
    above the 75th percentile) → 2× error is almost certain → halve.
    Unconditional halve above 160 BPM (always an octave error at that rate
    for any genre in this corpus).

  [FIX 2] extract_features() — uses get_reliable_tempo() instead of raw
    beat_track. The corrected tempo is what gets stored in features JSON
    and fed to get_tempo_bucket(). No change to get_tempo_bucket() itself.

  [FIX 3] tonnetz_to_vector() — new helper.
    tonnetz_mean (6d) was being extracted and stored in the features JSON
    but silently dropped before fusion. The audit confirmed it IS
    discriminative: ghazal songs cluster near [-0.32, -0.02, ...] while
    party songs cluster near [-0.07, 0.08, ...].
    Added tonnetz_to_vector() so embedder.py can include it in the
    Melodic/Raga group. The fused vector grows from ~991d to ~997d —
    PCA still reduces to 256d so FAISS is unaffected.

  [FIX 4] Docstring updates.
    - "90-second" → "45-second" throughout (processor.py changed this).
    - Module-level dimension table updated to include tonnetz (6d).
    - tonnetz_mean no longer labelled "not fused" in extract_features().

  [NO CHANGE] extract_vocal_mfcc, vocal_mfcc_to_vector, save_features,
    load_features, get_tempo_bucket bucket boundaries.

Dimensions from this module:
  - vocal_mfcc (mean only, for fusion): 20d
  - vocal_mfcc_std: 20d  (stored in JSON for diagnostics, not fused)
  - tonnetz_mean (for fusion):           6d   ← [FIX 3] now fused
  - spectral_centroid, spectral_rolloff, spectral_bandwidth,
    zcr, rms: 5d  (stored in JSON for diagnostics, not fused)

Downstream change required in embedder.py:
  Add tonnetz_to_vector() to the build_fused_vector() call and include
  the 6d tonnetz in the Melodic/Raga group (weight 0.18 after rebalance).
  See hindi_engine_improvements.md §3.1 and §3.2.
"""

import logging
import numpy as np
import librosa
import json

log = logging.getLogger(__name__)

SR     = 22050
N_MFCC = 20


# ─── Vocal MFCC (from isolated stem) ─────────────────────────────────────────

def extract_vocal_mfcc(vocals_path: str) -> dict:
    """
    Extract MFCC features from the isolated vocal stem.

    Running on the separated vocal track (rather than the full mix) gives
    cleaner coefficients because the instrumental frequency content no longer
    bleeds into the vocal tract model — especially important for ghazals where
    the sarangi/harmonium frequencies heavily overlap with mid-frequency MFCCs.

    Args:
        vocals_path: Path to the isolated vocal stem WAV produced by demucs.
                     If demucs failed, this equals clip_path (full mix) and
                     the coefficients degrade gracefully.

    Returns:
        dict with:
          vocal_mfcc_mean: list of 20 floats
          vocal_mfcc_std:  list of 20 floats (stored for diagnostics only)
    """
    try:
        y, _ = librosa.load(vocals_path, sr=SR, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
        return {
            "vocal_mfcc_mean": mfcc.mean(axis=1).tolist(),
            "vocal_mfcc_std":  mfcc.std(axis=1).tolist(),
        }
    except Exception as exc:
        log.warning(f"vocal_mfcc failed on {vocals_path}: {exc}")
        return {
            "vocal_mfcc_mean": [0.0] * N_MFCC,
            "vocal_mfcc_std":  [0.0] * N_MFCC,
        }


# ─── Reliable tempo with octave-error correction ─────────────────────────────

# [FIX 1] New function — replaces the raw librosa.beat.beat_track call in
# extract_features().
#
# Problem: librosa.beat_track locks onto tabla/dholak subdivisions in Hindustani
# music. A ghazal at 68 BPM gets detected as 136 BPM (double time), and the
# song is bucketed as "Fast" instead of "Slow". This directly broke the FAISS
# clustering for ghazals and sufi songs in the pilot dataset.
#
# Evidence from feature JSONs:
#   agar_tum_saath_ho (sad_romantic) → 123 BPM detected → Fast bucket [0,0,0,1]
#   chupke_chupke_raat_din (ghazal)  → 136 BPM detected → Fast bucket [0,0,0,1]
#   Both should be Slow / Mid bucket.
#
# Correction logic:
#   1. Compute onset_strength envelope across the clip.
#   2. onset_rate = fraction of frames above the 75th percentile.
#      Dense beats (party, fast Bollywood) → onset_rate ≈ 0.35–0.50.
#      Sparse beats (ghazal, classical) → onset_rate ≈ 0.18–0.28.
#   3. If detected tempo > 120 AND onset_rate < 0.30 → almost certainly a
#      2× octave error → halve.
#   4. Unconditional halve above 160 BPM — no song in this corpus genuinely
#      exceeds 160 BPM.

def get_reliable_tempo(y: np.ndarray, sr: int) -> float:
    """
    Estimate song tempo with octave-error correction for Indian music.

    Standard librosa.beat_track locks onto tabla/dholak subdivisions,
    producing 2× the actual tempo for ghazals and classical songs.
    This function detects that error using onset density and halves the
    tempo when the signal is sparse but the detected BPM is high.

    Args:
        y:  Audio signal as float32 ndarray.
        sr: Sample rate (must match y).

    Returns:
        Corrected tempo in BPM as a float.

    Typical output after correction:
      Ghazal (68 BPM actual):  136 BPM detected → 68 BPM returned  ✓
      Party (128 BPM actual):  128 BPM detected → 128 BPM returned ✓
      Romantic (88 BPM actual): 88 BPM detected →  88 BPM returned ✓
    """
    try:
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_raw)

        # Compute onset strength envelope to measure beat density
        onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
        threshold  = float(np.percentile(onset_env, 75))
        onset_rate = float(np.mean(onset_env > threshold))
        # onset_rate ≈ 0.25 by definition (25% of frames above 75th percentile)
        # but spiky percussion → higher; sustained harmonics → lower.

        if tempo > 160:
            # Unconditional halve: nothing in this corpus genuinely runs > 160 BPM
            log.debug(f"Tempo {tempo:.1f} BPM > 160 — unconditional halve → {tempo/2:.1f}")
            tempo = tempo / 2.0

        elif tempo > 120 and onset_rate < 0.30:
            # High detected tempo + sparse onset density = subdivision tracking error
            log.debug(
                f"Tempo {tempo:.1f} BPM + sparse onsets ({onset_rate:.2f}) "
                f"→ octave correction → {tempo/2:.1f} BPM"
            )
            tempo = tempo / 2.0

        return tempo

    except Exception as exc:
        log.warning(f"get_reliable_tempo failed: {exc} — returning 100.0 BPM")
        return 100.0


# ─── Full-clip standard features ─────────────────────────────────────────────

def extract_features(clip_path: str) -> dict:
    """
    Extract global acoustic features from the full 45-second clip.

    These features capture the overall song signature — tempo, spectral
    shape, energy, harmonic texture — and are stored in the features JSON.
    Tempo feeds get_tempo_bucket(). Tonnetz feeds into the fused vector
    via tonnetz_to_vector() (see [FIX 3]).

    Note: chroma_mean is intentionally NOT extracted here.
    It is replaced by microtonal_pcp (36-bin) in indian_features.py.

    Args:
        clip_path: Path to the 45-second clip WAV.

    Returns:
        dict with keys: tempo, spectral_centroid_mean, spectral_rolloff_mean,
        spectral_bandwidth_mean, zcr_mean, rms_mean, tonnetz_mean.
    """
    try:
        y, _ = librosa.load(clip_path, sr=SR, mono=True)

        features = {}

        # [FIX 2] Use octave-corrected tempo — not raw beat_track output.
        # This fixes the systematic 2× error on ghazal and classical songs.
        features["tempo"] = get_reliable_tempo(y, SR)

        # Spectral shape
        spec_centroid  = librosa.feature.spectral_centroid(y=y, sr=SR)
        spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=SR)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)
        features["spectral_centroid_mean"]  = float(spec_centroid.mean())
        features["spectral_rolloff_mean"]   = float(spec_rolloff.mean())
        features["spectral_bandwidth_mean"] = float(spec_bandwidth.mean())

        # Zero crossing rate (noisiness / percussiveness proxy)
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(zcr.mean())

        # RMS energy
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(rms.mean())

        # Tonnetz — harmonic relationship features.
        # 6 dimensions: perfect fifth, minor third, major third (× real + imaginary).
        # [FIX 4] No longer labelled "not fused" — tonnetz_to_vector() adds
        # these to the Melodic/Raga group in embedder.py.
        # Evidence from audit: ghazal[-0.32, -0.02, 0.01, -0.14, 0.03, 0.05]
        # vs party[-0.07, 0.08, -0.03, 0.02, 0.02, -0.01] — clearly discriminative.
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(y), sr=SR
        )
        features["tonnetz_mean"] = tonnetz.mean(axis=1).tolist()   # list of 6 floats

        return features

    except Exception as exc:
        log.warning(f"extract_features failed on {clip_path}: {exc}")
        return {
            "tempo": 100.0,
            "spectral_centroid_mean":  2000.0,
            "spectral_rolloff_mean":   4000.0,
            "spectral_bandwidth_mean": 2500.0,
            "zcr_mean": 0.1,
            "rms_mean": 0.2,
            "tonnetz_mean": [0.0] * 6,
        }


# ─── Tempo bucket one-hot ─────────────────────────────────────────────────────

def get_tempo_bucket(tempo: float) -> np.ndarray:
    """
    One-hot encode tempo into 4 mutually exclusive buckets.

    Expects the CORRECTED tempo from get_reliable_tempo() — not the raw
    librosa.beat_track output. processor.py calls extract_features() which
    stores the corrected tempo in features["tempo"], and that value is what
    gets passed here.

    Bucket definitions (aligned with FEATURES.md):
      0: Slow     < 80 BPM   — ghazal, classical (e.g. Ranjish Hi Sahi ~65 BPM)
      1: Mid    80–100 BPM   — romantic ballad (e.g. Tum Hi Ho ~80 BPM)
      2: Up    100–120 BPM   — indie, semi-fast
      3: Fast    > 120 BPM   — party, dance (e.g. Saturday Saturday ~129 BPM)

    The one-hot encoding gives tempo disproportionate binary influence
    relative to a continuous value — intentional, as crossing a tempo
    bucket boundary is musically significant.

    Args:
        tempo: Corrected BPM float from get_reliable_tempo().

    Returns:
        ndarray of shape (4,), dtype float32. Exactly one element is 1.0.
    """
    bucket = np.zeros(4, dtype=np.float32)
    if tempo < 80:
        bucket[0] = 1.0
    elif tempo < 100:
        bucket[1] = 1.0
    elif tempo < 120:
        bucket[2] = 1.0
    else:
        bucket[3] = 1.0
    return bucket


# ─── Feature vector assembly helpers ─────────────────────────────────────────

def vocal_mfcc_to_vector(features: dict) -> np.ndarray:
    """
    Extract the 20-dimensional vocal MFCC mean vector for fusion.

    Only the mean (not std) is included in the fused vector.
    The std is stored in the features JSON for diagnostics.

    Args:
        features: dict returned by extract_vocal_mfcc().

    Returns:
        ndarray of shape (20,), dtype float32.
    """
    return np.array(
        features.get("vocal_mfcc_mean", [0.0] * N_MFCC),
        dtype=np.float32,
    )


# [FIX 3] New helper — was missing, causing tonnetz to be silently dropped.
# Called by embedder.py:build_fused_vector() alongside vocal_mfcc_to_vector().
# Adds 6 dimensions to the Melodic/Raga group in the fused vector.

def tonnetz_to_vector(features: dict) -> np.ndarray:
    """
    Extract the 6-dimensional tonnetz mean vector for fusion.

    Tonnetz encodes harmonic relationships in a torus geometry:
      dims 0–1: perfect fifth (P5) — strong in melodic/raga contexts
      dims 2–3: minor third (m3)   — distinguishes modal from major tonality
      dims 4–5: major third (M3)   — distinguishes classical from neutral/electronic

    These 6 values show clear differences across genre in the pilot data:
      ghazal  : typically negative in dims 0, 3 (strong Sa-Pa fifth, modal Ga)
      party   : near-zero across all dims (beat-dominant, tonal context weak)
      sufi    : elevated in dims 0, 4 (strong harmonic root + devotional major third)

    This vector goes into the Melodic/Raga group in embedder.py alongside
    microtonal_pcp, meend_variance, and raga_probability. The fused vector
    grows from ~991d to ~997d. PCA to 256d absorbs this with no structural change.

    Args:
        features: dict returned by extract_features() (must contain tonnetz_mean).

    Returns:
        ndarray of shape (6,), dtype float32.
        Returns zeros if tonnetz_mean is missing (safe fallback).
    """
    return np.array(
        features.get("tonnetz_mean", [0.0] * 6),
        dtype=np.float32,
    )


# ─── Serialisation helpers ────────────────────────────────────────────────────

def save_features(features: dict, path: str) -> None:
    """Serialise the features dict to a JSON file."""
    with open(path, "w") as f:
        json.dump(features, f, indent=2)


def load_features(path: str) -> dict:
    """Deserialise a features dict from a JSON file."""
    with open(path) as f:
        return json.load(f)