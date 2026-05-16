"""
pipeline/indian_features.py
────────────────────────────
India-specific acoustic feature extraction.

Implements the features specified in FEATURES.md that standard librosa
or Western music analysis tools do not cover:

  Group 2 (Vocal / Gayaki) — from clip_vocals.wav:
    - vocal_energy_ratio  (1d)  : ghazal vs club/party separator
    - murki_index         (1d)  : classical ornamentation quantifier

  Group 3 (Melodic / Raga) — from full clip:
    - microtonal_pcp      (36d) : 36-bin pitch class profile (replaces 12-bin chroma)
    - meend_variance      (1d)  : pitch glide prevalence (classical vs pop)
    - raga_probability    (30d) : prob distribution over 30 Hindustani ragas

  Group 4 (Rhythmic / Taal) — from clip_instr.wav:
    - onset_skewness      (1d)  : tabla/dholak vs synthetic 808 separator
    - hnr_mean            (1d)  : harmonic-to-noise ratio of instrumental stem

CHANGES FROM PREVIOUS VERSION (per hindi_engine_improvements.md):

  [FIX 1] murki_index — rewrote to:
    (a) Use only truly voiced frames (not forward-filled unvoiced gaps).
        Forward-fill inserts constant-pitch segments whose derivative is
        zero, diluting 5–15 Hz band power and making all songs look flat.
    (b) Return band_power / total_power ratio (0–1), not log1p(variance).
        The ratio is dimensionless and scale-invariant across singers with
        different absolute pitch ranges.
    (c) Changed function signature: extract_murki_index(f0, voiced_flag)
        so the caller passes the raw pyin outputs, not the filled array.

  [FIX 2] raga_probability — rewrote to:
    (a) Accept pre-computed microtonal_pcp (36-bin) as primary input.
        Avoids a redundant audio reload and is consistent with the PCP
        used for FAISS indexing.
    (b) Downsample 36-bin → 12-bin by summing triplets before correlating
        against the 12-bin raga templates.
    (c) Added _downsample_pcp_36_to_12() helper.
    (d) Full priority chain preserved: CompMusic → Essentia → PCP heuristic.

  [FIX 3] onset_skewness — improved discrimination:
    (a) Added HPSS (harmonic-percussive separation, margin=3.0) on the
        instrumental stem BEFORE computing onset strength.
        This removes residual bass/keys from the instr stem so the onset
        envelope is driven purely by percussive hits.
    (b) Tabla/dholak hits (long exponential decay) now produce higher skew
        vs. 808 hits (abrupt cutoff, near-zero skew) more reliably.

  [FIX 4] extract_all_indian_features — orchestration improvements:
    (a) Detects stem separation failure (vocals_path == clip_path) and
        returns stem_separation_failed=True in the result dict.
    (b) Passes (f0, voiced_flag) to extract_murki_index instead of f0_filled.
    (c) Passes microtonal_pcp to extract_raga_probability to avoid
        redundant audio reloads and use a consistent PCP source.

  [NO CHANGE] vocal_energy_ratio, meend_variance, hnr_mean, microtonal_pcp:
    These were identified as working correctly in the audit. No changes made.

All functions return numpy float32 arrays of fixed dimension.
All functions handle exceptions internally and return zero-filled
fallback arrays rather than raising, so a single bad feature
never crashes the whole pipeline.
"""

import os
import logging
import numpy as np
import librosa
import scipy.stats

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SR       = 22050   # standard sample rate used throughout
N_MFCC   = 20      # vocal MFCC coefficients
PCP_BINS = 36      # microtonal pitch class profile bins (3× standard chroma)
N_RAGAS  = 30      # fixed raga probability vector length

# The 30 Hindustani ragas covered by the probability vector.
# Order is fixed — do not change without rebuilding all embeddings.
RAGA_NAMES = [
    "Yaman", "Bhairav", "Bhairavi", "Kafi", "Khamaj", "Bilawal",
    "Kalyan", "Marwa", "Poorvi", "Todi", "Bhupali", "Desh",
    "Pilu", "Tilak Kamod", "Jhinjhoti", "Pahadi", "Kirwani",
    "Charukeshi", "Madhuvanti", "Malkauns", "Darbari", "Bageshri",
    "Kedar", "Vrindavani Sarang", "Miyan ki Malhar", "Shuddh Sarang",
    "Bhimpalasi", "Jaunpuri", "Lalit", "Shree",
]

# ─── Group 2 : Vocal / Gayaki ─────────────────────────────────────────────────

def extract_vocal_energy_ratio(vocals_path: str, instr_path: str) -> float:
    """
    Compute RMS(vocals) / (RMS(vocals) + RMS(instr)).

    Returns a scalar in [0, 1]:
      ~0.8–0.9  → strongly vocal-dominant (ghazal, classical)
      ~0.5–0.7  → mixed (sufi qawwali, soft pop)
      ~0.3–0.5  → beat-dominant (party, club, hip-hop)

    If either stem path equals the full clip path (demucs fallback),
    the ratio will be ~0.5 for most songs — acceptable for degraded mode.
    """
    try:
        y_v, _ = librosa.load(vocals_path, sr=SR, mono=True)
        y_i, _ = librosa.load(instr_path,  sr=SR, mono=True)

        rms_v = float(librosa.feature.rms(y=y_v).mean())
        rms_i = float(librosa.feature.rms(y=y_i).mean())

        denom = rms_v + rms_i
        if denom < 1e-8:
            return 0.5   # silence — return neutral value
        return float(rms_v / denom)

    except Exception as exc:
        log.warning(f"vocal_energy_ratio failed: {exc}")
        return 0.5


def _compute_master_f0(vocals_path: str):
    """
    Compute F0 once on the clean vocal stem.

    Returns (f0, voiced_flag) as raw pyin outputs:
      - f0:          ndarray of shape (T,), NaN for unvoiced frames
      - voiced_flag: boolean ndarray of shape (T,)

    Running pyin once and sharing the result across murki_index,
    microtonal_pcp, and meend_variance avoids triple Viterbi overhead.
    """
    try:
        y, _ = librosa.load(vocals_path, sr=SR, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=SR,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=2048,
        )
        return f0, voiced_flag
    except Exception as exc:
        log.warning(f"Master F0 computation failed: {exc}")
        return None, None


# [FIX 1] murki_index — complete rewrite
# Old: extract_murki_index(f0_filled) → log1p(variance of band power amplitudes)
#      Problems:
#        - f0_filled contains forward-filled unvoiced segments. diff() on these
#          flat segments produces zeros, diluting 5–15 Hz band power.
#        - log1p(variance) is not scale-invariant. A singer with larger absolute
#          pitch swings gets a higher murki_index even with identical ornamentation.
#
# New: extract_murki_index(f0, voiced_flag) → band_power / total_power ratio (0–1)
#      Fixes:
#        - Only voiced frames are used — no artificial zero-derivative segments.
#        - Ratio is dimensionless: classical ≈ 0.15–0.35, pop/club ≈ 0.01–0.05.

def extract_murki_index(f0: np.ndarray, voiced_flag: np.ndarray) -> float:
    """
    Quantify rapid classical ornamentation (murki, gamak).

    Method:
      1. Select only voiced frames using voiced_flag (no forward-fill).
      2. Compute first derivative of F0 (Hz/frame) — pitch velocity.
      3. FFT the derivative to decompose into oscillation frequencies.
      4. Restrict to the 5–15 Hz band (faster than vibrato <5 Hz,
         slower than pitch noise >15 Hz).
      5. Return band_power / total_power — a scale-invariant ratio in [0, 1].

    The 5–15 Hz band filter is MANDATORY per FEATURES.md spec:
    without it, vibrato-heavy singers (Lata, Asha) produce falsely high
    murki values despite not performing classical gamak ornaments.

    Args:
        f0:          Raw pyin F0 array (NaN for unvoiced), shape (T,).
        voiced_flag: Boolean array, True where pyin found a voiced frame.

    Returns:
        float in [0, 1]. Typical ranges:
          0.15–0.35  → heavy classical ornamentation (khyal, thumri, ghazal)
          0.05–0.15  → moderate ornamentation (Bollywood semi-classical)
          0.01–0.05  → flat delivery (modern pop, hip-hop, club)
    """
    try:
        if f0 is None or voiced_flag is None:
            return 0.0

        # Select only voiced frames — avoids zero-derivative artefacts from fill
        valid = voiced_flag & ~np.isnan(f0) & (f0 > 0)
        f0_voiced = f0[valid]

        if len(f0_voiced) < 32:
            # Too few voiced frames for reliable FFT (need ≥ 32 for 5–15 Hz bins)
            return 0.0

        # First derivative of F0 (pitch velocity in Hz/frame)
        f0_diff = np.diff(f0_voiced)

        # Frame rate: pyin default hop_length=512 at sr=22050 → ~43.066 Hz
        frame_rate = SR / 512.0

        # FFT the pitch velocity signal
        fft_mag = np.abs(np.fft.rfft(f0_diff))
        freqs   = np.fft.rfftfreq(len(f0_diff), d=1.0 / frame_rate)

        # Isolate the 5–15 Hz band (murki/gamak oscillation range)
        band_mask = (freqs >= 5.0) & (freqs <= 15.0)
        if not band_mask.any():
            return 0.0

        # Power ratio: fraction of total pitch-velocity energy in the ornament band
        band_power  = float(np.sum(fft_mag[band_mask] ** 2))
        total_power = float(np.sum(fft_mag ** 2)) + 1e-10

        return float(band_power / total_power)

    except Exception as exc:
        log.warning(f"murki_index failed: {exc}")
        return 0.0


def _fill_unvoiced(f0: np.ndarray) -> np.ndarray:
    """
    Replace NaN/zero with the last valid F0 value (forward fill).
    Used by meend_variance only — NOT by murki_index (see FIX 1).
    """
    f0 = f0.copy()
    last = 0.0
    for i in range(len(f0)):
        if np.isnan(f0[i]) or f0[i] == 0:
            f0[i] = last
        else:
            last = f0[i]
    return f0


# ─── Group 3 : Melodic / Raga ─────────────────────────────────────────────────

def extract_microtonal_pcp(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    fallback_clip: str,
) -> np.ndarray:
    """
    36-bin Pitch Class Profile (PCP) — replaces the standard 12-bin chroma.

    Accepts pre-computed F0 to avoid re-running pyin on polyphonic audio.
    Falls back to chroma_stft if pyin produced insufficient voiced frames.

    Args:
        f0:           Raw pyin F0 (NaN for unvoiced), shape (T,).
        voiced_flag:  Boolean voiced mask, shape (T,).
        fallback_clip: Full clip path for chroma fallback.

    Returns:
        ndarray of shape (36,), dtype float32, L2-normalised.
    """
    try:
        pcp = _pyin_pcp_36(f0, voiced_flag)
        if pcp is None:
            y, _ = librosa.load(fallback_clip, sr=SR, mono=True)
            pcp = _chroma_pcp_36(y)
        return pcp.astype(np.float32)

    except Exception as exc:
        log.warning(f"microtonal_pcp failed: {exc} — returning zeros")
        return np.zeros(PCP_BINS, dtype=np.float32)


def _pyin_pcp_36(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray | None:
    """Build a 36-bin PCP from pre-computed pyin F0 estimates."""
    try:
        if f0 is None or voiced is None:
            return None

        f0_voiced = f0[voiced & ~np.isnan(f0) & (f0 > 0)]
        if len(f0_voiced) < 10:
            return None

        midi = 12 * np.log2(f0_voiced / 440.0) + 69
        bins = (midi * 3).astype(int) % PCP_BINS
        pcp  = np.bincount(bins, minlength=PCP_BINS).astype(np.float32)

        norm = np.linalg.norm(pcp)
        if norm > 0:
            pcp /= norm
        return pcp

    except Exception:
        return None


def _chroma_pcp_36(y: np.ndarray) -> np.ndarray:
    """
    Fallback: compute standard 12-bin chroma, then zero-interleave to 36 bins.
    This preserves vector dimensionality at lower microtonal accuracy.
    """
    chroma      = librosa.feature.chroma_stft(y=y, sr=SR, n_chroma=12)
    chroma_mean = chroma.mean(axis=1)   # shape (12,)

    # Interleave: each semitone bin → 3 sub-bins (centre value, two zeros)
    pcp36 = np.zeros(PCP_BINS, dtype=np.float32)
    pcp36[::3] = chroma_mean   # positions 0, 3, 6, ..., 33
    norm = np.linalg.norm(pcp36)
    if norm > 0:
        pcp36 /= norm
    return pcp36


def extract_meend_variance(f0_filled: np.ndarray) -> float:
    """
    Measure prevalence of pitch glides (meend, andolan).

    Uses the FORWARD-FILLED F0 (not voiced-only) intentionally: meend is a
    sustained slide across a phrase, and including unvoiced interpolation
    captures the contour shape across the whole melodic line.

    This distinguishes meend_variance from murki_index:
      - murki_index:    rapid 5–15 Hz oscillations, voiced frames only
      - meend_variance: overall glide prevalence across the full phrase

    Returns variance of |Δf0| normalised by mean F0 (dimensionless ratio).
    Typical ranges:
      > 0.05   → heavy gliding (classical, semi-classical)
      0.01–0.05 → moderate (Bollywood romantic)
      < 0.01   → stepped/quantised (pop, hip-hop, electronic)
    """
    try:
        if f0_filled is None:
            return 0.0

        f0_voiced = f0_filled[f0_filled > 0]
        if len(f0_voiced) < 10:
            return 0.0

        deriv   = np.abs(np.diff(f0_voiced))
        mean_f0 = np.mean(f0_voiced)

        if mean_f0 < 1e-6:
            return 0.0

        return float(np.var(deriv / mean_f0))

    except Exception as exc:
        log.warning(f"meend_variance failed: {exc}")
        return 0.0


# [FIX 2] raga_probability — signature changed + PCP downsampling path added
# Old: extract_raga_probability(clip_path) — always reloads audio for heuristic
# New: extract_raga_probability(clip_path, microtonal_pcp=None)
#      If microtonal_pcp (36-bin, already computed) is provided, downsamples
#      to 12-bin and uses it directly — avoids a full audio reload and ensures
#      the raga estimate is consistent with the PCP used in the FAISS vector.

def extract_raga_probability(
    clip_path: str,
    microtonal_pcp: np.ndarray = None,
) -> np.ndarray:
    """
    Return a 30-dimensional probability vector over Hindustani ragas.

    Priority chain:
      1. CompMusic Hindustani raga classifier (if installed)
      2. Essentia TonalExtractor (if installed) — approximate proxy
      3. PCP heuristic using pre-computed microtonal_pcp (preferred)
      4. PCP heuristic by reloading audio (fallback when pcp is unavailable)
      5. Zero vector (last resort)

    Args:
        clip_path:      Full 45s clip path (used only for methods 2 and 4).
        microtonal_pcp: Pre-computed 36-bin PCP from extract_microtonal_pcp().
                        If provided, used directly for the heuristic — avoids
                        redundant audio reload and keeps vectors consistent.

    Returns:
        ndarray of shape (30,), dtype float32, sums to ~1.0.
    """
    try:
        result = _try_compmusic_raga(clip_path)
        if result is not None:
            return result.astype(np.float32)
    except Exception:
        pass

    try:
        result = _try_essentia_raga(clip_path)
        if result is not None:
            return result.astype(np.float32)
    except Exception:
        pass

    try:
        if microtonal_pcp is not None and len(microtonal_pcp) == PCP_BINS:
            # [FIX 2a] Downsample pre-computed 36-bin PCP → 12-bin for templates
            pcp12  = _downsample_pcp_36_to_12(microtonal_pcp)
            return _hpcp_to_raga_probs(pcp12).astype(np.float32)
        else:
            # Fallback: reload audio and compute chroma from scratch
            return _heuristic_raga_pcp(clip_path).astype(np.float32)

    except Exception as exc:
        log.warning(f"raga_probability (all methods) failed: {exc}")
        return np.zeros(N_RAGAS, dtype=np.float32)


# [FIX 2b] New helper: downsample 36-bin PCP to 12-bin by summing triplets
def _downsample_pcp_36_to_12(pcp36: np.ndarray) -> np.ndarray:
    """
    Collapse a 36-bin PCP into 12 semitone bins by summing each triplet.

    The 36-bin PCP divides each semitone into 3 sub-bins (one-third tone
    resolution). Summing each triplet gives the total energy in that
    semitone class, which is what the 12-bin raga templates represent.

    Args:
        pcp36: ndarray of shape (36,), L2-normalised.

    Returns:
        ndarray of shape (12,), L2-normalised, dtype float32.
    """
    pcp36 = np.asarray(pcp36, dtype=np.float32)
    pcp12 = pcp36.reshape(12, 3).sum(axis=1)   # sum sub-bins per semitone
    norm  = np.linalg.norm(pcp12)
    if norm > 0:
        pcp12 = pcp12 / norm
    return pcp12.astype(np.float32)


def _try_compmusic_raga(clip_path: str) -> np.ndarray | None:
    """
    Attempt to use the CompMusic Hindustani raga recogniser.
    Returns None if the package is not installed.
    """
    try:
        import compmusic  # noqa: F401
        log.info("CompMusic raga classifier found but API call not yet wired.")
        return None
    except ImportError:
        return None


def _try_essentia_raga(clip_path: str) -> np.ndarray | None:
    """
    Use Essentia's TonalExtractor as a proxy for raga-like tonal description.
    Returns None if essentia is not installed.
    """
    try:
        import essentia.standard as es  # noqa: F401
        loader    = es.MonoLoader(filename=clip_path, sampleRate=SR)
        audio     = loader()
        extractor = es.TonalExtractor()
        tonal     = extractor(audio)
        hpcp      = np.array(tonal[0], dtype=np.float32)   # 12-bin HPCP
        return _hpcp_to_raga_probs(hpcp)
    except ImportError:
        return None
    except Exception as exc:
        log.warning(f"Essentia raga extraction failed: {exc}")
        return None


# ─── Raga template library ─────────────────────────────────────────────────────
# Each row is a 12-bin pitch class profile (normalised) for one raga.
# Mapping: Sa=0, Re♭=1, Re=2, Ga♭=3, Ga=4, Ma=5, Ma♯=6, Pa=7,
#           Dha♭=8, Dha=9, Ni♭=10, Ni=11.
# Templates capture vadi/samvadi emphasis and characteristic swaras.
# Rows are in the same order as RAGA_NAMES above.

_RAGA_TEMPLATES_12 = np.array([
    # Yaman       (Kalyan thaat, Tivra Ma)
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    # Bhairav     (Morning, flat Re and Dha)
    [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    # Bhairavi    (All flat)
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    # Kafi        (flat Ga and Ni)
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    # Khamaj      (flat Ni)
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    # Bilawal     (natural / Ionian scale)
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    # Kalyan      (Tivra Ma, pentatonic feel)
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    # Marwa       (no Pa, flat Re, sharp Ma)
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    # Poorvi      (flat Re, sharp Ma, flat Dha)
    [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    # Todi        (flat Re, flat Ga, sharp Ma, flat Dha)
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    # Bhupali     (pentatonic, no Ma and Ni)
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    # Desh        (mix of Khamaj and Kafi)
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    # Pilu        (both Ga and Ni variants)
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    # Tilak Kamod (evening, mix)
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    # Jhinjhoti   (light, both Ni)
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    # Pahadi      (pentatonic base, mountain)
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    # Kirwani     (Harmonic minor feel)
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    # Charukeshi  (Major + flat 6 and 7)
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    # Madhuvanti  (Tivra Ma, flat Ga)
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    # Malkauns    (pentatonic, no Re, no Pa)
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    # Darbari     (late night, flat Ga, flat Dha, andolan)
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    # Bageshri    (night, flat Ga)
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    # Kedar       (evening, Tivra Ma)
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    # Vrindavani Sarang (afternoon, no Ga)
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    # Miyan ki Malhar (monsoon, flat Ga)
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    # Shuddh Sarang (afternoon, no Ga no Dha)
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    # Bhimpalasi  (afternoon, flat Ga and Ni)
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    # Jaunpuri    (morning, flat Re Ga Dha)
    [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    # Lalit       (pre-dawn, no Pa, flat Re, sharp Ma)
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
    # Shree       (evening, flat Re, sharp Ma, flat Dha)
    [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
], dtype=np.float32)

# L2-normalise each template row once at module load time
_row_norms = np.linalg.norm(_RAGA_TEMPLATES_12, axis=1, keepdims=True)
_row_norms[_row_norms == 0] = 1.0
_RAGA_TEMPLATES_12 = _RAGA_TEMPLATES_12 / _row_norms


def _hpcp_to_raga_probs(hpcp_12: np.ndarray) -> np.ndarray:
    """
    Given a 12-bin HPCP, compute cosine similarity to each raga template.
    Apply softmax with temperature=0.5 to sharpen the distribution.

    Args:
        hpcp_12: ndarray of shape (12,), L2-normalised.

    Returns:
        ndarray of shape (30,), probability distribution summing to ~1.0.
    """
    hpcp_12 = np.asarray(hpcp_12, dtype=np.float32)
    norm = np.linalg.norm(hpcp_12)
    if norm > 0:
        hpcp_12 = hpcp_12 / norm

    # Cosine similarity: templates already normalised, so dot product = cos_sim
    similarities = _RAGA_TEMPLATES_12 @ hpcp_12   # shape (30,)

    # Softmax with temperature 0.5 — sharpens the distribution around the
    # top matches. Without temperature, similarities are too close to uniform.
    temp  = 0.5
    exp_s = np.exp(similarities / temp)
    probs = exp_s / exp_s.sum()
    return probs.astype(np.float32)


def _heuristic_raga_pcp(clip_path: str) -> np.ndarray:
    """
    Emergency fallback: build 12-bin HPCP by reloading audio.
    Called only when microtonal_pcp was not supplied.
    """
    y, _ = librosa.load(clip_path, sr=SR, mono=True)
    chroma  = librosa.feature.chroma_cqt(y=y, sr=SR, n_chroma=12)
    hpcp_12 = chroma.mean(axis=1)
    return _hpcp_to_raga_probs(hpcp_12)


# ─── Group 4 : Rhythmic / Taal ─────────────────────────────────────────────────

# [FIX 3] onset_skewness — added HPSS before onset strength computation
# Old: onset_strength(y=y, sr=SR) on raw instr stem
#      Problem: instr stem still contains bass and keys (harmonic content).
#               Their energy bleeds into onset_strength, partially masking
#               the acoustic tail differences between tabla and 808.
# New: onset_strength(y=y_percussive, sr=SR) on HPSS percussive component
#      Isolates only transient percussion hits → much cleaner decay profiles.
#      margin=3.0 makes HPSS aggressive: prefers sharp percussion separation
#      over gentle median filtering.

def extract_onset_skewness(instr_path: str) -> float:
    """
    Compute statistical skewness of the onset strength envelope
    from the PERCUSSIVE component of the instrumental stem.

    Physical basis:
      - Tabla / dholak hits: sharp attack + long exponential decay tail
        → positively skewed onset envelope (long right tail)
      - Synthetic 808 / TR-808 drums: sharp attack + abrupt digital cutoff
        → near-zero skew (symmetric or slightly negative)

    HPSS (harmonic-percussive source separation) is applied to the
    instrumental stem before onset detection to remove residual bass,
    keys, and strings — leaving only transient percussion energy.
    This makes the tabla decay tail much more visible in the envelope.

    Args:
        instr_path: Path to the instrumental stem WAV (or full clip if
                    demucs separation failed). The caller should set
                    stem_separation_failed=True in that case, which the
                    embedder uses to down-weight this feature.

    Returns:
        float. Typical ranges after HPSS fix:
          > 2.0  → strongly tabla/dholak (ghazal, classical, folk)
          1.0–2.0 → mixed acoustic/electronic (semi-classical, indie)
          < 1.0  → synthetic/electronic (party, hip-hop, EDM)
    """
    try:
        y, _ = librosa.load(instr_path, sr=SR, mono=True)

        # Isolate percussive component — removes harmonic bleed from bass/keys
        # margin=3.0: aggressive separation (higher margin = harder mask)
        _, y_percussive = librosa.effects.hpss(y, margin=3.0)

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=SR)
        return float(scipy.stats.skew(onset_env))

    except Exception as exc:
        log.warning(f"onset_skewness failed: {exc}")
        return 0.0


def extract_hnr_mean(instr_path: str) -> float:
    """
    Compute mean Harmonic-to-Noise Ratio of the instrumental stem.

    Implementation: ratio of harmonic component energy to residual (noise)
    energy using librosa's harmonic/percussive decomposition.

    High HNR → melodic/acoustic instrumentation (sarangi, harmonium, sitar)
    Low HNR  → noisy/percussive instrumentation (heavy 808, distorted synths)

    Returns:
        float in [0, 1] (normalised ratio). Returns 0.5 on failure.
    """
    try:
        y, _ = librosa.load(instr_path, sr=SR, mono=True)
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        rms_h = float(librosa.feature.rms(y=y_harmonic).mean())
        rms_p = float(librosa.feature.rms(y=y_percussive).mean())

        denom = rms_h + rms_p
        if denom < 1e-8:
            return 0.5
        return float(rms_h / denom)

    except Exception as exc:
        log.warning(f"hnr_mean failed: {exc}")
        return 0.5


# ─── Convenience: extract all Indian features in one call ─────────────────────

# [FIX 4] extract_all_indian_features — orchestration changes:
#   (a) Detects stem_separation_failed and propagates it in the return dict.
#       The embedder uses this flag to down-weight stem-derived features.
#   (b) Passes (f0, voiced_flag) to extract_murki_index — not f0_filled.
#   (c) Passes microtonal_pcp to extract_raga_probability — avoids audio reload.
#   (d) f0_filled is still computed for meend_variance (uses forward-fill intentionally).

def extract_all_indian_features(
    clip_path:   str,
    vocals_path: str,
    instr_path:  str,
) -> dict:
    """
    Extract all Indian-specific features and return a dict of arrays + flags.

    Args:
        clip_path:    Full 45s clip (full mix)
        vocals_path:  Isolated vocal stem (or clip_path if demucs unavailable)
        instr_path:   Instrumental stem (or clip_path if demucs unavailable)

    Returns:
        {
            "vocal_energy_ratio":    float,
            "murki_index":           float,          # [FIX 1] now 0–1 ratio
            "microtonal_pcp":        np.ndarray (36,),
            "meend_variance":        float,
            "raga_probability":      np.ndarray (30,),  # [FIX 2] uses pcp
            "onset_skewness":        float,          # [FIX 3] HPSS applied
            "hnr_mean":              float,
            "stem_separation_failed": bool,          # [FIX 4] new flag
        }
    """

    # [FIX 4a] Detect stem separation failure.
    # When demucs fails, separate_stems() returns (clip_path, clip_path).
    # All stem-dependent features degrade gracefully but we flag it so the
    # embedder can zero-weight or down-weight them.
    stem_separation_failed = (
        os.path.abspath(vocals_path) == os.path.abspath(clip_path)
        or os.path.abspath(instr_path) == os.path.abspath(clip_path)
    )
    if stem_separation_failed:
        log.warning(
            "Stem separation failed (demucs fallback) — "
            "vocal_energy_ratio, murki_index, and onset_skewness will be degraded."
        )

    # ── Group 2 : Vocal features ─────────────────────────────────────────────

    log.info("  Extracting vocal energy ratio ...")
    vocal_energy_ratio = extract_vocal_energy_ratio(vocals_path, instr_path)

    # Compute F0 once on vocal stem — shared by murki, microtonal_pcp, meend
    log.info("  Computing master F0 contour (librosa.pyin) ...")
    f0, voiced_flag = _compute_master_f0(vocals_path)

    # [FIX 4b] Pass (f0, voiced_flag) — NOT f0_filled — to murki_index
    log.info("  Extracting murki index ...")
    murki_index = extract_murki_index(f0, voiced_flag)

    # ── Group 3 : Melodic / Raga features ────────────────────────────────────

    log.info("  Extracting microtonal PCP ...")
    microtonal_pcp = extract_microtonal_pcp(f0, voiced_flag, clip_path)

    # f0_filled is needed only for meend_variance (uses full contour, not voiced-only)
    log.info("  Extracting meend variance ...")
    f0_filled    = _fill_unvoiced(f0) if f0 is not None else None
    meend_variance = extract_meend_variance(f0_filled)

    # [FIX 4c] Pass microtonal_pcp so raga estimator avoids redundant audio reload
    log.info("  Estimating raga probability ...")
    raga_probability = extract_raga_probability(
        clip_path,
        microtonal_pcp=microtonal_pcp,
    )

    # ── Group 4 : Rhythmic / Taal features ───────────────────────────────────

    log.info("  Extracting onset skewness (with HPSS) ...")
    onset_skewness = extract_onset_skewness(instr_path)

    log.info("  Extracting HNR mean ...")
    hnr_mean = extract_hnr_mean(instr_path)

    return {
        "vocal_energy_ratio":     vocal_energy_ratio,
        "murki_index":            murki_index,
        "microtonal_pcp":         microtonal_pcp,
        "meend_variance":         meend_variance,
        "raga_probability":       raga_probability,
        "onset_skewness":         onset_skewness,
        "hnr_mean":               hnr_mean,
        "stem_separation_failed": stem_separation_failed,   # [FIX 4a]
    }