"""
pipeline/stemmer.py
───────────────────
Source separation using demucs (htdemucs checkpoint).

Splits a 90-second WAV clip into:
  - clip_vocals.wav   → isolated vocal stem
  - clip_instr.wav    → instrumental stem (all non-vocal)

Both stems are written to the same directory as the input clip.
The caller (processor.py) is responsible for calling cleanup() on
both stems after feature extraction is complete.

Model choice: htdemucs
  - Faster than htdemucs on CPU (~2–3 min/song vs ~5 min/song)
  - Adequate separation quality for feature extraction purposes
  - Downloads ~200MB of weights on first run (cached in ~/.cache/torch/hub)

Fallback: if demucs fails for any reason, this module returns the
original full clip path for both stems so the pipeline can continue
with degraded (non-separated) features rather than failing entirely.
"""

import os
import subprocess
import shutil
import logging

log = logging.getLogger(__name__)


def _demucs_available() -> bool:
    return shutil.which("demucs") is not None


def separate_stems(clip_path: str) -> tuple[str, str]:
    """
    Run demucs on `clip_path` and return (vocals_path, instrumental_path).

    Demucs writes output to:
        <clip_dir>/htdemucs/<clip_stem>/vocals.wav
        <clip_dir>/htdemucs/<clip_stem>/no_vocals.wav  (drums+bass+other)

    We rename them to:
        <clip_dir>/<song_id>_vocals.wav
        <clip_dir>/<song_id>_instr.wav

    Returns:
        (vocals_path, instr_path) — both are absolute paths to WAV files.

    On any failure, returns (clip_path, clip_path) so downstream
    extraction degrades gracefully instead of crashing.
    """
    clip_dir  = os.path.dirname(os.path.abspath(clip_path))
    clip_stem = os.path.splitext(os.path.basename(clip_path))[0]

    vocals_out = os.path.join(clip_dir, f"{clip_stem}_vocals.wav")
    instr_out  = os.path.join(clip_dir, f"{clip_stem}_instr.wav")

    # Already separated in a prior run (idempotent)
    if os.path.exists(vocals_out) and os.path.exists(instr_out):
        return vocals_out, instr_out

    if not _demucs_available():
        log.warning("demucs not found in PATH — skipping source separation. "
                    "Install with: pip install demucs")
        return clip_path, clip_path

    try:
        # --two-stems=vocals produces exactly vocals.wav + no_vocals.wav
        # --clip-mode  rescale handles clipping without distortion
        # -n htdemucs  is the fast lightweight checkpoint
        cmd = [
            "demucs",
            "--two-stems", "vocals",
            "-n", "htdemucs",       # <-- FIX 1
            "-d", "mps",            # <-- FIX 2 (Mac GPU Acceleration)
            "--clip-mode", "rescale",
            "--out", clip_dir,
            clip_path,
        ]
        log.info(f"Running demucs on {os.path.basename(clip_path)} ...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,   # 10-minute hard cap; htdemucs is ~2-3 min on CPU
        )

        if result.returncode != 0:
            log.error(f"demucs failed (rc={result.returncode}): {result.stderr[:500]}")
            return clip_path, clip_path

        # Demucs writes to: <out>/<model>/<clip_stem>/vocals.wav
        demucs_vocals = os.path.join(clip_dir, "htdemucs", clip_stem, "vocals.wav")
        demucs_novox  = os.path.join(clip_dir, "htdemucs", clip_stem, "no_vocals.wav")

        if not os.path.exists(demucs_vocals) or not os.path.exists(demucs_novox):
            log.error(f"demucs ran but expected output files not found: "
                      f"{demucs_vocals}, {demucs_novox}")
            return clip_path, clip_path

        shutil.move(demucs_vocals, vocals_out)
        shutil.move(demucs_novox,  instr_out)

        # Clean up the demucs output subfolder
        demucs_subdir = os.path.join(clip_dir, "htdemucs", clip_stem)
        if os.path.isdir(demucs_subdir):
            shutil.rmtree(demucs_subdir, ignore_errors=True)
        demucs_model_dir = os.path.join(clip_dir, "htdemucs")
        try:
            os.rmdir(demucs_model_dir)   # only removes if empty
        except OSError:
            pass

        log.info(f"Stems ready: {os.path.basename(vocals_out)}, "
                 f"{os.path.basename(instr_out)}")
        return vocals_out, instr_out

    except subprocess.TimeoutExpired:
        log.error("demucs timed out after 10 minutes — falling back to full clip.")
        return clip_path, clip_path
    except Exception as exc:
        log.error(f"demucs raised an unexpected error: {exc}")
        return clip_path, clip_path