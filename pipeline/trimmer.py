"""
pipeline/trimmer.py

Extracts a 45-second clip from a raw audio download.

Default behaviour: scans the track to find the loudest (highest RMS) window,
which corresponds to the hook / chorus — the most musically dense segment.
This is better than a fixed offset for Indian music because the chorus is where
vocal ornaments (murki, meend) and the raga's characteristic phrases appear most.

Override: if clip_start is manually set in songs.csv (e.g. a long instrumental
intro that fools the RMS scan), pass force_start_sec to skip the scan entirely.
"""

import os
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_loudest_section_start(audio: AudioSegment, duration_ms: int) -> int:
    """
    Slide a window across the audio and return the start time (ms) of the
    window with the highest average RMS energy.

    Scan range: skip the first 20s (intros common in Bollywood) and the final
    <duration> of the track (outro fades).  Step size: 5s.
    """
    step_ms      = 5_000
    scan_start   = 20_000                    # skip first 20s
    scan_end     = len(audio) - duration_ms  # don't over-run end

    if scan_end <= scan_start:
        logger.debug("Track too short for scan window — using start=0")
        return 0

    best_start_ms = scan_start
    max_rms       = 0

    for start_ms in range(scan_start, scan_end, step_ms):
        window_rms = audio[start_ms : start_ms + duration_ms].rms
        if window_rms > max_rms:
            max_rms       = window_rms
            best_start_ms = start_ms

    return best_start_ms


# ── Public API ────────────────────────────────────────────────────────────────

def trim_audio(
    input_path:      str,
    output_path:     str,
    duration_sec:    int = 45,
    force_start_sec: int = None,
) -> int:
    """
    Extract a clip from *input_path* and write it to *output_path* (WAV).

    Args:
        input_path:      Path to the raw downloaded audio file.
        output_path:     Destination WAV path for the trimmed clip.
        duration_sec:    Clip length in seconds (default 45).
        force_start_sec: If provided, skip the RMS scan and start here.
                         Use when songs.csv has a manually tuned clip_start.

    Returns:
        Actual start offset used in seconds (useful for logging / DB).
    """
    duration_ms = duration_sec * 1000

    try:
        audio = AudioSegment.from_file(input_path)

        if force_start_sec is not None:
            start_ms = force_start_sec * 1000
            logger.info("Using forced start offset: %.1fs", force_start_sec)
        else:
            start_ms = _get_loudest_section_start(audio, duration_ms)
            logger.info("Smart-trim found hook at %.1fs", start_ms / 1000)

        clip = audio[start_ms : start_ms + duration_ms]
        clip.export(output_path, format="wav")
        return start_ms // 1000

    except Exception as exc:
        logger.warning("trim_audio failed (%s) — falling back to static 30s cut", exc)
        audio = AudioSegment.from_file(input_path)
        audio[30_000 : 30_000 + duration_ms].export(output_path, format="wav")
        return 30


def cleanup(file_path: str) -> None:
    """Delete a file from disk.  Safe to call with None or missing paths."""
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        logger.debug("Deleted: %s", file_path)