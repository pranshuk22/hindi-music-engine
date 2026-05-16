"""
pipeline/downloader.py

Downloads audio from YouTube using yt-dlp and converts to WAV.

Output spec:
  - Format:      WAV (PCM)
  - Sample rate: 44100 Hz  (demucs htdemucs_light native SR; librosa resamples cleanly from this)
  - Channels:    stereo (2)
  - The returned path is verified to exist before returning.

Thread safety:
  Each download writes to a unique filename derived from the YouTube video ID,
  so concurrent ThreadPoolExecutor workers never collide.
"""

import os
import re
import logging
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ydl_opts(output_dir: str) -> dict:
    """
    Build yt-dlp options that produce a consistently formatted WAV file.

    postprocessor_args sets the output to 44100 Hz stereo so every clip that
    enters demucs and librosa has the same sample rate, avoiding subtle
    feature drift from mixed-SR sources.
    """
    return {
        "format":    "bestaudio/best",
        "outtmpl":   os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key":              "FFmpegExtractAudio",
            "preferredcodec":   "wav",
            "preferredquality": "0",      # irrelevant for lossless WAV; set 0 to be explicit
        }],
        "postprocessor_args": [
            "-ar", "44100",               # 44100 Hz — demucs native SR
            "-ac", "2",                   # stereo
        ],
        "quiet":       True,
        "no_warnings": True,
    }


def _resolve_wav_path(output_dir: str, info: dict) -> str:
    """
    Reliably construct the WAV path that yt-dlp wrote after post-processing.

    yt-dlp's FFmpegExtractAudio strips the original extension and appends .wav,
    so `{id}.webm` → `{id}.wav`, `{id}.m4a` → `{id}.wav`, etc.
    We use prepare_filename() to get the pre-conversion path, then swap the suffix.
    """
    # prepare_filename gives us the template-expanded path before conversion
    pre_conversion = ydl_instance_prepare_filename(output_dir, info)
    wav_path = str(Path(pre_conversion).with_suffix(".wav"))
    return wav_path


def _download(url: str, output_dir: str) -> str:
    """
    Core download + convert logic. Returns the verified WAV path.
    Raises on yt-dlp error or if the output file is not found after download.
    """
    os.makedirs(output_dir, exist_ok=True)
    opts = _ydl_opts(output_dir)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # For search URLs, yt-dlp returns a playlist-style dict with entries
    video_info = info["entries"][0] if "entries" in info else info
    video_id   = video_info["id"]

    # Construct the expected WAV path: {output_dir}/{video_id}.wav
    # This is reliable because outtmpl uses %(id)s and FFmpegExtractAudio
    # strips the original ext and appends .wav
    wav_path = os.path.join(output_dir, f"{video_id}.wav")

    if not os.path.isfile(wav_path):
        # Fallback: scan the directory for any .wav matching the video ID
        # (handles edge cases where yt-dlp sanitizes the ID)
        candidates = [
            f for f in os.listdir(output_dir)
            if f.endswith(".wav") and video_id in f
        ]
        if candidates:
            wav_path = os.path.join(output_dir, candidates[0])
            logger.debug("WAV path resolved via scan: %s", wav_path)
        else:
            raise FileNotFoundError(
                f"yt-dlp completed but WAV not found for video ID '{video_id}' "
                f"in {output_dir}. Directory contents: {os.listdir(output_dir)}"
            )

    logger.info("Downloaded: %s → %s", video_id, wav_path)
    return wav_path


# ── Public API ────────────────────────────────────────────────────────────────

def download_audio(youtube_url: str, output_dir: str = "/tmp/hindi_music_temp") -> str:
    """
    Download audio from a YouTube URL and return the path to the WAV file.

    Args:
        youtube_url: Direct YouTube video URL.
        output_dir:  Directory to write the WAV into.

    Returns:
        Absolute path to the downloaded WAV file (verified to exist).

    Raises:
        yt_dlp.utils.DownloadError: on yt-dlp failure (unavailable, age-gated, etc.)
        FileNotFoundError:          if the WAV is not found after download.
    """
    logger.info("Downloading URL: %s", youtube_url)
    return _download(youtube_url, output_dir)


def search_and_download(
    title:      str,
    artist:     str,
    output_dir: str = "/tmp/hindi_music_temp",
) -> str:
    """
    Search YouTube for a song and download the best match.

    Search strategy (two-pass):
      Pass 1: "{title} {artist} official audio"  — prefers clean uploads
      Pass 2: "{title} {artist}"                 — broader fallback for older
                                                   ghazals and rare tracks that
                                                   have no "official audio" upload

    Args:
        title:      Song title.
        artist:     Artist name. Handles NaN / empty strings safely.
        output_dir: Directory to write the WAV into.

    Returns:
        Absolute path to the downloaded WAV file.

    Raises:
        RuntimeError: if both search passes fail.
    """
    # Sanitise artist — songs.csv sometimes has NaN in the artist column
    artist_clean = artist.strip() if artist and str(artist).lower() != "nan" else ""

    base_query    = f"{title} {artist_clean}".strip()
    queries = [
        f"{base_query} official audio",  # pass 1 — clean studio upload
        base_query,                       # pass 2 — broad fallback
    ]

    last_exc = None
    for query in queries:
        search_url = f"ytsearch1:{query}"
        logger.info("Searching YouTube: %s", query)
        try:
            return _download(search_url, output_dir)
        except Exception as exc:
            logger.warning("Search pass failed ('%s'): %s", query, exc)
            last_exc = exc

    raise RuntimeError(
        f"All search passes failed for '{title} — {artist}'. "
        f"Last error: {last_exc}"
    )


# ── Unused helper kept for import compatibility ───────────────────────────────
# (ydl.prepare_filename is an instance method; this avoids opening a second
#  ydl context just for path resolution — the scan fallback in _download handles it)
def ydl_instance_prepare_filename(output_dir: str, info: dict) -> str:
    """Not used in the main path. Kept as a utility for debugging."""
    opts = _ydl_opts(output_dir)
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.prepare_filename(info)