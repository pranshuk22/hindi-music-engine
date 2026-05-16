"""
scripts/run_pipeline.py
────────────────────────
Batch pipeline runner: parallel workers, resumability, failure logging.

Usage:
  python scripts/run_pipeline.py --csv data/songs.csv
  python scripts/run_pipeline.py --csv data/songs.csv --workers 2
  python scripts/run_pipeline.py --csv data/songs.csv --start-from 45
  python scripts/run_pipeline.py --csv data/songs.csv --retry-failed
  python scripts/run_pipeline.py --csv data/songs.csv --force

Expected CSV columns:
  title        (required)
  artist       (required)
  category     (required) — ghazal | sufi | sad_romantic | upbeat | party
  youtube_url  (optional) — direct URL; yt-dlp searches by name if blank
  clip_start   (optional) — seconds to skip before 90s window (default 15)

AGENTS.md Rule 8 compliance:
  CLAP is loaded ONCE at module import, before any worker thread starts.
  This prevents two workers racing to load the same ~2GB model into RAM.
  Max safe workers on 8GB RAM: 2.

Failed songs are written to data/failed.csv for targeted retry.
Songs already in the DB (processed=1) are skipped unless --force is passed.
"""

import sys
import os
import argparse
import time
import logging
import concurrent.futures

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")

# ── DB init before anything else ─────────────────────────────────────────────
# init_db() must run before any worker touches SQLite, so the table exists
# and migrations are applied. Do it here at module level, not inside workers.
from utils.db import init_db, song_exists
init_db()

from pipeline.processor import process_song

# ── Pre-warm CLAP at import time (AGENTS.md Rule 8) ──────────────────────────
# All threads share this singleton — do NOT call get_clap() inside _process_one.
from pipeline.embedder import get_clap
get_clap()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FAILED_CSV = "data/failed.csv"


# ─── Song ID helper (must match processor.py exactly) ────────────────────────

def _make_song_id(title: str, artist: str) -> str:
    raw = f"{title}_{artist}".lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in raw)[:50]


# ─── Single song worker ───────────────────────────────────────────────────────

def _process_one(row: dict, force: bool = False) -> tuple[str | None, str | None]:
    """
    Process one CSV row through the full pipeline.
    Returns (song_id, None) on success, (None, error_str) on failure.
    """
    title  = str(row.get("title",  "")).strip()
    artist = str(row.get("artist", "")).strip()

    if not title or not artist:
        return None, f"Missing title or artist in row: {row}"

    song_id = _make_song_id(title, artist)

    if not force and song_exists(song_id):
        log.info(f"  SKIP (already processed): {title} — {artist}")
        return song_id, None

    # ── youtube_url ────────────────────────────────────────────────────────
    y_url = row.get("youtube_url", "")
    if pd.isna(y_url) or str(y_url).strip().lower() in ("", "nan", "none"):
        y_url = None
    else:
        y_url = str(y_url).strip()

    # ── clip_start (seconds to skip before 90s window) ─────────────────────
    # Sourced from CSV column 'clip_start'; default 15s per README spec.
    clip_start = 15
    raw_start  = row.get("clip_start", "")
    if raw_start is not None and not (isinstance(raw_start, float) and pd.isna(raw_start)):
        try:
            clip_start = int(float(str(raw_start).strip()))
            if clip_start < 0:
                log.warning(f"  {title}: clip_start={clip_start} is negative — using 0")
                clip_start = 0
        except (ValueError, TypeError):
            log.warning(f"  {title}: invalid clip_start '{raw_start}' — using default 15s")
            clip_start = 15

    # ── category ───────────────────────────────────────────────────────────
    category = str(row.get("category", "unknown")).strip()
    if not category or category.lower() in ("nan", "none", ""):
        category = "unknown"

    try:
        sid = process_song(
            title       = title,
            artist      = artist,
            youtube_url = y_url,
            clip_start  = clip_start,
            category    = category,
        )
        time.sleep(1)   # polite pause between yt-dlp requests
        return sid, None

    except Exception as exc:
        log.error(f"  FAILED: {title} — {artist}: {exc}")
        return None, str(exc)


# ─── Batch runner ─────────────────────────────────────────────────────────────

def run_batch(
    csv_path:     str,
    start_from:   int  = 0,
    n_workers:    int  = 1,
    force:        bool = False,
    retry_failed: bool = False,
):
    """
    Run the pipeline on all rows in `csv_path`.

    Args:
        csv_path:     Path to songs CSV (title, artist, category, youtube_url, clip_start)
        start_from:   Skip the first N rows (0-indexed) — for resuming
        n_workers:    Number of parallel threads (max 2 on 8GB RAM)
        force:        Reprocess songs already marked processed=1 in DB
        retry_failed: Read data/failed.csv instead of full CSV
    """
    if retry_failed and os.path.exists(FAILED_CSV):
        log.info(f"Retry mode: reading failed songs from {FAILED_CSV}")
        df_failed = pd.read_csv(FAILED_CSV)

        # Merge back with original CSV to recover youtube_url, clip_start, category
        try:
            df_orig = pd.read_csv(csv_path)
            # Merge on title (best stable key in failed.csv)
            df = df_failed[["title"]].merge(df_orig, on="title", how="left")
            if df.empty:
                log.warning("Merge returned empty — falling back to failed.csv as-is")
                df = df_failed
        except Exception as exc:
            log.warning(f"Could not merge with original CSV ({exc}) — using failed.csv as-is")
            df = df_failed

    else:
        df = pd.read_csv(csv_path)

    # Validate required columns
    missing_cols = [c for c in ("title", "artist") if c not in df.columns]
    if missing_cols:
        log.error(f"CSV is missing required columns: {missing_cols}")
        sys.exit(1)

    # Warn if important optional columns are absent
    for col, hint in [
        ("category",    "all songs will be labelled 'unknown'"),
        ("clip_start",  "all songs will use default 15s start offset"),
        ("youtube_url", "yt-dlp will search by title+artist name"),
    ]:
        if col not in df.columns:
            log.warning(f"  CSV has no '{col}' column — {hint}")

    if start_from > 0:
        df = df.iloc[start_from:].reset_index(drop=True)
        log.info(f"Skipped first {start_from} rows (--start-from)")

    rows = df.to_dict("records")
    log.info(f"Processing {len(rows)} songs with {n_workers} worker(s) ...")

    failed = []

    if n_workers == 1:
        for row in tqdm(rows, desc="Pipeline", unit="song"):
            sid, err = _process_one(row, force=force)
            if err:
                failed.append({"title": row.get("title", ""), "error": err})
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_one, row, force): row
                for row in rows
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Pipeline",
                unit="song",
            ):
                row = futures[future]
                try:
                    sid, err = future.result()
                    if err:
                        failed.append({"title": row.get("title", ""), "error": err})
                except Exception as exc:
                    failed.append({"title": row.get("title", ""), "error": str(exc)})

    # ── Summary ───────────────────────────────────────────────────────────────
    succeeded = len(rows) - len(failed)
    log.info(f"\n{'─'*50}")
    log.info(f"Done.  Succeeded: {succeeded}  |  Failed: {len(failed)}")

    if failed:
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(failed).to_csv(FAILED_CSV, index=False)
        log.warning(f"Failed songs written → {FAILED_CSV}")
        log.warning("Retry with: python scripts/run_pipeline.py --retry-failed")
    else:
        log.info("All songs processed successfully.")

    log.info(f"{'─'*50}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-process songs through the Hindi Music Engine pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CSV format (columns):
  title        (required)
  artist       (required)
  category     ghazal | sufi | sad_romantic | upbeat | party | unknown
  youtube_url  optional — yt-dlp searches by name if blank
  clip_start   seconds to skip before 90s clip window (default: 15)

Examples:
  # Standard run
  python scripts/run_pipeline.py --csv data/songs.csv

  # 2 workers (max on 8GB RAM)
  python scripts/run_pipeline.py --csv data/songs.csv --workers 2

  # Resume after interruption at row 30
  python scripts/run_pipeline.py --csv data/songs.csv --start-from 30

  # Retry only failed songs from last run
  python scripts/run_pipeline.py --csv data/songs.csv --retry-failed

  # Force reprocess (e.g., after feature bug fixes)
  python scripts/run_pipeline.py --csv data/songs.csv --force
        """,
    )
    parser.add_argument(
        "--csv",
        default="data/songs.csv",
        help="Path to songs CSV (default: data/songs.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1; max 2 on 8GB RAM)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N rows — for resuming interrupted runs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess songs already marked as processed in the database",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only retry songs listed in data/failed.csv",
    )
    args = parser.parse_args()

    if args.workers > 2:
        log.warning(
            f"--workers={args.workers} may cause OOM on 8GB RAM "
            "(CLAP model + demucs together use ~5–6GB). Proceeding anyway."
        )

    if not os.path.exists(args.csv):
        log.error(f"CSV not found: {args.csv}")
        sys.exit(1)

    run_batch(
        csv_path     = args.csv,
        start_from   = args.start_from,
        n_workers    = args.workers,
        force        = args.force,
        retry_failed = args.retry_failed,
    )


if __name__ == "__main__":
    main()