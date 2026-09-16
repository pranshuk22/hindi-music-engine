"""
scripts/fetch_metadata.py
────────────────────────────
Automated metadata backfill: composer, lyricist, genre, release year — via
pipeline/metadata_fetcher.py (Wikidata), NOT hand-typed knowledge.

This is the scalable replacement for the one-off data/composer_map.json +
the now-removed apply_composer_metadata.py. The same fetcher this script
calls is also wired into pipeline/processor.py, so every NEW song gets
this automatically at ingestion time — this script exists only to backfill
songs that were processed before that wiring existed, and is safe to
re-run at any time (skips songs that already have a composer OR lyricist
OR genre set, unless --force).

No manual per-song knowledge lives here or in metadata_fetcher.py. If
Wikidata doesn't have a confident match for a song, the fields stay NULL —
never guessed. This is expected to scale to any catalog size: the only
per-song cost is two lightweight API calls plus politeness sleep, run
once per song, ever.

Usage:
  python scripts/fetch_metadata.py
  python scripts/fetch_metadata.py --force
  python scripts/fetch_metadata.py --limit 5
"""

import sys
import time
import argparse
import logging

sys.path.insert(0, ".")

from utils.db import init_db, get_all_songs, update_song_metadata
from pipeline.metadata_fetcher import fetch_song_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Spacing between full per-song lookups (each does ~2 internal API calls).
# Increased 2.0 -> 4.0 after a real backfill run (2026-09-15) hit a storm
# of 429s partway through even with 2.0s spacing — likely a cumulative
# per-IP throttle from all of this session's earlier testing, not purely
# this script's own pace. The lookup_failed distinction (see
# metadata_fetcher.py) means a too-fast run now safely retries instead of
# corrupting data either way, but slower is still better than triggering
# it in the first place.
INTER_SONG_SLEEP = 4.0


def _already_enriched(row) -> bool:
    return bool(row["composer"] or row["lyricist"] or row["genre"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Redo even already-enriched songs")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    init_db()
    songs = get_all_songs()
    if args.limit:
        songs = songs[: args.limit]

    log.info(f"Found {len(songs)} songs.")

    enriched, skipped, no_match, failed = 0, 0, 0, []

    for i, row in enumerate(songs, 1):
        song_id = row["id"]

        if not args.force and _already_enriched(row):
            skipped += 1
            log.info(f"[{i}/{len(songs)}] SKIP (already enriched): {song_id}")
            continue

        log.info(f"[{i}/{len(songs)}] Fetching metadata: {row['title']} — {row['artist']}")
        meta = fetch_song_metadata(row["title"], row["artist"])

        if meta["lookup_failed"]:
            # Request itself failed (rate limit/network) — NOT a confirmed
            # absence. Do not write anything; leave for a re-run (this
            # script skips already-enriched songs, so a plain re-run will
            # naturally retry exactly these, as long as --force isn't
            # passed again for songs that DID succeed).
            failed.append(song_id)
            log.warning(f"  LOOKUP FAILED (will retry on next run, not marked as no-match)")
        elif not meta["wikidata_id"]:
            no_match += 1
            log.info(f"  (no confident Wikidata match — confirmed, left null)")
        else:
            update_song_metadata(
                song_id,
                composer=meta["composer"],
                lyricist=meta["lyricist"],
                genre=meta["genre"],
                release_year=meta["year"],
                wikidata_id=meta["wikidata_id"],
            )
            if meta["composer"] or meta["lyricist"] or meta["genre"]:
                enriched += 1
                log.info(
                    f"  ✓ composer={meta['composer']!r}  lyricist={meta['lyricist']!r}  "
                    f"genre={meta['genre']}  year={meta['year']}"
                )
            else:
                log.info(f"  (matched {meta['wikidata_id']} but no composer/lyricist/genre claims)")

        time.sleep(INTER_SONG_SLEEP)

    log.info(
        f"\nEnriched: {enriched}  |  Skipped (already had data): {skipped}  |  "
        f"No confident match (confirmed): {no_match}  |  Lookup failed (retry needed): {len(failed)}"
    )
    if failed:
        log.info(f"Re-run this script to retry: {failed}")


if __name__ == "__main__":
    main()
