"""
scripts/recover_lyrics_mood.py
─────────────────────────────────
Backfill raw lyric text (data/lyrics/<song_id>.txt) and a mood tag
(feat["mood"], see pipeline/mood_tagger.py) for songs processed before
2026-09-15, when only the lyric embedding was persisted — not the raw text.

Lean and cheap compared to CLAP recovery: no audio download, no GPU, just
a Genius API text lookup per song (~1-2s + rate-limit sleep). Does NOT
touch the NLP embedding (data/nlp/*.npy) or the fused vector — those are
still correct and don't need redoing. This only adds the mood tag as a new
Stage 2 rerank feature (index/rerank.py's `mood` scorer, currently inert
at weight 0.0 until swept against the golden set).

Resumable: skips songs that already have data/lyrics/<song_id>.txt unless
--force.

Usage:
  python scripts/recover_lyrics_mood.py
  python scripts/recover_lyrics_mood.py --force
  python scripts/recover_lyrics_mood.py --limit 5
"""

import sys
import os
import json
import time
import argparse
import logging

sys.path.insert(0, ".")

from utils.db import init_db, get_all_songs
from pipeline.lyrics_extractor import fetch_lyrics
from pipeline.mood_tagger import tag_mood

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

LYRICS_DIR = "data/lyrics"
os.makedirs(LYRICS_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    init_db()
    songs = get_all_songs()
    if args.limit:
        songs = songs[: args.limit]

    log.info(f"Found {len(songs)} songs.")

    recovered, skipped, no_lyrics, tagged = 0, 0, 0, 0
    mood_counts = {}

    for i, row in enumerate(songs, 1):
        song_id = row["id"]
        lyrics_path = os.path.join(LYRICS_DIR, f"{song_id}.txt")

        if not args.force and os.path.exists(lyrics_path):
            skipped += 1
            log.info(f"[{i}/{len(songs)}] SKIP (already have lyrics): {song_id}")
            continue

        log.info(f"[{i}/{len(songs)}] Fetching lyrics: {row['title']} — {row['artist']}")
        lyrics = fetch_lyrics(row["title"], row["artist"])

        if not lyrics or not lyrics.strip():
            no_lyrics += 1
            log.info(f"  (no lyrics found — consistent with lyrics_missing flag)")
            time.sleep(1)
            continue

        with open(lyrics_path, "w") as f:
            f.write(lyrics)
        recovered += 1

        mood = tag_mood(lyrics)
        if mood:
            tagged += 1
            mood_counts[mood] = mood_counts.get(mood, 0) + 1

        features_path = row["features_path"]
        if features_path and os.path.exists(features_path):
            with open(features_path) as f:
                feat = json.load(f)
            feat["mood"] = mood
            with open(features_path, "w") as f:
                json.dump(feat, f, indent=2)

        log.info(f"  ✓ {song_id}  mood={mood}")
        time.sleep(1)  # polite pause between Genius requests

    log.info(
        f"\nRecovered lyrics: {recovered}  |  Skipped (already had): {skipped}  |  "
        f"No lyrics found: {no_lyrics}  |  Tagged with a mood: {tagged}"
    )
    log.info(f"Mood distribution: {mood_counts}")
    log.info(
        "\nMood is now available as a Stage 2 rerank signal (index/rerank.py "
        "'mood' scorer) but defaults to weight 0.0 (inert). Sweep it with:\n"
        "  python scripts/evaluate_golden.py --rerank-weight mood=0.1"
    )


if __name__ == "__main__":
    main()
