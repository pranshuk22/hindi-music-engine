"""
scripts/expand_golden_from_lastfm.py
───────────────────────────────────────
Auto-generates relevance judgments from Last.fm's real listener-similarity
data (pipeline/lastfm_client.py) — an external, crowd-sourced alternative
to hand-typed judgments, addressing two real problems raised about the
current golden set: it's slow for one person to grow by hand, and it
reflects only that one person's taste.

For every song in the catalog, fetches its Last.fm similar tracks and
checks which of them are ALSO in our own catalog — that overlap becomes a
graded judgment (grade derived from Last.fm's own match confidence; see
_grade_from_match, marked NEEDS CALIBRATION until run against real data).

Writes to a SEPARATE file (default data/eval/golden_relevance_lastfm.json),
never touching data/eval/golden_relevance.json — the owner's hand-reviewed
file stays authoritative and untouched. Run scripts/evaluate_golden.py
--golden data/eval/golden_relevance_lastfm.json to score against this one
independently; merging the two (or not) is a decision for later, after
spot-checking what Last.fm actually proposes for this repertoire.

Coverage note: only songs that are BOTH suggested by Last.fm AND already
in our own catalog can become judgments here — this necessarily grows with
catalog size (Phase 6 scaling), so expect this to become much more
powerful once the corpus is larger, not just more useful now.

Usage:
  python scripts/expand_golden_from_lastfm.py
  python scripts/expand_golden_from_lastfm.py --limit 5      # smoke test
  python scripts/expand_golden_from_lastfm.py --min-matches 2  # only keep anchors with >=2 judged neighbors
"""

import sys
import os
import json
import time
import argparse
import difflib
import logging

sys.path.insert(0, ".")

from utils.db import init_db, get_all_songs
from pipeline.lastfm_client import get_similar_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/eval/golden_relevance_lastfm.json"
INTER_SONG_SLEEP = 1.0


def _normalize(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).strip()


def _fuzzy_match_catalog(title: str, artist: str, catalog: list[dict], threshold: float = 0.72):
    """
    Find the best-matching song in our own catalog for a Last.fm result.
    Returns the matching song_id, or None if nothing clears the threshold.

    Uses difflib's SequenceMatcher on normalized (title, artist) strings —
    no external fuzzy-matching dependency needed for a 50-1000 song
    catalog. threshold=0.72 is a starting point, NOT calibrated against
    real Last.fm output yet (no API key available while writing this) —
    inspect false-positive/negative matches on the first real run and
    adjust; see the module docstring.
    """
    target_title = _normalize(title)
    target_artist = _normalize(artist)

    best_id, best_score = None, 0.0
    for row in catalog:
        cat_title = _normalize(row["title"])
        cat_artist = _normalize(row["artist"])
        title_sim = difflib.SequenceMatcher(None, target_title, cat_title).ratio()
        artist_sim = difflib.SequenceMatcher(None, target_artist, cat_artist).ratio()
        # Title match matters more than artist (Last.fm sometimes credits
        # featured/remix artists differently than our single-artist field).
        combined = 0.7 * title_sim + 0.3 * artist_sim
        if combined > best_score:
            best_score, best_id = combined, row["id"]

    if best_score >= threshold:
        return best_id
    return None


def _grade_from_match(match: float) -> int:
    """
    Map Last.fm's match confidence to a 0/1/2 grade.

    NEEDS CALIBRATION: these thresholds are a starting guess, not derived
    from real data (no API key was available while writing this script —
    see pipeline/lastfm_client.py's docstring on what `match` actually
    means). Before trusting this file's grades, print the actual match
    value distribution from a real run and sanity-check a sample of
    grade-2 vs grade-1 assignments by ear.
    """
    if match >= 0.4:
        return 2
    if match >= 0.1:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N songs (smoke test)")
    parser.add_argument("--min-matches", type=int, default=1,
                         help="Drop anchors with fewer than this many in-catalog judged neighbors")
    args = parser.parse_args()

    init_db()
    catalog = get_all_songs()
    songs = catalog[: args.limit] if args.limit else catalog

    log.info(f"Catalog: {len(catalog)} songs. Processing {len(songs)} as candidate anchors.")

    anchors = []
    match_values_seen = []

    for i, row in enumerate(songs, 1):
        log.info(f"[{i}/{len(songs)}] {row['title']} — {row['artist']}")
        similar = get_similar_tracks(row["title"], row["artist"], limit=30)

        judgments = []
        for s in similar:
            match_values_seen.append(s["match"])
            song_id = _fuzzy_match_catalog(s["title"], s["artist"], catalog)
            if song_id is None or song_id == row["id"]:
                continue
            grade = _grade_from_match(s["match"])
            if grade > 0:
                judgments.append({
                    "song_id": song_id,
                    "grade": grade,
                    "lastfm_match": round(s["match"], 3),
                    "lastfm_title": s["title"],
                    "lastfm_artist": s["artist"],
                })

        if len(judgments) >= args.min_matches:
            anchors.append({
                "anchor_id": row["id"],
                "note": f"Auto-generated from Last.fm similar-tracks for {row['title']} — {row['artist']}",
                "judgments": judgments,
            })
            log.info(f"  -> {len(judgments)} in-catalog judged neighbors")
        else:
            log.info(f"  -> {len(judgments)} in-catalog matches (below --min-matches, skipped as anchor)")

        time.sleep(INTER_SONG_SLEEP)

    output = {
        "_readme": (
            "AUTO-GENERATED from Last.fm real-listener similarity data "
            "(pipeline/lastfm_client.py + this script), NOT hand-judged by "
            "the project owner or drafted by an AI reviewing metadata. "
            "Grade thresholds are an uncalibrated starting guess — see "
            "_grade_from_match() in scripts/expand_golden_from_lastfm.py. "
            "This is a SEPARATE file from data/eval/golden_relevance.json "
            "(the owner-reviewed file) — spot-check a sample of these "
            "judgments before trusting them, and decide whether/how to "
            "merge with the hand-reviewed set. Score against this file "
            "independently with: python scripts/evaluate_golden.py "
            "--golden data/eval/golden_relevance_lastfm.json"
        ),
        "anchors": anchors,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"\nWrote {len(anchors)} anchors to {args.output}")
    if match_values_seen:
        mv = sorted(match_values_seen)
        log.info(
            f"Last.fm match value distribution: min={mv[0]:.3f} "
            f"median={mv[len(mv)//2]:.3f} max={mv[-1]:.3f} "
            f"(n={len(mv)}) — use this to sanity-check/recalibrate "
            f"_grade_from_match() thresholds."
        )


if __name__ == "__main__":
    main()
