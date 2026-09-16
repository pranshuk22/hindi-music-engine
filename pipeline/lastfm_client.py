"""
pipeline/lastfm_client.py
────────────────────────────
Fetches Last.fm's `track.getsimilar` — real listener co-occurrence/tag-based
similarity, not editorial metadata — as an external, crowd-sourced ground
truth source for data/eval/golden_relevance.json (see scripts/
expand_golden_from_lastfm.py).

Why this exists: the golden set was entirely hand-judged by the project
owner (or, in its first draft, by an AI reviewing metadata). Both have a
real "one perspective, not universally applicable" problem. Last.fm's
similarity graph is built from many real listeners' scrobbles/co-plays —
still not a perfectly unbiased ground truth (its user base skews toward a
particular kind of listener), but a meaningfully different and broader
perspective than any single judge, and near-zero effort to query at scale.

Same discipline as every other external-data module in this pipeline
(metadata_fetcher.py, lyrics_extractor.py): never fabricate a match, no
hardcoded per-song knowledge, fail soft (empty list) rather than raise, so
a batch run degrades gracefully rather than crashing on one bad lookup.

Token:
  Set LASTFM_API_KEY in your environment or a .env file (mirrors
  GENIUS_TOKEN's handling in lyrics_extractor.py). Get a free key at
  https://www.last.fm/api/account/create — no approval wait, no cost.
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)

LASTFM_API = "https://ws.audioscrobbler.com/2.0/"
REQUEST_TIMEOUT = 10

_api_key = None


def _get_api_key() -> str:
    global _api_key
    if _api_key is not None:
        return _api_key

    from dotenv import load_dotenv
    load_dotenv()

    key = os.environ.get("LASTFM_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "LASTFM_API_KEY environment variable is not set. "
            "Get a free key at https://www.last.fm/api/account/create and add it to your .env file."
        )
    _api_key = key
    return _api_key


def get_similar_tracks(title: str, artist: str, limit: int = 30) -> list[dict]:
    """
    Query Last.fm for tracks similar to (title, artist).

    Returns a list of dicts, ranked by Last.fm's own relevance order:
        [{"title": str, "artist": str, "match": float}, ...]
    `match` is Last.fm's own similarity confidence in [0, 1] (NOT
    normalized/comparable across different query tracks in any strict
    sense — Last.fm's own docs describe it as a relative relevance score
    for that one query, not a universal similarity metric — so use it for
    within-query ranking/grading, not for comparing across anchors).

    Returns an empty list (never raises) if the track isn't found, the API
    key is missing, or the request fails — callers doing a batch run should
    treat that as "no data available for this song," not a hard error.
    """
    try:
        api_key = _get_api_key()
    except EnvironmentError as exc:
        logger.warning(str(exc))
        return []

    try:
        resp = requests.get(
            LASTFM_API,
            params={
                "method": "track.getsimilar",
                "artist": artist,
                "track": title,
                "api_key": api_key,
                "format": "json",
                "limit": limit,
                "autocorrect": 1,  # let Last.fm fix minor title/artist typos
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"Last.fm request failed for '{title}' — '{artist}': {exc}")
        return []

    if "error" in data:
        # Common case: track not found in Last.fm's catalog. Not an error
        # worth raising on — just means no data for this song.
        logger.info(f"Last.fm: no data for '{title}' — '{artist}' ({data.get('message', '')})")
        return []

    tracks = data.get("similartracks", {}).get("track", [])
    out = []
    for t in tracks:
        try:
            out.append({
                "title": t["name"],
                "artist": t["artist"]["name"],
                "match": float(t.get("match", 0.0)),
            })
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed entries rather than fail the whole batch

    return out
