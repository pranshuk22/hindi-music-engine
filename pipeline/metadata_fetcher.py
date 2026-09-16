"""
pipeline/metadata_fetcher.py
───────────────────────────────
Automated, scalable song metadata enrichment via the Wikidata API.

This REPLACES the hand-curated data/composer_map.json approach. That file
was hand-typed from memory for 50 songs to quickly test whether "composer
as a signal" was worth pursuing at all — it was — but hand-typing metadata
does not scale to hundreds or thousands of songs, and the whole point of
this pipeline is to keep working unattended as the catalog grows. This
module is the real, automated replacement: given a title + artist, it
queries Wikidata's public API and returns whatever structured metadata is
actually published, or None for fields that aren't — it never guesses.

Why Wikidata over MusicBrainz: both were evaluated (2026-09-15). MusicBrainz
has an established API but proved unreliable in testing (repeated "server
busy" errors) and its Bollywood/Hindustani coverage is inconsistent.
Wikidata's coverage for Bollywood film songs is strong (composer P86,
lyricist P676, performer P175, publication date P577, genre P136 were all
populated on the first song tested), it's a single well-documented JSON API
with no auth, and — per manual Wikipedia exploration earlier in this
project — Bollywood soundtrack pages sometimes even carry the specific
raga name, a real lead for eventually fixing the fabricated
raga_probability heuristic (see FEATURES.md and plan.md Phase 5), though
that is not implemented here.

No hardcoded per-song knowledge lives in this file. If Wikidata's coverage
for a song is thin, the fields come back None — same "leave it blank
rather than guess" discipline as everywhere else in this pipeline
(lyrics_missing, raga_missing, composer_map.json's own confidence gaps).

Rate limiting: short sleep between the two API calls per lookup (search,
then batch label-resolution) — Wikidata is not as aggressively throttled as
MusicBrainz/Genius but a descriptive User-Agent and politeness delay is
required Wikimedia API etiquette, not optional.
"""

import time
import logging
import requests

logger = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "HindiMusicEngine/0.1 (personal research project)"
REQUEST_TIMEOUT = 10
RATE_LIMIT_SLEEP = 0.7
# Tested empirically (2026-09-15): a tight loop over several songs with
# only 1.5s between full fetch_song_metadata() calls started silently
# losing the second (claims) request to rate limiting, causing a
# genuinely-matched song to come back with all-None fields — not a search/
# scoring bug, confirmed by re-running the same call in isolation and
# getting the correct result. Callers doing a batch backfill (e.g.
# scripts/fetch_metadata.py) MUST sleep at least ~2s between songs on top
# of this module's own internal spacing.

# Property IDs on a Wikidata "song" / "single" / "musical work" item.
P_COMPOSER     = "P86"
P_LYRICIST     = "P676"
P_PERFORMER    = "P175"
P_PUB_DATE     = "P577"
P_GENRE        = "P136"

# Description keyword tiers for disambiguating search hits. Wikidata often
# has TWO items per song — a "vocal track" / "recording" sub-item (usually
# missing composer/lyricist, which live on the main work) and the actual
# song item (whose description tends to explicitly say "composed by" or
# "written by"). Tested empirically (2026-09-15): a flat score tied these
# two for "Tum Hi Ho" and a naive max() picked the wrong (recording) one —
# this tiering fixes that by weighting "composed by" far above generic
# song-ish words, and penalizing the "vocal track" tell.
_STRONG_HINTS = ("composed by", "written by", "written and composed")
_WEAK_HINTS   = ("song", "single", "soundtrack", "album")
_PENALTY_HINTS = ("vocal track",)
# NOTE (2026-09-15): many Bollywood songs (e.g. "Dil Chahta Hai") have no
# standalone song-level Wikidata item at all — only a FILM item and an
# ALBUM item. Checked empirically: the album item generally does NOT carry
# P86/composer or P676/lyricist (those properties don't apply at
# album-granularity the way Wikidata is normally modeled), only P136/genre
# and similar. Accepting album matches therefore recovers genre (and
# sometimes year) for a real chunk of songs that would otherwise get
# nothing, but composer/lyricist will still correctly come back None for
# them — that's not a bug, the data isn't there at that granularity.



# Sentinel distinguishing "the request itself never completed" (rate
# limited, network error, exhausted retries) from "the request succeeded
# and there was genuinely nothing to find." Conflating these was a real bug
# caught in testing (2026-09-15): a burst of 429s during a batch backfill
# was silently recorded as "no confident match" for several songs, which
# would have permanently (falsely) marked them as checked-and-absent
# instead of leaving them for a retry.
class _RequestFailed(Exception):
    pass


def _get(params: dict, retries: int = 5) -> dict:
    """Returns the parsed JSON response. Raises _RequestFailed if the
    request never completed after all retries — callers MUST catch this
    and treat it as 'unknown, retry later', never as 'confirmed absent'."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                WIKIDATA_API,
                params={**params, "format": "json"},
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                wait = 4 * (attempt + 1)
                logger.info(f"Wikidata rate-limited (429) — backing off {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError:
            raise _RequestFailed(f"HTTP error after status check")
        except Exception as exc:
            raise _RequestFailed(str(exc))
    raise _RequestFailed("exhausted retries after repeated 429s")


def _search_song_entity(title: str, artist: str) -> str | None:
    """
    Search Wikidata for the best-matching song entity, return its Q-id, or
    None if nothing clears the confidence bar. See tiering/threshold
    constants above for the disambiguation rules — both were derived from
    actual mismatches found during testing, not guessed upfront.

    Raises _RequestFailed (does not return None) if the search request
    itself never completed — the caller must not treat that the same as a
    confirmed empty result.
    """
    data = _get({
        "action": "wbsearchentities",
        "search": title,
        "language": "en",
        "limit": 10,
    })
    if not data.get("search"):
        return None

    results = data["search"]

    def _evaluate(r: dict) -> tuple[int, bool]:
        """Returns (score for ranking among accepted candidates, accept: bool).

        Acceptance is a rule, not a scalar threshold — tested empirically
        (2026-09-15) that a single cutoff let a title-only match through
        with no artist confirmation at all (a same-titled song by a
        completely different artist scored just high enough on a generic
        "song"-ish description alone). Requiring either a strong
        "composed by" hint (self-sufficient — that phrase only appears on
        the real work item) or a weak hint PLUS an explicit artist-name
        match closes that gap without losing genuinely good matches whose
        description is thinner (e.g. just "single by <artist>").
        """
        desc = (r.get("description") or "").lower()
        has_strong  = any(h in desc for h in _STRONG_HINTS)
        has_weak    = any(h in desc for h in _WEAK_HINTS)
        has_penalty = any(h in desc for h in _PENALTY_HINTS)
        has_artist  = bool(artist) and artist.lower() in desc

        score = 0
        if has_strong:  score += 5
        if has_weak:    score += 2
        if has_penalty: score -= 3
        if has_artist:  score += 1

        accept = has_strong or (has_weak and has_artist)
        return score, accept

    evaluated = [(r, *_evaluate(r)) for r in results]
    accepted = [(r, score) for r, score, accept in evaluated if accept]
    if not accepted:
        return None

    best, _ = max(accepted, key=lambda pair: pair[1])
    return best.get("id")


def _resolve_labels(qids: list[str]) -> dict[str, str]:
    """Batch-resolve a list of Q-ids to their English labels. Raises
    _RequestFailed if the request never completed (see _get)."""
    qids = [q for q in dict.fromkeys(qids) if q]  # dedupe, preserve order
    if not qids:
        return {}
    data = _get({
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels",
        "languages": "en",
    })
    out = {}
    for qid, entity in data.get("entities", {}).items():
        label = entity.get("labels", {}).get("en", {}).get("value")
        if label:
            out[qid] = label
    return out


def fetch_song_metadata(title: str, artist: str) -> dict:
    """
    Look up composer, lyricist, genre, and year for a song via Wikidata.

    Returns:
        {
            "composer": str | None,
            "lyricist": str | None,
            "genre": list[str],       # possibly empty
            "year": int | None,
            "wikidata_id": str | None,  # for debugging / manual spot-checks
            "lookup_failed": bool,      # True = the request itself failed
                                        # (rate limit, network) — treat as
                                        # "unknown, retry later", NOT as a
                                        # confirmed absence. False + all
                                        # None fields means Wikidata was
                                        # successfully queried and genuinely
                                        # has no confident match.
        }
    Fields are None/empty rather than guessed when genuinely not found.
    Never raises — network/rate-limit failures are caught here and
    reported via lookup_failed so a batch backfill can retry them later
    instead of silently recording them as absent.
    """
    result = {
        "composer": None, "lyricist": None, "genre": [], "year": None,
        "wikidata_id": None, "lookup_failed": False,
    }

    try:
        qid = _search_song_entity(title, artist)
    except _RequestFailed as exc:
        logger.warning(f"Wikidata search failed for '{title}' — '{artist}': {exc}")
        result["lookup_failed"] = True
        return result

    if not qid:
        return result
    result["wikidata_id"] = qid

    time.sleep(RATE_LIMIT_SLEEP)
    try:
        entity_data = _get({
            "action": "wbgetentities",
            "ids": qid,
            "props": "claims",
        })
    except _RequestFailed as exc:
        logger.warning(f"Wikidata claims fetch failed for '{title}': {exc}")
        result["lookup_failed"] = True
        return result

    claims = entity_data.get("entities", {}).get(qid, {}).get("claims", {})

    def _entity_ids(prop: str) -> list[str]:
        out = []
        for c in claims.get(prop, []):
            v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(v, dict) and v.get("id"):
                out.append(v["id"])
        return out

    composer_qids = _entity_ids(P_COMPOSER)
    lyricist_qids = _entity_ids(P_LYRICIST)
    genre_qids    = _entity_ids(P_GENRE)

    # Publication date is a plain time value, not a Q-id — extract the year directly.
    for c in claims.get(P_PUB_DATE, []):
        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(v, dict) and v.get("time"):
            try:
                result["year"] = int(v["time"][1:5])  # "+2013-00-00T..." -> 2013
            except (ValueError, IndexError):
                pass
        break

    time.sleep(RATE_LIMIT_SLEEP)
    try:
        labels = _resolve_labels(composer_qids + lyricist_qids + genre_qids)
    except _RequestFailed as exc:
        logger.warning(f"Wikidata label resolution failed for '{title}': {exc}")
        result["lookup_failed"] = True
        return result

    if composer_qids:
        result["composer"] = labels.get(composer_qids[0])
    if lyricist_qids:
        result["lyricist"] = labels.get(lyricist_qids[0])
    result["genre"] = [labels[q] for q in genre_qids if q in labels]

    return result
