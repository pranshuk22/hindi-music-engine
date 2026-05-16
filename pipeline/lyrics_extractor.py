"""
pipeline/lyrics_extractor.py

Fetches lyrics from Genius API for Hindi/Urdu/Bollywood songs.

Spec rules:
- Use lyricsgenius (Genius API free tier)
- Handle Devanagari and Romanised Hindi input
- For Urdu-script lyrics (ghazals), pass raw — the multilingual model handles it
- If Genius returns no result: return (None, True) so processor sets lyrics_missing=1
  and uses a zero NLP vector with redistributed weights
- Rate limit: ~1 req/sec — use sleep(1) between requests
- Do NOT hallucinate or approximate lyrics

Token:
  Set GENIUS_TOKEN in your environment or a .env file.
  Never hardcode the token in source.
"""

import os
import re
import time
import logging

logger = logging.getLogger(__name__)

# ── Lazy client initialisation ────────────────────────────────────────────────
# Initialised on first call so importing this module never crashes if
# lyricsgenius isn't installed or the token is missing.
_genius_client = None

def _get_client():
    global _genius_client
    if _genius_client is not None:
        return _genius_client

    # --- NEW: Force Python to read the .env file ---
    from dotenv import load_dotenv
    load_dotenv() 
    # -----------------------------------------------

    token = os.environ.get("GENIUS_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "GENIUS_TOKEN environment variable is not set. "
            "Get a free token at https://genius.com/api-clients and add it to your .env file."
        )

    import lyricsgenius
    client = lyricsgenius.Genius(
        token,
        timeout              = 15,
        retries              = 3,
        remove_section_headers = True,  # strip [Verse], [Chorus] etc.
        skip_non_songs       = True,    # skip albums / descriptions
    )
    client.verbose = False  # silence lyricsgenius console output (some versions ignore the constructor arg)
    _genius_client = client
    return client


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _clean_lyrics(raw: str) -> str:
    """
    Strip Genius-specific boilerplate that pollutes the NLP embedding.

    Removes:
      - Trailing "NNNEmbed" artifact (e.g. "543Embed")
      - Leading title line that Genius prepends (e.g. "Tum Hi Ho Lyrics")
      - Excessive blank lines
    """
    text = raw

    # Remove trailing digit+Embed artifact
    text = re.sub(r"\d+Embed$", "", text.strip())

    # Remove leading "Song Title Lyrics" line if present
    lines = text.splitlines()
    if lines and lines[0].lower().endswith("lyrics"):
        lines = lines[1:]
    text = "\n".join(lines).strip()

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_lyrics(title: str, artist: str) -> str:
    """
    Search Genius for lyrics and return a (lyrics, lyrics_missing) tuple.

    Args:
        title:   Song title — Devanagari, Romanised Hindi, or English title all work.
        artist:  Artist name.

    Returns:
        (lyrics_text, False)  — lyrics found and cleaned
        (None, True)          — not found or fetch error; caller should zero the NLP vector
    """
    try:
        genius = _get_client()

        logger.debug("Fetching lyrics: %s — %s", title, artist)
        # HACK: Combine title and artist to bypass strict artist-matching
        search_query = f"{title} {artist}"
        song = genius.search_song(search_query)
        # Respect Genius rate limit
        time.sleep(1)

        if song and song.lyrics:
            lyrics = _clean_lyrics(song.lyrics)
            if lyrics:
                logger.info("  Lyrics found (%d chars)", len(lyrics))
                return lyrics

        logger.info("  Lyrics not found for '%s' — will use zero NLP vector.", title)
        return

    except Exception as exc:
        logger.warning("Lyrics fetch failed for '%s — %s': %s", title, artist, exc)
        return
    
# ─── Standalone Smoke Test ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Configure basic logging to see details in the terminal
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # Using the first song from your dataset as a guinea pig
    test_title = "Aaj Jaane Ki Zid Na Karo"
    test_artist = "Farida Khanum"
    
    print(f"\n🚀 Running standalone lyrics fetch for: {test_title} — {test_artist}...")
    lyrics_sample = fetch_lyrics(test_title, test_artist)
    
    print("\n" + "="*50)
    if lyrics_sample:
        print("✅ TEST PASSED: Lyrics fetched and cleaned successfully!")
        print("="*50)
        print("\n📄 Preview of the first 400 characters:\n")
        print(lyrics_sample[:400] + "\n\n... [truncated] ...")
    else:
        print("❌ TEST FAILED: Returned an empty string.")
        print("Possible causes:")
        print("  1. Your GENIUS_TOKEN in the .env file is wrong or missing.")
        print("  2. Genius API is blocking the request (check internet connection).")
    print("="*50 + "\n")