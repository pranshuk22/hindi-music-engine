"""
pipeline/mood_tagger.py
──────────────────────────
V1 lyric mood/theme tagger — keyword-based, NOT a trained classifier.

Answers a direct question raised during this project: lyrics currently only
contribute a semantic embedding (nlp_embedder.py) to the fused vector; there
is no explicit, interpretable mood/theme signal derived from them anywhere.
This is a first cut at one, in the same spirit — and under the same
discipline — as data/composer_map.json: a plausible heuristic that MUST be
validated by ablation against the golden set before being trusted, not
assumed to help because the category names sound sensible.

Method: count keyword hits per category (Romanized + Devanagari, since
Genius returns either depending on the song) in the raw lyric text, assign
the category with the most hits (ties or zero hits -> None). This is
deliberately simple/interpretable rather than a black-box sentiment model —
consistent with this project's broader lesson that unvalidated ML-flavoured
heuristics (see: the old raga_probability implementation) are worse than
no signal at all until proven otherwise. If this DOES prove out via
ablation, upgrading to a real multilingual sentiment/emotion model later is
a natural next step; if it doesn't help, a heavier model likely wouldn't
either, and this was the cheap way to find that out.

Categories are intentionally coarse and Bollywood/ghazal/sufi-specific,
not a generic sentiment scale (positive/negative/neutral would be far less
informative for this domain — see FEATURES.md's own framing of category
labels like ghazal/sufi/sad_romantic/party).
"""

import re

# Keyword lists: Romanized (transliterated) + Devanagari. Not exhaustive —
# tuned for common Bollywood/ghazal/qawwali/sufi vocabulary, not general
# Hindi/Urdu text. Case-insensitive substring match on Romanized forms.
_KEYWORDS = {
    "devotional": [
        # Romanized
        "khuda", "rab", "allah", "khwaja", "maula", "rehmat", "ibaadat",
        "sajda", "bandagi", "karam", "noor", "dargah", "kalandar", "mast qalandar",
        "ishwar", "bhagwan", "prabhu", "hari om", "bhakti", "aarti", "sufiyana",
        "rooh", "khudaya", "meherbaan", "duaa", "namaz", "haq", "manzil-e-noor",
        # Devanagari
        "ख़ुदा", "खुदा", "रब", "अल्लाह", "मौला", "ईश्वर", "भगवान", "प्रभु",
        "इबादत", "करम", "नूर", "दुआ", "भक्ति", "आरती",
    ],
    "romantic": [
        "pyaar", "pyar", "ishq", "mohabbat", "dilbar", "mehbooba", "sanam",
        "dilruba", "chahat", "prem", "deewana", "deewani", "dil", "aashiqui",
        "jaana", "mehboob", "dildaar", "yaariyan", "wajood",
        "प्यार", "इश्क़", "मोहब्बत", "दिल", "आशिकी", "जाना", "महबूब",
    ],
    "sad_longing": [
        "judaai", "judai", "tanhai", "tanhaai", "gham", "dard", "aansu",
        "bewafa", "bichadna", "tadap", "viraha", "dukh", "akela", "akeli",
        "bichhad", "rulaana", "yaad", "khoya", "kho gaya", "bujh gaya",
        "जुदाई", "तन्हाई", "ग़म", "गम", "दर्द", "आंसू", "बेवफा", "तड़प",
        "दुख", "अकेला", "याद",
    ],
    "celebratory_party": [
        "party", "naach", "jhoom", "masti", "hungama", "dance", "shaadi",
        "baaraat", "disco", "saturday", "high rated", "chull", "thumka",
        "balle balle", "gabru", "nasha", "chashma",
        "पार्टी", "नाच", "झूम", "मस्ती", "हंगामा", "शादी", "बारात",
    ],
}


def tag_mood(lyrics_text: str) -> str | None:
    """
    Return the best-matching mood category, or None if no category gets a
    clear hit (fewer than 1 keyword match, or a tie between categories).

    Args:
        lyrics_text: Raw lyric text (Romanized or Devanagari), as returned
                     by pipeline.lyrics_extractor.fetch_lyrics().

    Returns:
        One of "devotional", "romantic", "sad_longing", "celebratory_party",
        or None (no confident match — left as a gap rather than a guess,
        same discipline as composer_map.json leaving low-confidence
        composers null).
    """
    if not lyrics_text or not lyrics_text.strip():
        return None

    text_lower = lyrics_text.lower()
    scores = {}
    for category, keywords in _KEYWORDS.items():
        count = 0
        for kw in keywords:
            # Devanagari keywords aren't lowercased meaningfully but the
            # lower() call above is harmless for them; substring match works
            # the same either way.
            count += len(re.findall(re.escape(kw.lower()), text_lower))
        scores[category] = count

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    if best_score == 0:
        return None

    # Tie-check: if another category matches the same top count, it's not
    # a confident call — leave it unlabeled rather than pick arbitrarily.
    tied = [cat for cat, s in scores.items() if s == best_score]
    if len(tied) > 1:
        return None

    return best_category
