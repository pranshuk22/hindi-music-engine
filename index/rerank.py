"""
index/rerank.py
──────────────────
Stage 2 reranker: explicit, interpretable feature-based scoring applied on
top of the Stage 1 (FAISS cosine) candidate pool.

Architectural rationale (plan.md Phase 4): a single blind fused vector with
hand-picked scalar weights can't be right for every kind of similarity at
once — a weight that helps ghazal-vs-ghazal matching can hurt party-vs-party
matching. The fix isn't a better blind fusion formula, it's separating
"get a broad, reasonably relevant candidate pool" (Stage 1 — still the
existing FAISS search over the fused vector) from "precisely rank those
candidates" (Stage 2 — this module), where each signal is explicit, its
own weight, and independently tunable/ablatable.

This generalizes what was previously a single hardcoded composer-boost in
index/search.py into a small, extensible framework: adding a new rerank
signal means writing one scorer function and adding one line to SCORERS —
not re-deriving a global fusion formula and hoping.

Each scorer takes (anchor_ctx, candidate_ctx) -> float and should return a
value roughly in [0, 1] (or [-1, 1] for the cosine-based ones) so weights
stay comparable across signals. build_context() assembles the dict a scorer
needs from a DB row (features JSON + NLP embedding + composer).

Weight discipline (see experiment_log.md): a weight only gets a nonzero
default after being swept against the golden set, same as composer's 0.15.
Every other weight defaults to 0.0 (inert) until it earns its place —
that discipline is the entire point of this module existing.
"""

import os
import json
import numpy as np

_features_cache: dict[str, dict] = {}


def _load_features(features_path: str) -> dict:
    if not features_path or not os.path.exists(features_path):
        return {}
    if features_path not in _features_cache:
        with open(features_path) as f:
            _features_cache[features_path] = json.load(f)
    return _features_cache[features_path]


def build_context(row) -> dict:
    """
    Assemble everything a Stage 2 scorer might need for one song, from its
    DB row. Cheap — features JSON is cached per-process, NLP embedding is
    a single small .npy load.
    """
    feat = _load_features(row["features_path"])
    nlp_emb = None
    nlp_path = row["nlp_path"]
    if nlp_path and os.path.exists(nlp_path):
        nlp_emb = np.load(nlp_path).astype(np.float32)
    return {
        "composer":           row["composer"],
        "lyricist":           row["lyricist"],   # automated Wikidata fetch, see pipeline/metadata_fetcher.py
        "tempo":              feat.get("tempo"),
        "vocal_energy_ratio": feat.get("vocal_energy_ratio"),
        "tonnetz_mean":       feat.get("tonnetz_mean"),
        "mood":               feat.get("mood"),  # Phase 3 lyric mood tag; None until computed
        "nlp_emb":            nlp_emb,
    }


# ─── Individual scorers ───────────────────────────────────────────────────────

def _score_composer(a: dict, c: dict) -> float:
    """1.0 if same (non-null) composer, else 0.0."""
    return 1.0 if (a["composer"] and c["composer"] and a["composer"] == c["composer"]) else 0.0


def _score_lyricist(a: dict, c: dict) -> float:
    """1.0 if same (non-null) lyricist, else 0.0. Automated Wikidata field, added 2026-09-15 — untested, starts at weight 0.0 like every other new signal."""
    return 1.0 if (a["lyricist"] and c["lyricist"] and a["lyricist"] == c["lyricist"]) else 0.0


def _score_vocal_energy(a: dict, c: dict) -> float:
    """1.0 = identical vocal-vs-instrumental balance, 0.0 = maximally different (both in [0,1])."""
    av, cv = a["vocal_energy_ratio"], c["vocal_energy_ratio"]
    if av is None or cv is None:
        return 0.0
    return 1.0 - abs(av - cv)


def _score_tempo(a: dict, c: dict) -> float:
    """1.0 = identical BPM, decaying linearly to 0.0 at a 60 BPM gap (roughly one tempo-bucket span)."""
    at, ct = a["tempo"], c["tempo"]
    if at is None or ct is None:
        return 0.0
    return max(0.0, 1.0 - abs(at - ct) / 60.0)


def _score_tonnetz(a: dict, c: dict) -> float:
    """Cosine similarity between 6d tonnetz (harmonic-relationship) vectors, roughly [-1, 1]."""
    ta, tc = a["tonnetz_mean"], c["tonnetz_mean"]
    if not ta or not tc:
        return 0.0
    ta = np.asarray(ta, dtype=np.float32)
    tc = np.asarray(tc, dtype=np.float32)
    na, nc = float(np.linalg.norm(ta)), float(np.linalg.norm(tc))
    if na < 1e-8 or nc < 1e-8:
        return 0.0
    return float(np.dot(ta, tc) / (na * nc))


def _score_lyric(a: dict, c: dict) -> float:
    """
    Cosine similarity between lyric NLP embeddings (already L2-normalised,
    so this is a plain dot product), roughly [-1, 1].

    Note: this overlaps with what Stage 1 already sees, since the NLP
    embedding is also fused into the indexed vector. Scoring it again here
    is deliberate, not double-counting by accident — it lets this signal's
    influence be tuned independently of whatever weight the blind fusion
    formula happened to give it, same rationale as re-scoring composer
    explicitly even though vocal/timbre features partially proxy for it.
    """
    ae, ce = a["nlp_emb"], c["nlp_emb"]
    if ae is None or ce is None:
        return 0.0
    if np.allclose(ae, 0) or np.allclose(ce, 0):
        return 0.0  # zero vector means lyrics_missing — no real signal here
    return float(np.dot(ae, ce))


def _score_mood(a: dict, c: dict) -> float:
    """1.0 if same lyric mood/theme tag, else 0.0. Inert until Phase 3 mood tagging exists."""
    return 1.0 if (a["mood"] and c["mood"] and a["mood"] == c["mood"]) else 0.0


SCORERS = {
    "composer":     _score_composer,
    "lyricist":     _score_lyricist,
    "vocal_energy": _score_vocal_energy,
    "tempo":        _score_tempo,
    "tonnetz":      _score_tonnetz,
    "lyric":        _score_lyric,
    "mood":         _score_mood,
}

# RETRACTED 2026-09-15: composer previously defaulted to 0.15 here, swept
# against a `data/composer_map.json` that turned out to still be hand-typed
# (AI-guessed from memory) despite being labelled Phase 2 metadata. Once
# replaced with the real automated Wikidata fetcher (pipeline/metadata_fetcher.py),
# real coverage on this 49-song pilot is only 16% (8/50) and composer's
# effect on the golden set drops to exactly 0.0 at every weight tested —
# it never overlaps an anchor and a candidate. The mechanism is verified
# correct and Wikidata's data is trustworthy where present; the problem is
# coverage at this corpus size, not the signal itself. Re-sweep once the
# corpus is larger (Phase 6) and coverage is meaningfully higher — do not
# reintroduce a nonzero default before that. Every weight below is 0.0
# (inert) until it earns a nonzero value via a real sweep on real data —
# see experiment_log.md for the full history of what NOT to assume here.
DEFAULT_WEIGHTS = {
    "composer":     0.0,
    "lyricist":     0.0,
    "vocal_energy": 0.0,
    "tempo":        0.0,
    "tonnetz":      0.0,
    "lyric":        0.0,
    "mood":         0.0,
}


def rerank_score(anchor_ctx: dict, candidate_ctx: dict, weights: dict) -> float:
    """Weighted sum of every scorer with a nonzero weight, for one candidate."""
    total = 0.0
    for name, scorer in SCORERS.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        total += w * scorer(anchor_ctx, candidate_ctx)
    return total


def any_active(weights: dict) -> bool:
    """True if at least one rerank signal has a nonzero weight."""
    return any(weights.get(name, 0.0) != 0.0 for name in SCORERS)
