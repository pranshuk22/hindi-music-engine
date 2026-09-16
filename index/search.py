"""
index/search.py
────────────────
Similarity search, Rocchio feedback, and playlist centroid queries.

Supports three query modes (AGENTS.md):
  1. Single song  — look up by song_id, return top-K similar songs
  2. Rocchio      — adjust the query vector with ✅/❌ feedback, re-query
  3. Playlist     — compute centroid of multiple song vectors, return top-K

All queries operate in the 256d PCA-compressed embedding space.
FAISS index must be built with index/build_index.py before searching.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import numpy as np
import faiss

sys.path.insert(0, ".")
from utils.db       import get_song_by_id, get_all_songs, init_db
from pipeline.embedder import (
    load_embedding,
    rocchio_adjust,
    playlist_centroid,
)
from index import rerank

log = logging.getLogger(__name__)

INDEX_PATH    = "index/faiss_index.bin"
SONG_MAP_PATH = "index/song_id_map.npy"

# ─── Index loading ────────────────────────────────────────────────────────────

_index    = None
_song_ids = None


def load_index():
    """Load and cache the FAISS index and song ID map."""
    global _index, _song_ids
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run: python index/build_index.py"
            )
        _index    = faiss.read_index(INDEX_PATH)
        _song_ids = np.load(SONG_MAP_PATH, allow_pickle=True).tolist()
        log.info(f"FAISS index loaded: {_index.ntotal} songs")
    return _index, _song_ids


def reload_index():
    """Force-reload the index (call after adding new songs)."""
    global _index, _song_ids
    _index = _song_ids = None
    return load_index()


# ─── Core search ─────────────────────────────────────────────────────────────

def _run_search(
    query_vec: np.ndarray,
    top_k:     int,
    exclude_ids: list[str] = None,
    category:       str  = None,
    tempo_bucket:   int  = None,
    exclude_artist: str  = None,
    anchor_ctx:     dict = None,
    rerank_weights: dict = None,
) -> list[dict]:
    """
    Run a FAISS inner-product search, apply the Stage 2 rerank (index/rerank.py),
    and return formatted result dicts. Filters are applied during retrieval.

    anchor_ctx: built via rerank.build_context(anchor_row) — everything the
    Stage 2 scorers need about the query song. None (e.g. playlist-centroid
    or manual-vector queries with no single anchor song) means no rerank
    signal can be computed, so effectively no reranking happens regardless
    of rerank_weights.
    rerank_weights: defaults to rerank.DEFAULT_WEIGHTS (currently only
    `composer` is active). Pass an override dict to ablate/tune individual
    signals — see scripts/evaluate_golden.py --rerank-weight.
    """
    import json
    index, song_ids = load_index()
    exclude_set = set(exclude_ids or [])

    if rerank_weights is None:
        rerank_weights = rerank.DEFAULT_WEIGHTS
    rerank_active = anchor_ctx is not None and rerank.any_active(rerank_weights)

    # Over-fetch heavily if filters OR reranking are active, since FAISS is
    # blind to metadata/rerank signals and a rerank can pull a lower-cosine
    # result above ones we'd otherwise have already cut off.
    has_filters = any([category, tempo_bucket is not None, exclude_artist, rerank_active])
    fetch_k = (top_k * 5) if has_filters else (top_k + len(exclude_set) + 5)
    fetch_k = min(fetch_k, len(song_ids)) # Don't fetch more than we have

    distances, indices = index.search(
        query_vec.reshape(1, -1).astype(np.float32),
        fetch_k,
    )

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(song_ids):
            continue

        sid = song_ids[idx]
        if sid in exclude_set:
            continue

        row = get_song_by_id(sid)
        if row is None:
            continue

        # --- METADATA FILTERING ---
        if category and row["category"] != category:
            continue

        if exclude_artist and row["artist"].lower() == exclude_artist.lower():
            continue

        if tempo_bucket is not None:
            feat_path = row["features_path"]
            if feat_path and os.path.exists(feat_path):
                with open(feat_path) as f:
                    feats = json.load(f)
                tb = feats.get("tempo_bucket", [0, 0, 0, 0])
                if not (isinstance(tb, list) and len(tb) == 4 and tb[tempo_bucket] == 1.0):
                    continue
            else:
                continue # Skip if we can't verify the tempo

        raw_score = float(dist)
        rerank_delta = 0.0
        if rerank_active:
            candidate_ctx = rerank.build_context(row)
            rerank_delta = rerank.rerank_score(anchor_ctx, candidate_ctx, rerank_weights)
        adjusted_score = raw_score + rerank_delta

        candidates.append({
            "song_id":  sid,
            "title":    row["title"],
            "artist":   row["artist"],
            "category": row["category"],
            "score":    round(adjusted_score, 4),
        })

        # Without reranking active, stop exactly when we hit the requested
        # amount — FAISS already returns results in score order, so this is
        # equivalent to plain cosine top-k. With reranking active, a later
        # (lower-cosine) candidate can still outrank an earlier one after
        # adjustment, so keep collecting up to fetch_k and sort below.
        if not rerank_active and len(candidates) >= top_k:
            break

    if rerank_active:
        candidates.sort(key=lambda r: r["score"], reverse=True)

    return candidates[:top_k]

# ─── Mode 1: Single-song similarity ──────────────────────────────────────────

def find_similar_by_id(song_id: str, top_k: int = 10, **kwargs) -> list[dict]:
    """
    Return top-K songs similar to the given song_id.
    The song must already be in the database and have a saved embedding.

    Auto-builds the Stage 2 rerank anchor context (index/rerank.py) from
    this song's own row, unless the caller already passed anchor_ctx
    explicitly (pass anchor_ctx=None to force reranking off). Harmless
    when a feature is unpopulated for a song — that scorer just returns 0.
    """
    row = get_song_by_id(song_id)
    if row is None:
        raise ValueError(f"Song '{song_id}' not found in database.")

    emb_path = row["embedding_path"]
    if not emb_path or not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding not found for {song_id}: {emb_path}")

    if "anchor_ctx" not in kwargs:
        kwargs["anchor_ctx"] = rerank.build_context(row)

    query_vec = load_embedding(emb_path)
    return _run_search(query_vec, top_k, exclude_ids=[song_id], **kwargs)

def find_similar_by_name(title: str, artist: str, top_k: int = 10, **kwargs) -> list[dict]:
    """
    Look up a song by title+artist and return similar songs.
    Constructs the song_id from title+artist (matching processor.py logic).
    """
    raw = f"{title}_{artist}".lower()
    song_id = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)[:50]
    return find_similar_by_id(song_id, top_k=top_k, **kwargs)


def get_results(self, top_k: int = 10, **kwargs) -> list[dict]:
    """
    Look up a song by title+artist and return similar songs.
    Constructs the song_id from title+artist (matching processor.py logic).
    """
    return _run_search(self._query_vec, top_k, exclude_ids=list(self._excluded), **kwargs)


# ─── Mode 2: Rocchio feedback ────────────────────────────────────────────────

def _l2norm(vec: np.ndarray) -> np.ndarray:
    """Safely L2-normalize a vector locally."""
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

class SearchSession:
    """
    Stateful search session for the Tick/Cross feedback loop.

    Usage:
        session = SearchSession(song_id="tum_hi_ho_arijit_singh")
        results = session.get_results(top_k=5)

        # User clicks ✅ on result[0], ❌ on result[2]
        session.approve(results[0]["song_id"])
        session.reject(results[2]["song_id"])
        refined = session.get_results(top_k=5)
    """

    def __init__(
        self,
        song_id:      str  = None,
        query_vec:    np.ndarray = None,
        alpha: float  = 1.0,
        beta:  float  = 0.75,
        gamma: float  = 0.25,
    ):
        """
        Initialise with either a song_id (looks up its embedding)
        or a pre-built query_vec (for playlist centroid queries).
        """
        if query_vec is not None:
            self._query_vec = _l2norm(query_vec.astype(np.float32))
        elif song_id is not None:
            row = get_song_by_id(song_id)
            if row is None:
                raise ValueError(f"Song '{song_id}' not in database.")
            self._query_vec = load_embedding(row["embedding_path"])
        else:
            raise ValueError("Provide either song_id or query_vec.")

        self._original_vec  = self._query_vec.copy()
        self._approved_vecs = []
        self._rejected_vecs = []
        self._alpha = alpha
        self._beta  = beta
        self._gamma = gamma
        self._excluded = {song_id} if song_id else set()

    def approve(self, song_id: str):
        """Mark a song as approved (✅) and adjust the query vector."""
        row = get_song_by_id(song_id)
        if row and row["embedding_path"] and os.path.exists(row["embedding_path"]):
            vec = load_embedding(row["embedding_path"])
            self._approved_vecs.append(vec)
            self._excluded.add(song_id)
            self._update_query()

    def reject(self, song_id: str):
        """Mark a song as rejected (❌) and adjust the query vector."""
        row = get_song_by_id(song_id)
        if row and row["embedding_path"] and os.path.exists(row["embedding_path"]):
            vec = load_embedding(row["embedding_path"])
            self._rejected_vecs.append(vec)
            self._excluded.add(song_id)
            self._update_query()

    def _update_query(self):
        self._query_vec = rocchio_adjust(
            self._original_vec,
            self._approved_vecs,
            self._rejected_vecs,
            alpha=self._alpha,
            beta=self._beta,
            gamma=self._gamma,
        )

    def get_results(self, top_k: int = 10) -> list[dict]:
        """Run a search with the current (possibly adjusted) query vector."""
        return _run_search(
            self._query_vec,
            top_k,
            exclude_ids=list(self._excluded),
        )

    def reset(self):
        """Reset the session back to the original query (no feedback applied)."""
        self._query_vec     = self._original_vec.copy()
        self._approved_vecs = []
        self._rejected_vecs = []


# ─── Mode 3: Playlist centroid ────────────────────────────────────────────────

def find_similar_to_playlist(song_ids: list[str], top_k: int = 10) -> list[dict]:
    """
    Given a list of song IDs, compute the playlist centroid vector
    and return the top-K songs closest to that centroid that are NOT
    already in the playlist.

    Used for the Playlist Expansion Engine feature.
    """
    vecs = []
    for sid in song_ids:
        row = get_song_by_id(sid)
        if row is None:
            log.warning(f"Playlist song '{sid}' not in database — skipped.")
            continue
        emb_path = row["embedding_path"]
        if emb_path and os.path.exists(emb_path):
            vecs.append(load_embedding(emb_path))
        else:
            log.warning(f"Missing embedding for playlist song '{sid}' — skipped.")

    if not vecs:
        raise ValueError("None of the playlist songs have embeddings.")

    centroid = playlist_centroid(vecs)
    return _run_search(centroid, top_k, exclude_ids=song_ids)


# ─── Metadata filtering (post-FAISS) ─────────────────────────────────────────

def filter_results(
    results:        list[dict],
    category:       str  = None,
    tempo_bucket:   int  = None,
    exclude_artist: str  = None,
) -> list[dict]:
    """
    Apply metadata filters to FAISS results.

    Post-FAISS filtering (not pre-filtering) per AGENTS.md directive:
    "Filtering in vector space is handled by metadata, not by modifying
    the query vector."

    Args:
        results:        Output of _run_search()
        category:       If set, keep only songs in this category
        tempo_bucket:   If set (0–3), keep only songs in this tempo bucket
        exclude_artist: If set, remove songs by this artist

    Returns:
        Filtered list (may be shorter than the original)
    """
    import json

    filtered = []
    for r in results:
        sid = r["song_id"]
        row = get_song_by_id(sid)
        if row is None:
            continue

        if category and row["category"] != category:
            continue

        if exclude_artist and row["artist"].lower() == exclude_artist.lower():
            continue

        if tempo_bucket is not None:
            feat_path = row["features_path"]
            if feat_path and os.path.exists(feat_path):
                with open(feat_path) as f:
                    feats = json.load(f)
                tb = feats.get("tempo_bucket", [0, 0, 0, 0])
                if not (isinstance(tb, list) and len(tb) == 4
                        and tb[tempo_bucket] == 1.0):
                    continue

        filtered.append(r)

    return filtered


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    songs = get_all_songs()
    if not songs:
        print("No songs in the database.")
        sys.exit(0)

    test_song = songs[0]
    print(f"\nSearching for songs similar to: {test_song['title']} — {test_song['artist']}")
    try:
        results = find_similar_by_id(test_song["id"], top_k=5)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}]  {r['title']} — {r['artist']}  ({r['category']})")
    except Exception as e:
        print(f"Search failed: {e}")
        print("Build the index first: python index/build_index.py")