"""
pipeline/nlp_embedder.py

Generates 384-dimensional semantic embeddings from Hindi/Urdu lyrics.

Model: paraphrase-multilingual-mpnet-base-v2 (~970 MB on first download)
  - Spec choice: handles Devanagari, Romanised Hindi, and Urdu script
  - Better multilingual semantic quality than MiniLM-L12-v2 for Hindi/Urdu
  - Both produce 384d vectors, but mpnet aligns semantically similar phrases
    across scripts more reliably (critical for ghazal vs filmi sad-song separation)

If you are constrained on disk space or download time:
  MiniLM-L12-v2 (~470 MB) is the fallback — same 384d output, lower quality.
  Change MODEL_NAME below and nothing else needs to change.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"   # spec default
NLP_DIM    = 768


# ── Lazy singleton ────────────────────────────────────────────────────────────
_nlp_model = None

def _get_model():
    global _nlp_model
    if _nlp_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading NLP model: %s (first load ~5–15s)…", MODEL_NAME)
        _nlp_model = SentenceTransformer(MODEL_NAME)
        logger.info("NLP model loaded.")
    return _nlp_model


# ── Public API ────────────────────────────────────────────────────────────────

def get_text_embedding(text: str | None) -> np.ndarray:
    """
    Encode lyrics into a 384-dimensional L2-normalised vector.

    Args:
        text: Lyrics string (Devanagari, Romanised Hindi, or Urdu script).
              Pass None or empty string when lyrics are unavailable.

    Returns:
        ndarray of shape (384,), dtype float32.
        Returns a zero vector when text is None or blank — caller must handle
        the lyrics_missing flag and redistribute fusion weights accordingly.
    """
    # Guard: None, empty, or whitespace-only → zero vector
    if not text or not text.strip():
        logger.debug("No lyrics text — returning zero NLP vector.")
        return np.zeros(NLP_DIM, dtype=np.float32)

    model     = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)

    # L2-normalise so cosine similarity = dot product (consistent with other groups)
    norm = np.linalg.norm(embedding)
    if norm > 1e-10:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def save_embedding(embedding: np.ndarray, path: str) -> None:
    """Save a NLP embedding vector to disk as a .npy file."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    np.save(path, embedding)
    logger.debug("NLP embedding saved → %s", path)