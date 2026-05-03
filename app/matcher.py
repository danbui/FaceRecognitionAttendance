"""
Face embedding matcher – vectorized cosine similarity search.

Optimization for Raspberry Pi 4 (4GB RAM):
  - EmbeddingCache: loads embeddings from DB once, keeps in RAM as NumPy matrix.
  - Vectorized matching: single np.dot() call instead of Python loop.
  - SFace outputs L2-normalized vectors, so cosine_sim = dot product directly.

For < 1000 employees, this runs in < 0.5 ms on Pi 4.
If scaling beyond 5000, consider FAISS or Annoy for ANN search.
"""
import numpy as np
import time
from typing import List, Dict, Any, Optional

from .config import RECOGNITION_COSINE_THRESHOLD
from .database import load_embeddings


class EmbeddingCache:
    """
    In-memory cache for face embeddings.

    Loads all embeddings from SQLite once and stores them as a pre-built
    NumPy matrix (N, 128) for vectorized matching.  Call invalidate()
    after enrolling a new employee to force a reload on next match.
    """

    def __init__(self):
        self._rows: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None   # shape (N, 128)
        self._dirty: bool = True                      # needs reload
        self._last_load: float = 0.0

    # ── public API ────────────────────────────────────────

    def invalidate(self):
        """Mark cache as stale – will reload from DB on next get()."""
        self._dirty = True

    def get(self):
        """Return (rows, matrix).  Reloads from DB only if dirty."""
        if self._dirty:
            self._reload()
        return self._rows, self._matrix

    # ── internal ──────────────────────────────────────────

    def _reload(self):
        self._rows = load_embeddings()
        if self._rows:
            # Stack all (1, 128) arrays into (N, 128) matrix
            self._matrix = np.vstack(
                [r["embedding"].reshape(1, -1) for r in self._rows]
            ).astype(np.float32)
            # Pre-normalize rows (SFace should already be L2-normed, but guard)
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            self._matrix = self._matrix / norms
        else:
            self._matrix = None
        self._dirty = False
        self._last_load = time.time()


# ── Module-level singleton ────────────────────────────────
embedding_cache = EmbeddingCache()


def match_embedding(
    query_embedding: np.ndarray,
    threshold: float = RECOGNITION_COSINE_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """
    Find the best matching face embedding using vectorized cosine similarity.

    Args:
        query_embedding: numpy array shape (1, 128) from FaceEmbedder.
        threshold: Minimum cosine similarity to consider a match.

    Returns:
        Best matching dict with added 'confidence' key, or None.
    """
    rows, matrix = embedding_cache.get()

    if matrix is None or len(rows) == 0:
        return None

    # Flatten & normalize query
    query = query_embedding.flatten().astype(np.float32)
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-8:
        return None
    query = query / q_norm

    # ── Vectorized cosine similarity in one shot ──
    # matrix is (N, 128), query is (128,) → scores is (N,)
    scores = matrix @ query

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        result = dict(rows[best_idx])
        result["confidence"] = best_score
        return result

    return None