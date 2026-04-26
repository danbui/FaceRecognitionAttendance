"""
Face embedding matcher – cosine similarity search over the database.

For < 1000 employees, linear scan is fast enough (~1-2 ms).
If scaling beyond 5000, consider FAISS or Annoy for ANN search.
"""
import numpy as np
from typing import List, Dict, Any, Optional

from .config import RECOGNITION_COSINE_THRESHOLD


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized embedding vectors."""
    # SFace embeddings are already L2-normalized, so dot product = cosine sim
    return float(np.dot(a.flatten(), b.flatten()))


def match_embedding(
    query_embedding: np.ndarray,
    db_embeddings: List[Dict[str, Any]],
    threshold: float = RECOGNITION_COSINE_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """
    Find the best matching face embedding from the database.

    Args:
        query_embedding: numpy array shape (1, 128) from FaceEmbedder.
        db_embeddings: List of dicts with keys: id, employee_id, employee_code, full_name, embedding (np.ndarray).
        threshold: Minimum cosine similarity to consider a match (default: 0.363 for SFace).

    Returns:
        Best matching dict with added 'confidence' key, or None if no match above threshold.
    """
    if not db_embeddings:
        return None

    best = None
    best_score = -1.0
    query_flat = query_embedding.flatten()

    for row in db_embeddings:
        db_emb = row["embedding"]
        if isinstance(db_emb, np.ndarray):
            db_flat = db_emb.flatten()
        else:
            db_flat = np.array(db_emb, dtype=np.float32).flatten()

        score = float(np.dot(query_flat, db_flat))
        if score > best_score:
            best_score = score
            best = row

    if best is not None and best_score >= threshold:
        result = dict(best)
        result["confidence"] = best_score
        return result

    return None