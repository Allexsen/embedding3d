"""Sentence embeddings via the same model family that built the corpus (mpnet)."""
import numpy as np

from common import CONFIG

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(CONFIG["embedder"]["id"])
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Normalized float32 embeddings, shape (n, dim)."""
    vecs = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
