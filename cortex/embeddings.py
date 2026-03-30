"""
Embedding module for CORTEX.

Provides lightweight text→vector encoding with zero external dependencies
(beyond numpy). Uses a hash-based projection with TF-IDF-like weighting.

Optionally uses sentence-transformers for higher-quality embeddings when
the package is available.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Optional

import numpy as np

# Dimension for the hash-based embeddings
DEFAULT_DIM = 256

# Try to import sentence-transformers for optional high-quality embeddings
_TRANSFORMER_MODEL = None


def _get_transformer():
    global _TRANSFORMER_MODEL
    if _TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _TRANSFORMER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _TRANSFORMER_MODEL = False  # sentinel: tried and failed
    return _TRANSFORMER_MODEL if _TRANSFORMER_MODEL is not False else None


# ── Tokenisation ─────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the is was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but if or because until while about".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ── Hash-based embedding ────────────────────────────────────

def _token_hash_vector(token: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic hash → unit vector for a single token."""
    h = hashlib.sha256(token.encode()).digest()
    # Use hash bytes as seed for a deterministic random vector
    rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_text(text: str, dim: int = DEFAULT_DIM, use_transformer: bool = False) -> np.ndarray:
    """
    Convert text to a dense vector.

    Parameters
    ----------
    text : str
        Input text.
    dim : int
        Vector dimension (only for hash-based mode).
    use_transformer : bool
        If True, try sentence-transformers first.

    Returns
    -------
    np.ndarray of shape (dim,) or (384,) for transformers.
    """
    if use_transformer:
        model = _get_transformer()
        if model is not None:
            vec = model.encode(text, convert_to_numpy=True)
            return vec.astype(np.float32)

    tokens = tokenize(text)
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    # TF-IDF-like weighting: TF from document, IDF approximated by inverse token frequency
    tf = Counter(tokens)
    total = len(tokens)

    vec = np.zeros(dim, dtype=np.float32)
    for token, count in tf.items():
        weight = (count / total) * (1.0 / math.log1p(count))  # TF * pseudo-IDF
        vec += weight * _token_hash_vector(token, dim)

    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of query against each row in matrix."""
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    normed = matrix / norms[:, np.newaxis]
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.zeros(len(matrix), dtype=np.float32)
    return (normed @ (query / q_norm)).astype(np.float32)
