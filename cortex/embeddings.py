"""
Embedding module for CORTEX.

Provides text→vector encoding with multiple backend support:
- **Hash-based** (default fallback): zero dependencies beyond NumPy, uses
  hash-projected TF-IDF weighting. Good for lexical matching.
- **Sentence-Transformers**: high-quality semantic embeddings via
  ``all-MiniLM-L6-v2`` (384-dim). Requires ``sentence-transformers``.
- **ONNX Runtime**: lightweight semantic embeddings via ONNX-exported
  ``all-MiniLM-L6-v2`` (384-dim) without PyTorch. Requires
  ``onnxruntime`` + ``tokenizers``.

Auto-detection tries ONNX → sentence-transformers → hash, so the best
available backend is used transparently.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# ── Constants ────────────────────────────────────────────────
DEFAULT_DIM = 256
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_TRANSFORMER_DIM = 384
_MAX_CACHE_SIZE = 2048


# ── Backend Enum ─────────────────────────────────────────────

class EmbeddingBackend(Enum):
    """Available embedding backends."""
    HASH = "hash"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    ONNX = "onnx"


# ── Configuration ────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding system."""
    backend: EmbeddingBackend = EmbeddingBackend.HASH
    model_name: str = _DEFAULT_MODEL_NAME
    dimension: int = DEFAULT_DIM
    cache_dir: Optional[str] = None
    cache_size: int = _MAX_CACHE_SIZE

    def __post_init__(self):
        if self.backend in (EmbeddingBackend.SENTENCE_TRANSFORMERS, EmbeddingBackend.ONNX):
            self.dimension = _TRANSFORMER_DIM


# ── LRU Embedding Cache ─────────────────────────────────────

class _EmbeddingCache:
    """Simple LRU cache for embeddings keyed by (text, backend, dim)."""

    def __init__(self, maxsize: int = _MAX_CACHE_SIZE):
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: tuple, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)


# ── Module-level state ───────────────────────────────────────
_embedding_cache = _EmbeddingCache()
_active_config: Optional[EmbeddingConfig] = None

# Lazy-loaded model singletons
_TRANSFORMER_MODEL = None
_ONNX_SESSION = None
_ONNX_TOKENIZER = None


# ── Backend detection & initialization ───────────────────────

def _check_onnx_available() -> bool:
    """Check if ONNX Runtime + tokenizers are importable."""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        return True
    except ImportError:
        return False


def _check_transformers_available() -> bool:
    """Check if sentence-transformers is importable."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except ImportError:
        return False


def detect_best_backend() -> EmbeddingBackend:
    """Auto-detect the best available backend: ONNX → transformers → hash."""
    if _check_onnx_available():
        return EmbeddingBackend.ONNX
    if _check_transformers_available():
        return EmbeddingBackend.SENTENCE_TRANSFORMERS
    return EmbeddingBackend.HASH


def _get_transformer():
    """Lazy-load sentence-transformers model."""
    global _TRANSFORMER_MODEL
    if _TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _TRANSFORMER_MODEL = SentenceTransformer(_DEFAULT_MODEL_NAME)
        except ImportError:
            _TRANSFORMER_MODEL = False  # sentinel: tried and failed
    return _TRANSFORMER_MODEL if _TRANSFORMER_MODEL is not False else None


def _get_onnx_session():
    """Lazy-load ONNX Runtime session and tokenizer for MiniLM."""
    global _ONNX_SESSION, _ONNX_TOKENIZER
    if _ONNX_SESSION is None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            import os

            # Try to find or download the ONNX model
            cache_dir = os.path.expanduser("~/.cache/cortex-embeddings")
            model_dir = os.path.join(cache_dir, "all-MiniLM-L6-v2-onnx")
            model_path = os.path.join(model_dir, "model.onnx")
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")

            if not os.path.exists(model_path):
                # Try downloading from huggingface hub
                try:
                    from huggingface_hub import hf_hub_download
                    os.makedirs(model_dir, exist_ok=True)
                    hf_hub_download(
                        repo_id="sentence-transformers/all-MiniLM-L6-v2",
                        filename="onnx/model.onnx",
                        local_dir=model_dir,
                        local_dir_use_symlinks=False,
                    )
                    hf_hub_download(
                        repo_id="sentence-transformers/all-MiniLM-L6-v2",
                        filename="tokenizer.json",
                        local_dir=model_dir,
                        local_dir_use_symlinks=False,
                    )
                    # The download might put files in subdirs
                    onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
                    if os.path.exists(onnx_path) and not os.path.exists(model_path):
                        import shutil
                        shutil.move(onnx_path, model_path)
                except Exception:
                    _ONNX_SESSION = False
                    _ONNX_TOKENIZER = False
                    return None, None

            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                _ONNX_SESSION = False
                _ONNX_TOKENIZER = False
                return None, None

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            _ONNX_SESSION = ort.InferenceSession(model_path, sess_options)
            _ONNX_TOKENIZER = Tokenizer.from_file(tokenizer_path)
            _ONNX_TOKENIZER.enable_truncation(max_length=128)
            _ONNX_TOKENIZER.enable_padding(length=128)

        except (ImportError, Exception):
            _ONNX_SESSION = False
            _ONNX_TOKENIZER = False

    sess = _ONNX_SESSION if _ONNX_SESSION is not False else None
    tok = _ONNX_TOKENIZER if _ONNX_TOKENIZER is not False else None
    return sess, tok


def _onnx_encode(text: str) -> Optional[np.ndarray]:
    """Encode text using ONNX Runtime MiniLM."""
    session, tokenizer = _get_onnx_session()
    if session is None or tokenizer is None:
        return None

    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    # Mean pooling over token embeddings (output[0] is last_hidden_state)
    token_embeddings = outputs[0]  # shape: (1, seq_len, 384)
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.sum(mask_expanded, axis=1)
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
    pooled = sum_embeddings / sum_mask

    # L2 normalize
    vec = pooled[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


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
    rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _hash_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Hash-based TF-IDF embedding (zero external dependencies)."""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(dim, dtype=np.float32)

    tf = Counter(tokens)
    total = len(tokens)

    vec = np.zeros(dim, dtype=np.float32)
    for token, count in tf.items():
        weight = (count / total) * (1.0 / math.log1p(count))
        vec += weight * _token_hash_vector(token, dim)

    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


# ── Main API ─────────────────────────────────────────────────

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
        If True, try sentence-transformers first (legacy flag).
        For more control, use ``embed_text_with_backend()``.

    Returns
    -------
    np.ndarray of shape (dim,) for hash, or (384,) for transformers/ONNX.
    """
    if use_transformer:
        # Legacy path: try transformer first
        model = _get_transformer()
        if model is not None:
            cache_key = (text, "sentence_transformers", _TRANSFORMER_DIM)
            cached = _embedding_cache.get(cache_key)
            if cached is not None:
                return cached
            vec = model.encode(text, convert_to_numpy=True).astype(np.float32)
            _embedding_cache.put(cache_key, vec)
            return vec

    # Hash-based fallback
    cache_key = (text, "hash", dim)
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    vec = _hash_embed(text, dim)
    _embedding_cache.put(cache_key, vec)
    return vec


def embed_text_with_backend(
    text: str,
    backend: EmbeddingBackend = EmbeddingBackend.HASH,
    dim: int = DEFAULT_DIM,
) -> np.ndarray:
    """
    Embed text using a specific backend.

    Parameters
    ----------
    text : str
        Input text.
    backend : EmbeddingBackend
        Which backend to use.
    dim : int
        Dimension for hash-based backend only.

    Returns
    -------
    np.ndarray
    """
    effective_dim = _TRANSFORMER_DIM if backend != EmbeddingBackend.HASH else dim
    cache_key = (text, backend.value, effective_dim)
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    vec: np.ndarray

    if backend == EmbeddingBackend.ONNX:
        result = _onnx_encode(text)
        if result is not None:
            vec = result
        else:
            # Fallback to hash if ONNX fails
            vec = _hash_embed(text, dim)
    elif backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
        model = _get_transformer()
        if model is not None:
            vec = model.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            vec = _hash_embed(text, dim)
    else:
        vec = _hash_embed(text, dim)

    _embedding_cache.put(cache_key, vec)
    return vec


def embed_batch(
    texts: list[str],
    backend: EmbeddingBackend = EmbeddingBackend.HASH,
    dim: int = DEFAULT_DIM,
) -> np.ndarray:
    """
    Embed a batch of texts.

    Parameters
    ----------
    texts : list[str]
        Input texts.
    backend : EmbeddingBackend
        Which backend to use.
    dim : int
        Dimension for hash-based backend.

    Returns
    -------
    np.ndarray of shape (len(texts), dim) or (len(texts), 384).
    """
    if not texts:
        out_dim = _TRANSFORMER_DIM if backend != EmbeddingBackend.HASH else dim
        return np.zeros((0, out_dim), dtype=np.float32)

    # For sentence-transformers, batch encode is more efficient
    if backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
        model = _get_transformer()
        if model is not None:
            # Check cache first
            results = []
            uncached_indices = []
            uncached_texts = []
            for i, t in enumerate(texts):
                cached = _embedding_cache.get((t, "sentence_transformers", _TRANSFORMER_DIM))
                if cached is not None:
                    results.append((i, cached))
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(t)

            if uncached_texts:
                vecs = model.encode(uncached_texts, convert_to_numpy=True).astype(np.float32)
                for j, idx in enumerate(uncached_indices):
                    _embedding_cache.put((uncached_texts[j], "sentence_transformers", _TRANSFORMER_DIM), vecs[j])
                    results.append((idx, vecs[j]))

            results.sort(key=lambda x: x[0])
            return np.stack([r[1] for r in results])

    # For hash and ONNX, just loop
    return np.stack([embed_text_with_backend(t, backend, dim) for t in texts])


# ── Similarity functions ─────────────────────────────────────

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


# ── Backend info ─────────────────────────────────────────────

def get_backend_info() -> dict:
    """
    Report which embedding backends are available and which is active.

    Returns
    -------
    dict with keys:
        - active_backend: str
        - available_backends: list[str]
        - model_name: str
        - dimension: int
        - cache_size: int
        - cache_hits: int
        - cache_misses: int
    """
    available = [EmbeddingBackend.HASH.value]  # always available
    if _check_transformers_available():
        available.append(EmbeddingBackend.SENTENCE_TRANSFORMERS.value)
    if _check_onnx_available():
        available.append(EmbeddingBackend.ONNX.value)

    best = detect_best_backend()

    if best == EmbeddingBackend.HASH:
        dim = DEFAULT_DIM
    else:
        dim = _TRANSFORMER_DIM

    return {
        "active_backend": best.value,
        "available_backends": available,
        "model_name": _DEFAULT_MODEL_NAME if best != EmbeddingBackend.HASH else "hash-tfidf",
        "dimension": dim,
        "cache_size": len(_embedding_cache),
        "cache_hits": _embedding_cache.hits,
        "cache_misses": _embedding_cache.misses,
    }


def configure(config: EmbeddingConfig) -> EmbeddingConfig:
    """
    Apply an embedding configuration globally.

    Parameters
    ----------
    config : EmbeddingConfig
        Desired configuration. If backend is not available, falls back.

    Returns
    -------
    EmbeddingConfig — the actual applied configuration (may differ if fallback).
    """
    global _active_config

    actual_backend = config.backend
    if actual_backend == EmbeddingBackend.ONNX and not _check_onnx_available():
        actual_backend = EmbeddingBackend.SENTENCE_TRANSFORMERS
    if actual_backend == EmbeddingBackend.SENTENCE_TRANSFORMERS and not _check_transformers_available():
        actual_backend = EmbeddingBackend.HASH

    _active_config = EmbeddingConfig(
        backend=actual_backend,
        model_name=config.model_name,
        dimension=_TRANSFORMER_DIM if actual_backend != EmbeddingBackend.HASH else config.dimension,
        cache_dir=config.cache_dir,
        cache_size=config.cache_size,
    )

    # Resize cache if needed
    _embedding_cache._maxsize = _active_config.cache_size
    return _active_config


def get_cache_stats() -> dict:
    """Return embedding cache statistics."""
    return {
        "size": len(_embedding_cache),
        "hits": _embedding_cache.hits,
        "misses": _embedding_cache.misses,
        "hit_rate": (
            _embedding_cache.hits / (_embedding_cache.hits + _embedding_cache.misses)
            if (_embedding_cache.hits + _embedding_cache.misses) > 0
            else 0.0
        ),
    }


def clear_cache() -> None:
    """Clear the embedding cache."""
    _embedding_cache.clear()
