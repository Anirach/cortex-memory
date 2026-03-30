"""Tests for the CORTEX multi-backend embedding module."""

import numpy as np
import pytest

from cortex.embeddings import (
    EmbeddingBackend,
    EmbeddingConfig,
    _EmbeddingCache,
    _hash_embed,
    batch_cosine_similarity,
    clear_cache,
    configure,
    cosine_similarity,
    detect_best_backend,
    embed_batch,
    embed_text,
    embed_text_with_backend,
    get_backend_info,
    get_cache_stats,
    tokenize,
)


# ── Tokenization ─────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        tokens = tokenize("The cat sat on the mat")
        assert "cat" in tokens
        assert "sat" in tokens
        assert "mat" in tokens
        # stop words removed
        assert "the" not in tokens
        assert "on" not in tokens

    def test_empty(self):
        assert tokenize("") == []

    def test_single_char_removed(self):
        tokens = tokenize("I a b cat")
        assert "cat" in tokens
        assert "a" not in tokens
        assert "b" not in tokens


# ── Hash Backend ─────────────────────────────────────────────

class TestHashBackend:
    def test_embed_returns_correct_dim(self):
        vec = embed_text_with_backend("hello world", EmbeddingBackend.HASH, dim=256)
        assert vec.shape == (256,)
        assert vec.dtype == np.float32

    def test_embed_custom_dim(self):
        vec = embed_text_with_backend("hello world", EmbeddingBackend.HASH, dim=128)
        assert vec.shape == (128,)

    def test_embed_deterministic(self):
        v1 = embed_text_with_backend("test sentence", EmbeddingBackend.HASH)
        clear_cache()
        v2 = embed_text_with_backend("test sentence", EmbeddingBackend.HASH)
        np.testing.assert_array_equal(v1, v2)

    def test_embed_empty_text(self):
        vec = embed_text_with_backend("", EmbeddingBackend.HASH)
        assert np.allclose(vec, 0.0)

    def test_embed_normalized(self):
        vec = embed_text_with_backend("machine learning is great", EmbeddingBackend.HASH)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_different_texts_different_vectors(self):
        v1 = embed_text_with_backend("cats and dogs", EmbeddingBackend.HASH)
        v2 = embed_text_with_backend("quantum physics theory", EmbeddingBackend.HASH)
        sim = cosine_similarity(v1, v2)
        assert sim < 0.99  # should not be identical

    def test_legacy_embed_text(self):
        """Test backward-compatible embed_text function."""
        vec = embed_text("hello world", dim=256)
        assert vec.shape == (256,)

    def test_legacy_embed_text_transformer_flag(self):
        """Test that use_transformer=True falls back to hash when unavailable."""
        vec = embed_text("hello world", use_transformer=True)
        assert vec.shape[0] > 0  # should work regardless


# ── Auto-detection ───────────────────────────────────────────

class TestAutoDetection:
    def test_detect_returns_valid_backend(self):
        backend = detect_best_backend()
        assert isinstance(backend, EmbeddingBackend)

    def test_hash_always_available(self):
        info = get_backend_info()
        assert "hash" in info["available_backends"]

    def test_backend_info_structure(self):
        info = get_backend_info()
        assert "active_backend" in info
        assert "available_backends" in info
        assert "model_name" in info
        assert "dimension" in info
        assert "cache_size" in info
        assert "cache_hits" in info
        assert "cache_misses" in info

    def test_configure_hash(self):
        cfg = configure(EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=128))
        assert cfg.backend == EmbeddingBackend.HASH
        assert cfg.dimension == 128

    def test_configure_unavailable_falls_back(self):
        """If ONNX not installed, should fall back gracefully."""
        cfg = configure(EmbeddingConfig(backend=EmbeddingBackend.ONNX))
        # Should be ONNX if available, else fall back
        assert isinstance(cfg.backend, EmbeddingBackend)


# ── Embedding Cache ──────────────────────────────────────────

class TestEmbeddingCache:
    def test_cache_basic(self):
        cache = _EmbeddingCache(maxsize=10)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put(("test", "hash"), vec)
        result = cache.get(("test", "hash"))
        assert result is not None
        np.testing.assert_array_equal(result, vec)

    def test_cache_miss(self):
        cache = _EmbeddingCache(maxsize=10)
        result = cache.get(("nonexistent", "hash"))
        assert result is None
        assert cache.misses == 1

    def test_cache_hit_count(self):
        cache = _EmbeddingCache(maxsize=10)
        vec = np.array([1.0], dtype=np.float32)
        cache.put(("a", "hash"), vec)
        cache.get(("a", "hash"))
        cache.get(("a", "hash"))
        assert cache.hits == 2

    def test_cache_eviction(self):
        cache = _EmbeddingCache(maxsize=3)
        for i in range(5):
            cache.put((str(i), "hash"), np.array([float(i)], dtype=np.float32))
        assert len(cache) == 3
        # Oldest entries (0, 1) should be evicted
        assert cache.get(("0", "hash")) is None
        assert cache.get(("1", "hash")) is None
        assert cache.get(("4", "hash")) is not None

    def test_cache_lru_order(self):
        cache = _EmbeddingCache(maxsize=3)
        for i in range(3):
            cache.put((str(i), "hash"), np.array([float(i)], dtype=np.float32))
        # Access item 0 to make it most recently used
        cache.get(("0", "hash"))
        # Add new item, should evict item 1 (least recently used)
        cache.put(("3", "hash"), np.array([3.0], dtype=np.float32))
        assert cache.get(("0", "hash")) is not None  # still there
        assert cache.get(("1", "hash")) is None  # evicted

    def test_cache_clear(self):
        cache = _EmbeddingCache(maxsize=10)
        cache.put(("a", "hash"), np.array([1.0], dtype=np.float32))
        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_module_cache_stats(self):
        clear_cache()
        embed_text_with_backend("hello", EmbeddingBackend.HASH)
        embed_text_with_backend("hello", EmbeddingBackend.HASH)  # cache hit
        stats = get_cache_stats()
        assert stats["hits"] >= 1
        assert stats["size"] >= 1
        assert stats["hit_rate"] > 0


# ── Cosine Similarity ───────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(v1, v2)) < 1e-5

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(v1, v2) + 1.0) < 1e-5

    def test_zero_vector(self):
        v1 = np.zeros(3, dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == 0.0

    def test_batch_similarity(self):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        scores = batch_cosine_similarity(query, matrix)
        assert scores[0] > scores[1]  # first is most similar
        assert len(scores) == 3

    def test_batch_zero_query(self):
        query = np.zeros(3, dtype=np.float32)
        matrix = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores = batch_cosine_similarity(query, matrix)
        assert scores[0] == 0.0


# ── Batch Operations ─────────────────────────────────────────

class TestBatchOperations:
    def test_batch_embed_hash(self):
        texts = ["hello world", "foo bar", "test sentence"]
        result = embed_batch(texts, EmbeddingBackend.HASH)
        assert result.shape == (3, 256)

    def test_batch_embed_empty(self):
        result = embed_batch([], EmbeddingBackend.HASH)
        assert result.shape == (0, 256)

    def test_batch_consistent_with_single(self):
        texts = ["alpha beta", "gamma delta"]
        batch_result = embed_batch(texts, EmbeddingBackend.HASH)
        single_results = [embed_text_with_backend(t, EmbeddingBackend.HASH) for t in texts]
        for i in range(len(texts)):
            np.testing.assert_array_almost_equal(batch_result[i], single_results[i])


# ── EmbeddingConfig ──────────────────────────────────────────

class TestEmbeddingConfig:
    def test_default_config(self):
        cfg = EmbeddingConfig()
        assert cfg.backend == EmbeddingBackend.HASH
        assert cfg.dimension == 256
        assert cfg.model_name == "all-MiniLM-L6-v2"

    def test_transformer_config_sets_dim(self):
        cfg = EmbeddingConfig(backend=EmbeddingBackend.SENTENCE_TRANSFORMERS)
        assert cfg.dimension == 384

    def test_onnx_config_sets_dim(self):
        cfg = EmbeddingConfig(backend=EmbeddingBackend.ONNX)
        assert cfg.dimension == 384

    def test_hash_config_custom_dim(self):
        cfg = EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=512)
        assert cfg.dimension == 512
