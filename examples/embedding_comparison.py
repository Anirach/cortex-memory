#!/usr/bin/env python3
"""
Embedding Backend Comparison Demo for CORTEX.

Compares hash-based vs sentence-transformers embeddings to demonstrate
that real embeddings capture semantic similarity while hash-based don't.

Usage:
    python examples/embedding_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from cortex.embeddings import (
    EmbeddingBackend,
    embed_text_with_backend,
    cosine_similarity,
    get_backend_info,
    detect_best_backend,
)


# ── Sample texts ─────────────────────────────────────────────
SAMPLE_TEXTS = [
    "The cat sat on the mat.",
    "A kitten rested on the rug.",
    "Dogs are loyal companions to humans.",
    "Python is a popular programming language.",
    "Machine learning models learn from data.",
    "Deep neural networks have many layers.",
    "The weather is sunny and warm today.",
    "It's a beautiful clear day outside.",
    "Quantum computing uses qubits for computation.",
    "The stock market crashed yesterday.",
    "Financial markets experienced a sharp decline.",
    "I love eating pizza for dinner.",
    "Artificial intelligence is transforming healthcare.",
    "Medical diagnosis is being improved by AI systems.",
]

# Pairs expected to be semantically similar
SIMILAR_PAIRS = [
    (0, 1),   # cat/mat ↔ kitten/rug
    (4, 5),   # ML models ↔ deep neural nets
    (6, 7),   # sunny warm ↔ beautiful clear day
    (9, 10),  # stock market crash ↔ financial decline
    (12, 13), # AI healthcare ↔ medical AI
]

# Pairs expected to be dissimilar
DISSIMILAR_PAIRS = [
    (0, 3),   # cat/mat ↔ Python programming
    (2, 8),   # dogs ↔ quantum computing
    (6, 9),   # weather ↔ stock market
    (11, 4),  # pizza ↔ machine learning
]


def compute_similarity_matrix(texts: list[str], backend: EmbeddingBackend) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    embeddings = [embed_text_with_backend(t, backend) for t in texts]
    n = len(embeddings)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    return matrix


def print_similarity_table(matrix: np.ndarray, texts: list[str], label: str):
    """Print a compact similarity table for selected pairs."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n  Similar Pairs (should be HIGH):")
    print(f"  {'Pair':<55} {'Sim':>6}")
    print(f"  {'-'*55} {'-'*6}")
    for i, j in SIMILAR_PAIRS:
        short_a = texts[i][:25]
        short_b = texts[j][:25]
        sim = matrix[i][j]
        marker = "✓" if sim > 0.5 else "✗"
        print(f"  {short_a} ↔ {short_b:<25} {sim:6.3f} {marker}")

    print(f"\n  Dissimilar Pairs (should be LOW):")
    print(f"  {'Pair':<55} {'Sim':>6}")
    print(f"  {'-'*55} {'-'*6}")
    for i, j in DISSIMILAR_PAIRS:
        short_a = texts[i][:25]
        short_b = texts[j][:25]
        sim = matrix[i][j]
        marker = "✓" if sim < 0.3 else "✗"
        print(f"  {short_a} ↔ {short_b:<25} {sim:6.3f} {marker}")


def main():
    print("CORTEX Embedding Comparison Demo")
    print("=" * 70)

    info = get_backend_info()
    print(f"Available backends: {info['available_backends']}")
    print(f"Best available:     {info['active_backend']}")
    print(f"Texts to compare:   {len(SAMPLE_TEXTS)}")

    # Always test hash-based
    print("\nComputing hash-based embeddings...")
    hash_matrix = compute_similarity_matrix(SAMPLE_TEXTS, EmbeddingBackend.HASH)
    print_similarity_table(hash_matrix, SAMPLE_TEXTS, "HASH-BASED (TF-IDF, 256-dim)")

    # Test sentence-transformers if available
    best = detect_best_backend()
    if best == EmbeddingBackend.SENTENCE_TRANSFORMERS:
        print("\nComputing sentence-transformers embeddings...")
        st_matrix = compute_similarity_matrix(SAMPLE_TEXTS, EmbeddingBackend.SENTENCE_TRANSFORMERS)
        print_similarity_table(st_matrix, SAMPLE_TEXTS, "SENTENCE-TRANSFORMERS (MiniLM, 384-dim)")
    elif best == EmbeddingBackend.ONNX:
        print("\nComputing ONNX Runtime embeddings...")
        onnx_matrix = compute_similarity_matrix(SAMPLE_TEXTS, EmbeddingBackend.ONNX)
        print_similarity_table(onnx_matrix, SAMPLE_TEXTS, "ONNX RUNTIME (MiniLM, 384-dim)")
    else:
        print("\n⚠ No semantic embedding backend available.")
        print("  Install: pip install sentence-transformers")
        print("  Or:      pip install onnxruntime tokenizers")

    # Summary comparison
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    hash_sim_avg = np.mean([hash_matrix[i][j] for i, j in SIMILAR_PAIRS])
    hash_dis_avg = np.mean([hash_matrix[i][j] for i, j in DISSIMILAR_PAIRS])
    print(f"\n  Hash-based:")
    print(f"    Avg similar-pair similarity:    {hash_sim_avg:.3f}")
    print(f"    Avg dissimilar-pair similarity:  {hash_dis_avg:.3f}")
    print(f"    Discrimination (delta):          {hash_sim_avg - hash_dis_avg:.3f}")

    if best != EmbeddingBackend.HASH:
        if best == EmbeddingBackend.SENTENCE_TRANSFORMERS:
            sem_matrix = compute_similarity_matrix(SAMPLE_TEXTS, EmbeddingBackend.SENTENCE_TRANSFORMERS)
            label = "Sentence-Transformers"
        else:
            sem_matrix = compute_similarity_matrix(SAMPLE_TEXTS, EmbeddingBackend.ONNX)
            label = "ONNX Runtime"

        sem_sim_avg = np.mean([sem_matrix[i][j] for i, j in SIMILAR_PAIRS])
        sem_dis_avg = np.mean([sem_matrix[i][j] for i, j in DISSIMILAR_PAIRS])
        print(f"\n  {label}:")
        print(f"    Avg similar-pair similarity:    {sem_sim_avg:.3f}")
        print(f"    Avg dissimilar-pair similarity:  {sem_dis_avg:.3f}")
        print(f"    Discrimination (delta):          {sem_sim_avg - sem_dis_avg:.3f}")
        print(f"\n  → Semantic embeddings show {(sem_sim_avg - sem_dis_avg) / max(0.001, hash_sim_avg - hash_dis_avg):.1f}x better discrimination")


if __name__ == "__main__":
    main()
