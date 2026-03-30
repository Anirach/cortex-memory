#!/usr/bin/env python3
"""
Watch retrieval strategies evolve over generations.

Simulates feedback from an agent using memories, letting the genetic algorithm
discover optimal retrieval weights.
"""

import random
from cortex import CortexEngine


def main():
    engine = CortexEngine(db_path=":memory:")

    # ── Seed knowledge base ──────────────────────────────────
    print("=== Seeding Knowledge Base ===\n")

    facts = [
        ("Python is a high-level programming language", "programming"),
        ("Machine learning is a subset of artificial intelligence", "ai"),
        ("Bangkok is the capital of Thailand", "geography"),
        ("Docker containers provide OS-level virtualization", "devops"),
        ("The Transformer architecture revolutionized NLP", "ai"),
        ("SQLite is a self-contained database engine", "databases"),
        ("Git is a distributed version control system", "devops"),
        ("Neural networks are inspired by biological brains", "ai"),
        ("REST APIs use HTTP methods for CRUD operations", "programming"),
        ("Kubernetes orchestrates container deployments", "devops"),
    ]

    for content, category in facts:
        engine.store(content, memory_type="semantic", category=category, importance=0.7)

    # ── Simulate retrieval feedback ──────────────────────────
    print("=== Simulating Retrieval + Feedback ===\n")

    queries = [
        ("How do containers work?", ["docker", "kubernetes"]),
        ("Tell me about AI", ["machine learning", "transformer", "neural"]),
        ("What programming languages?", ["python"]),
        ("Database options?", ["sqlite"]),
    ]

    for epoch in range(5):
        for query, useful_keywords in queries:
            results = engine.recall(query, top_k=3)
            for r in results:
                is_useful = any(kw in r.content.lower() for kw in useful_keywords)
                engine.feedback(query, r.id, useful=is_useful)

    # ── Evolve strategies ────────────────────────────────────
    print("=== Evolution ===\n")

    print(f"Before evolution — Generation {engine.evolution.get_current_generation()}")
    best_before = engine.get_best_strategy()
    if best_before:
        print(f"  Best fitness: {best_before.fitness:.3f}")
        print(f"  Params: { {k: round(v, 3) for k, v in best_before.params.items()} }")

    history = engine.evolve(generations=10)

    print(f"\nAfter 10 generations:")
    for h in history:
        print(f"  Gen {h['generation']:2d}: best={h['best_fitness']:.3f}  avg={h['avg_fitness']:.3f}")

    best_after = engine.get_best_strategy()
    if best_after:
        print(f"\n  Current best strategy:")
        for k, v in sorted(best_after.params.items()):
            print(f"    {k}: {v:.4f}")

    # ── Full evolution history ───────────────────────────────
    print("\n=== Evolution History ===\n")
    full_history = engine.get_evolution_history()
    print(f"Total generations recorded: {len(full_history)}")

    if full_history:
        first = full_history[0]
        last = full_history[-1]
        print(f"  First gen best fitness: {first['best_fitness']:.3f}")
        print(f"  Last gen best fitness:  {last['best_fitness']:.3f}")

    print(f"\nFinal: {engine}")
    engine.close()


if __name__ == "__main__":
    main()
