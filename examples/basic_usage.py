#!/usr/bin/env python3
"""
Basic CORTEX usage — store, recall, consolidate.
"""

from cortex import CortexEngine


def main():
    # Create an engine (use a file path for persistence, or :memory: for testing)
    engine = CortexEngine(db_path=":memory:")

    # ── Store memories across different types ────────────────
    print("=== Storing Memories ===")

    engine.store("The capital of France is Paris", memory_type="semantic", category="geography")
    engine.store("France is in Western Europe", memory_type="semantic", category="geography")
    engine.store("The Eiffel Tower is 330 meters tall", memory_type="semantic", category="landmarks")

    engine.remember("Had a great meeting with the team about AI architecture")
    engine.remember("Reviewed pull request #42 — needs refactoring")

    engine.learn_skill(
        "When user asks about weather, call wttr.in API with curl",
        trigger="weather request",
        pattern="api_call",
    )

    engine.store("Quick thought: should we use Redis for caching?", memory_type="working")

    print(f"Stats: {engine.stats()}")

    # ── Recall memories ──────────────────────────────────────
    print("\n=== Recalling Memories ===")

    results = engine.recall("What do I know about France?")
    for r in results:
        print(f"  [{r.memory_type}] {r.content[:80]}  (score: {r.score:.3f})")

    print()
    results = engine.recall("weather API")
    for r in results:
        print(f"  [{r.memory_type}] {r.content[:80]}  (score: {r.score:.3f})")

    # ── Consolidation (sleep/dream cycle) ────────────────────
    print("\n=== Consolidating ===")
    report = engine.consolidate()
    print(f"  Working → Episodic: {report.working_to_episodic}")
    print(f"  Episodic → Semantic: {report.episodic_to_semantic}")
    print(f"  Episodic → Procedural: {report.episodic_to_procedural}")
    print(f"  Pruned: {report.pruned_episodic}")
    print(f"  Duration: {report.duration_ms:.1f}ms")

    # ── Link memories ────────────────────────────────────────
    print("\n=== Linking Memories ===")
    facts = engine.semantic.recall("France", top_k=2)
    if len(facts) >= 2:
        engine.link(facts[0].id, facts[1].id, "related_to")
        related = engine.get_related(facts[0].id)
        print(f"  {facts[0].content[:40]} → {len(related)} related memories")

    # ── Stats ────────────────────────────────────────────────
    print(f"\nFinal stats: {engine.stats()}")
    engine.close()


if __name__ == "__main__":
    main()
