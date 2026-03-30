#!/usr/bin/env python3
"""
How to integrate CORTEX with an LLM agent.

Shows: memory-augmented conversation, self-improvement, gap filling.
"""

from cortex import CortexEngine


def simulate_agent():
    engine = CortexEngine(db_path=":memory:")

    # ── Agent learns from conversations ──────────────────────
    print("=== Agent Learning from Conversations ===\n")

    conversations = [
        ("user", "What's the weather in Bangkok?"),
        ("agent", "It's currently 32°C and sunny in Bangkok."),
        ("user", "How do you check weather?"),
        ("agent", "I use the wttr.in API: curl wttr.in/Bangkok"),
        ("user", "Remember that I prefer Celsius, not Fahrenheit."),
        ("agent", "Noted! I'll always use Celsius for you."),
    ]

    for role, message in conversations:
        engine.remember(
            f"[{role}] {message}",
            importance=0.6 if role == "user" else 0.4,
            tags=["conversation"],
        )

    # Store user preferences as semantic memory
    engine.store(
        "User prefers Celsius temperature format",
        memory_type="semantic",
        importance=0.9,
        category="user_preferences",
    )

    # Store learned skill
    engine.learn_skill(
        "To check weather, use: curl wttr.in/{city}?format=3",
        trigger="weather request",
        pattern="api_call",
    )

    # ── Agent handles a new query using memory ───────────────
    print("--- New query: 'What temperature format do I like?' ---")
    results = engine.recall("temperature format preference")
    for r in results[:3]:
        print(f"  Memory: {r.content[:80]} (score: {r.score:.3f})")

    # ── Self-improvement: agent made a mistake ───────────────
    print("\n=== Self-Improvement ===\n")

    err_id = engine.log_error(
        "Gave temperature in Fahrenheit instead of Celsius",
        context="User asked about Bangkok weather",
        category="format_error",
        severity=0.7,
    )
    print(f"Logged error: {err_id}")

    cor_id = engine.log_correction(
        correct="Always use Celsius for this user",
        error_id=err_id,
        wrong="Used Fahrenheit by default",
        pattern="temperature_format",
    )
    print(f"Logged correction: {cor_id}")

    report = engine.get_improvement_report()
    print(f"Errors: {report.total_errors}, Resolved: {report.resolved_errors}")
    print(f"Trend: {report.improvement_trend}")

    # ── Gap filling ──────────────────────────────────────────
    print("\n=== Gap Analysis ===\n")

    gaps = engine.find_gaps(top_k=5)
    for gap in gaps:
        print(f"  Gap: {gap.description[:70]} (type={gap.gap_type.value}, priority={gap.priority:.2f})")

    fill_results = engine.fill_gaps(max_fills=3)
    for fr in fill_results:
        status = "✓" if fr.success else "✗"
        print(f"  {status} Fill ({fr.strategy}): {fr.content[:60] if fr.content else fr.reason}")

    # ── Consolidate ──────────────────────────────────────────
    print("\n=== Consolidation ===\n")
    report = engine.consolidate()
    print(f"Moved {report.working_to_episodic} working → episodic")
    print(f"Promoted {report.episodic_to_semantic} episodic → semantic")

    # ── Meta-cognition ───────────────────────────────────────
    print("\n=== Meta-Cognition ===\n")

    confidence = engine.assess_confidence("weather")
    print(f"Weather knowledge: confidence={confidence.confidence:.2f}, coverage={confidence.coverage:.2f}")

    evaluation = engine.self_evaluate()
    print(f"Overall quality: {evaluation.overall_quality:.2f}")
    for rec in evaluation.recommendations[:3]:
        print(f"  → {rec}")

    print(f"\nFinal: {engine}")
    engine.close()


if __name__ == "__main__":
    simulate_agent()
