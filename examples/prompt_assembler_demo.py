#!/usr/bin/env python3
"""
Prompt Assembler Demo — shows automatic technique selection based on task complexity.

Demonstrates how CORTEX's PromptAssembler analyzes queries and builds
optimized prompts using different prompt engineering techniques.
"""

from cortex.engine import CortexEngine
from cortex.prompt_assembler import TaskComplexity


def print_assembled(label: str, result):
    """Pretty-print an assembled prompt."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Complexity:  {result.complexity.value}")
    print(f"  Techniques:  {[t.value for t in result.techniques_applied]}")
    print(f"  Confidence:  {result.confidence}")
    print(f"  Role:        {result.metadata.get('role', 'default')}")
    print(f"  Context:     {len(result.context_memories)} memories")
    print(f"  Few-shot:    {len(result.few_shot_examples)} examples")
    print()
    print(f"  --- SYSTEM PROMPT ---")
    for line in result.system_prompt.split("\n"):
        print(f"  | {line}")
    print()
    print(f"  --- USER PROMPT ---")
    for line in result.user_prompt.split("\n")[:15]:
        print(f"  | {line}")
    if len(result.user_prompt.split("\n")) > 15:
        print(f"  | ... ({len(result.user_prompt.split(chr(10)))} lines total)")
    print()


def main():
    print("CORTEX PromptAssembler Demo")
    print("=" * 70)

    engine = CortexEngine(":memory:")

    # Seed some memories
    engine.store("Python is a high-level programming language created by Guido van Rossum", memory_type="semantic")
    engine.store("Machine learning uses statistical methods to learn from data", memory_type="semantic")
    engine.store("Neural networks are inspired by biological neurons", memory_type="semantic")
    engine.store("Docker containers package applications with dependencies", memory_type="semantic")

    engine.learn_skill(
        "When writing unit tests, use pytest with fixtures and parametrize",
        trigger="write tests",
        pattern="testing pattern",
    )
    engine.learn_skill(
        "When deploying, run tests first, then build Docker image, then push",
        trigger="deploy",
        pattern="deployment pipeline",
    )

    # ── Demo 1: Simple query → Zero-shot + RAG ──────────
    result = engine.assemble_prompt("What is Python?")
    print_assembled("1. SIMPLE QUERY → Zero-shot + RAG", result)

    # ── Demo 2: Complex analytical → CoT + RAG + Self-Critique ──
    result = engine.assemble_prompt(
        "Compare and contrast neural networks with traditional machine learning. "
        "How do they differ in approach and why would you choose one over the other? "
        "Furthermore, analyze the tradeoffs step by step"
    )
    print_assembled("2. COMPLEX QUERY → CoT + RAG + Self-Critique", result)

    # ── Demo 3: Knowledge gap → ReAct loop ──────────────
    result = engine.assemble_prompt("What is the Riemann hypothesis and its implications?")
    print_assembled("3. KNOWLEDGE GAP → ReAct + CoT", result)

    # ── Demo 4: Skill execution → Few-Shot from procedural ─
    result = engine.assemble_prompt(
        "Write tests for the payment module",
        task_hint=TaskComplexity.SKILL_EXECUTION,
    )
    print_assembled("4. SKILL EXECUTION → Few-Shot + Structured Output", result)

    # ── Demo 5: Custom role + structured output ─────────
    result = engine.assemble_prompt(
        "Evaluate the scalability of our microservice architecture",
        role="analyst",
        output_format="JSON with sections: summary, strengths, weaknesses, recommendation",
    )
    print_assembled("5. CUSTOM ROLE + STRUCTURED OUTPUT", result)

    # ── Record outcomes and show stats ──────────────────
    results = [
        engine.assemble_prompt("What is Python?"),
        engine.assemble_prompt("How does Docker work?"),
        engine.assemble_prompt("Compare X and Y", task_hint=TaskComplexity.COMPLEX),
    ]
    engine.prompt_assembler.record_outcome(results[0], success=True)
    engine.prompt_assembler.record_outcome(results[1], success=True)
    engine.prompt_assembler.record_outcome(results[2], success=False, feedback="Incomplete comparison")

    stats = engine.prompt_assembler.get_technique_stats()
    print(f"\n{'='*70}")
    print("  TECHNIQUE STATS")
    print(f"{'='*70}")
    print(f"  Total assemblies: {stats['total_assemblies']}")
    print(f"  By complexity:    {stats['by_complexity']}")
    print(f"  By technique:     {stats['by_technique']}")

    engine.close()
    print(f"\n{'='*70}")
    print("  Demo complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
