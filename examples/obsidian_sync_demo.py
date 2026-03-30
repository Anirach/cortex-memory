#!/usr/bin/env python3
"""
Obsidian Vault Integration Demo — bidirectional sync between CORTEX and a vault.
"""

import tempfile
from pathlib import Path
from cortex import CortexEngine


def main():
    # Create a temporary vault for demo
    vault_dir = tempfile.mkdtemp(prefix="cortex_vault_")
    vault = Path(vault_dir)
    print(f"Demo vault: {vault_dir}\n")

    # ── Create sample vault notes ────────────────────────────
    print("=== Creating Sample Vault ===\n")

    (vault / "Knowledge").mkdir()
    (vault / "memory").mkdir()
    (vault / "Projects").mkdir()
    (vault / "Guides").mkdir()

    (vault / "Knowledge" / "Python.md").write_text("""---
tags: [programming, python]
---

# Python

Python is a versatile programming language created by Guido van Rossum.
Used extensively in AI, data science, and web development.

See also: [[Machine Learning]], [[Django]], [[FastAPI]]
""")

    (vault / "Knowledge" / "Machine Learning.md").write_text("""---
tags: [ai, ml]
---

# Machine Learning

Machine learning is a subset of AI that learns from data.
Key frameworks: [[PyTorch]], [[TensorFlow]], [[scikit-learn]]

Related: [[Python]], [[Deep Learning]]
""")

    (vault / "memory" / "2026-03-30.md").write_text("""# 2026-03-30

## Morning
Built the CORTEX memory engine. Implemented 7 cognitive layers.

## Afternoon
Added Obsidian vault integration for bidirectional sync.
""")

    (vault / "Guides" / "How to Deploy Docker.md").write_text("""# How to Deploy Docker

Steps:
1. Write a Dockerfile
2. Build: `docker build -t myapp .`
3. Run: `docker run -p 8080:8080 myapp`
4. Deploy to registry
""")

    (vault / "Projects" / "CORTEX.md").write_text("""# Project CORTEX

A cognitive memory architecture for AI agents.

## Links
- [[Python]] — implementation language
- [[Machine Learning]] — related field
- [[NonExistentNote]] — dead link (will be detected as gap)
""")

    # ── Initialize CORTEX with Obsidian ──────────────────────
    engine = CortexEngine(
        db_path=":memory:",
        obsidian_vault=vault_dir,
        obsidian_sync=True,
        obsidian_para=True,
        obsidian_export_inferred=True,
    )

    # ── Ingest vault ─────────────────────────────────────────
    print("=== Ingesting Vault → CORTEX ===\n")
    stats = engine.obsidian.ingest(vault_dir)
    print(f"  Imported: {stats['imported']}")
    print(f"  Links:    {stats['links']}")
    print(f"  Errors:   {stats['errors']}")

    # ── Recall from ingested memories ────────────────────────
    print("\n=== Recalling Vault Knowledge ===\n")
    for query in ["Python programming", "deploy docker", "CORTEX project"]:
        results = engine.recall(query, top_k=2)
        print(f'  "{query}":')
        for r in results[:2]:
            print(f"    [{r.memory_type}] {r.content[:60]}... (score: {r.score:.3f})")

    # ── Add CORTEX-generated knowledge ───────────────────────
    print("\n=== Adding CORTEX Knowledge ===\n")
    engine.store("FastAPI is a modern Python web framework", memory_type="semantic",
                 category="programming", source="user")
    engine.store("Always run tests before deploying", memory_type="procedural",
                 trigger="deployment", source="user")

    # ── Export CORTEX → Vault ────────────────────────────────
    print("=== Exporting CORTEX → Vault ===\n")
    export_stats = engine.obsidian.export(vault_dir)
    print(f"  Notes created: {export_stats['notes_created']}")
    print(f"  MOCs generated: {export_stats['mocs_generated']}")

    # Show generated files
    print("\n=== Generated Files ===\n")
    cortex_dir = vault / "CORTEX"
    if cortex_dir.exists():
        for f in sorted(cortex_dir.rglob("*.md")):
            print(f"  {f.relative_to(vault)}")

    # Check status file
    status = vault / "CORTEX-STATUS.md"
    if status.exists():
        print(f"\n=== CORTEX-STATUS.md ===\n")
        print(status.read_text()[:500])

    # ── Gap detection (dead links) ───────────────────────────
    print("\n=== Gap Report ===\n")
    report = engine.gap_report()
    print(f"  Total gaps: {report.total_gaps}")
    print(f"  By type: {report.gaps_by_type}")
    for gap in report.top_priority_gaps[:5]:
        print(f"  - {gap.description[:70]} (priority: {gap.priority:.2f})")

    print(f"\nFinal: {engine}")
    engine.close()

    # Cleanup
    import shutil
    shutil.rmtree(vault_dir, ignore_errors=True)
    print("\nDemo vault cleaned up.")


if __name__ == "__main__":
    main()
