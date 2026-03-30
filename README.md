# 🧠 CORTEX Memory Engine

**Cognitive Orchestrated Retrieval with Temporal EXperience**

A self-improving cognitive memory architecture for AI agents. CORTEX provides persistent, multi-layered memory with automatic consolidation, forgetting curves, self-evolution, gap detection, and Obsidian vault integration.

```
pip install cortex-memory
```

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   CortexEngine                      │
                        │                  (Orchestrator)                     │
                        └──────────┬──────────┬──────────┬───────────────────┘
                                   │          │          │
          ┌────────────────────────┼──────────┼──────────┼────────────────────┐
          │                        │          │          │                    │
  ┌───────▼───────┐  ┌────────────▼──┐  ┌────▼────┐  ┌─▼──────────────┐    │
  │   Working     │  │   Episodic    │  │Semantic │  │  Procedural    │    │
  │   Memory      │  │   Memory     │  │ Memory  │  │   Memory       │    │
  │  (Ring Buf)   │  │  (Events)    │  │ (Facts) │  │  (Skills)      │    │
  └───────┬───────┘  └──────┬───────┘  └────┬────┘  └──────┬─────────┘    │
          │                  │               │              │              │
          │    ┌─────────────▼───────────────▼──────────────▼─┐            │
          │    │         Hippocampal Index                     │            │
          │    │  (Vector + BM25 + Temporal + Graph Search)    │            │
          │    └──────────────────────────────────────────────-┘            │
          │                                                                │
  ┌───────▼──────────────────────────────────────────────────────────┐     │
  │                    Consolidation Engine                          │     │
  │        Working → Episodic → Semantic / Procedural               │     │
  │          + Ebbinghaus Forgetting Curve Pruning                  │     │
  └─────────────────────────────────────────────────────────────────┘     │
                                                                          │
  ┌────────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
  │  Self-Improvement  │  │  Self-Evolution   │  │  Meta-Cognition  │     │
  │  (Error Tracking   │  │  (Genetic Algo    │  │  (Confidence,    │     │
  │   + Corrections)   │  │   for Retrieval)  │  │   Knowledge      │     │
  │                    │  │                   │  │   Gaps)           │     │
  └────────────────────┘  └───────────────────┘  └──────────────────┘     │
                                                                          │
  ┌────────────────────┐  ┌───────────────────────────────────────────┐   │
  │   Gap Filling      │  │         Obsidian Integration              │   │
  │  (Inference,       │  │  (Bidirectional sync, PARA, wikilinks,   │   │
  │   Consolidation,   │  │   MOC generation, dead link detection)    │   │
  │   Pattern Fill)    │  │                                           │   │
  └────────────────────┘  └───────────────────────────────────────────┘   │
          │                                                                │
          └──────────────── All backed by SQLite ──────────────────────────┘
```

## Quick Start

```python
from cortex import CortexEngine

# Create engine (file-backed for persistence, or :memory: for testing)
engine = CortexEngine("my_agent.db")

# Store different types of memories
engine.store("The capital of France is Paris", memory_type="semantic")
engine.remember("Had meeting with John about project X")
engine.learn_skill("When user asks for weather, use wttr.in API",
                   trigger="weather request")

# Recall with hybrid search (vector + BM25 + temporal + graph)
results = engine.recall("What do I know about France?")
for r in results:
    print(f"[{r.memory_type}] {r.content} (score: {r.score:.3f})")

# Run sleep/dream consolidation cycle
report = engine.consolidate()

# Run retrieval strategy evolution
engine.evolve(generations=5)

# Reflect on what you know (and don't know)
engine.reflect()
```

## 7 Cognitive Layers

### 1. Working Memory
Fast, limited-capacity ring buffer for immediate context.

```python
engine.store("Current conversation topic: AI safety", memory_type="working")
engine.working.get_recent(5)  # last 5 items
```

### 2. Episodic Memory
Timestamped events with temporal decay via Ebbinghaus forgetting curve.

```python
engine.remember("Team standup — discussed sprint goals", importance=0.7)
engine.episodic.recall("sprint goals", top_k=5)
```

### 3. Semantic Memory
Facts and knowledge with importance scoring and categories.

```python
engine.store("Python 3.12 supports PEP 695 type syntax",
             memory_type="semantic", category="programming", confidence=0.95)
```

### 4. Procedural Memory
Learned skills with success/failure tracking and reliability scoring.

```python
engine.learn_skill("Parse dates with dateutil.parser.parse()",
                   trigger="date parsing", pattern="string_to_date")
engine.procedural.record_outcome(skill_id, success=True)
```

### 5. Self-Improvement Engine
Tracks errors, learns correction patterns, measures learning velocity.

```python
err_id = engine.log_error("Used wrong date format", context="report generation")
engine.log_correction("Use YYYY-MM-DD format", error_id=err_id,
                      wrong="DD/MM/YYYY", pattern="date_format")
report = engine.get_improvement_report()
print(f"Resolution rate: {report.resolution_rate:.0%}")
print(f"Trend: {report.improvement_trend}")
```

### 6. Self-Evolution Engine
Genetic algorithm that evolves retrieval strategies over time.

```python
# Provide feedback on retrieval quality
engine.feedback(query="weather", result_id="sem_abc123", useful=True)

# Evolve strategies based on accumulated feedback
history = engine.evolve(generations=10)
best = engine.get_best_strategy()
print(f"Best strategy fitness: {best.fitness:.3f}")
print(f"Params: {best.params}")
```

Each strategy is a set of weights:
- `recency_weight` — how much to favor recent memories
- `frequency_weight` — how much to favor frequently accessed memories  
- `importance_weight` — how much to favor important memories
- `similarity_weight` — how much to favor semantically similar memories
- `decay_rate` — forgetting speed
- `consolidation_threshold` — when to promote memories
- `cross_memory_weight` — cross-type inference weight

### 7. Meta-Cognition Layer
Self-awareness about knowledge, confidence, and gaps.

```python
# Confidence assessment
confidence = engine.assess_confidence("quantum physics")
print(f"Confidence: {confidence.confidence:.2f}")
print(f"Coverage: {confidence.coverage:.2f}")
print(f"Gaps: {confidence.gaps}")

# Find what you don't know
gaps = engine.find_knowledge_gaps()

# Full self-evaluation
evaluation = engine.self_evaluate()
print(f"Quality: {evaluation.overall_quality:.2f}")
for rec in evaluation.recommendations:
    print(f"  → {rec}")
```

## Memory Consolidation (Sleep/Dream Cycle)

```python
report = engine.consolidate()
```

What happens:
1. **Working → Episodic**: Flushes the scratch pad into timestamped events
2. **Episodic → Semantic**: Promotes frequently-accessed or high-importance episodes to facts
3. **Episodic → Procedural**: Detects action patterns ("when X, do Y") and extracts skills
4. **Pruning**: Removes memories below the Ebbinghaus retention threshold

## Gap Filling

CORTEX actively fills knowledge gaps rather than just reporting them.

```python
# Find gaps
gaps = engine.find_gaps()
for gap in gaps:
    print(f"{gap.gap_type.value}: {gap.description} (priority: {gap.priority:.2f})")

# Auto-fill top gaps
results = engine.fill_gaps(max_fills=10)
for r in results:
    status = "✓" if r.success else "✗"
    print(f"  {status} [{r.strategy}] {r.content[:60]}")

# Fill specific gap
result = engine.fill_gap(gap_id, strategy="inference")

# Get question to ask the user
question = engine.request_fill(gap_id)

# Full gap analysis
report = engine.gap_report()
print(f"Total gaps: {report.total_gaps}")
print(f"Fill accuracy: {report.fill_accuracy:.0%}")
```

### Gap Types & Fill Strategies

| Gap Type | Description | Auto-Fillable? |
|----------|-------------|----------------|
| `INFERRABLE` | Can deduce from existing memories | ✓ |
| `CONSOLIDATABLE` | Can merge fragmented episodes | ✓ |
| `SEARCHABLE` | Needs external data | Returns search query |
| `ASKABLE` | Needs user input | Returns question |

### Fill Strategies
- **Inference**: Combine related facts to deduce new ones
- **Consolidation**: Merge episodic fragments into semantic summaries
- **Pattern**: Apply procedural patterns to fill gaps
- **Temporal**: Interpolate between known time-stamped events
- **Cross-Memory**: Use one memory type to fill gaps in another

## Obsidian Vault Integration

Bidirectional sync between CORTEX and an Obsidian vault.

```python
engine = CortexEngine(
    "cortex.db",
    obsidian_vault="/path/to/vault",
    obsidian_sync=True,        # auto-sync on consolidation
    obsidian_para=True,        # detect PARA method structure
    obsidian_export_inferred=True,
)

# Ingest vault into CORTEX
stats = engine.obsidian.ingest("/path/to/vault")

# Export CORTEX memories as Obsidian notes
stats = engine.obsidian.export("/path/to/vault")

# Bidirectional sync
stats = engine.obsidian.sync("/path/to/vault")
```

### What Gets Imported
| Note Type | Detection | Memory Type |
|-----------|-----------|-------------|
| Daily notes | `YYYY-MM-DD` filename | Episodic |
| MOCs / Indexes | "Map of Content", many links | Semantic |
| How-to guides | "How to", "Steps", "Guide" | Procedural |
| Zettelkasten cards | Short atomic notes | Semantic |
| Project notes | In `Projects/` folder | Episodic |

### What Gets Exported
- Semantic facts → `CORTEX/Knowledge/` notes
- Inferred memories → `CORTEX/Inferred/` notes (tagged `#cortex/inferred`)
- Procedural skills → `CORTEX/Skills/` notes
- Episodic memories → `memory/YYYY-MM-DD.md` entries
- Auto-generated **Maps of Content** from memory clusters
- `CORTEX-STATUS.md` with stats, coverage, gaps

### PARA Method Support
Detects the PARA folder structure:
- `Projects/` → Episodic memory (active, timestamped)
- `Areas/` → Semantic memory (ongoing knowledge)
- `Resources/` → Semantic + Procedural (reference material)
- `Archive/` → Low-importance memories

### Wikilink Graph
`[[wikilinks]]` are parsed and used for:
- Memory graph edges (linked notes share context)
- Importance scoring (highly-linked notes rank higher)
- Dead link detection → knowledge gaps

## Hippocampal Index (Hybrid Search)

Every `recall()` combines 5 signals:

| Signal | Description |
|--------|-------------|
| **Vector similarity** | Cosine similarity of TF-IDF hash embeddings |
| **BM25** | Okapi BM25 text matching |
| **Temporal** | Recency scoring (log-decay) |
| **Graph** | Edge connectivity in the memory graph |
| **Importance** | Stored importance score |

Weights are controlled by the evolving retrieval strategy.

## Tech Stack

- **Python 3.10+** — type-annotated throughout
- **SQLite** — zero-config persistent storage (WAL mode)
- **NumPy** — vector operations and embeddings
- **No external APIs** — runs fully offline
- **Optional**: `sentence-transformers` for higher-quality embeddings

## Installation

```bash
# From source
git clone https://github.com/anirach/cortex-memory.git
cd cortex-memory
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Optional: better embeddings
pip install sentence-transformers
```

## Project Structure

```
cortex-memory/
├── cortex/
│   ├── __init__.py           # Package init, exports CortexEngine
│   ├── engine.py             # Main orchestrator
│   ├── memories/
│   │   ├── working.py        # Ring buffer (Layer 1)
│   │   ├── episodic.py       # Timestamped events (Layer 2)
│   │   ├── semantic.py       # Facts & knowledge (Layer 3)
│   │   └── procedural.py     # Skills & patterns (Layer 4)
│   ├── consolidation.py      # Sleep/dream cycle
│   ├── forgetting.py         # Ebbinghaus forgetting curve
│   ├── self_improvement.py   # Error tracking (Layer 5)
│   ├── evolution.py          # Genetic algorithm (Layer 6)
│   ├── metacognition.py      # Confidence & gaps (Layer 7)
│   ├── gap_filling.py        # Active gap filling
│   ├── hippocampus.py        # Hybrid search index
│   ├── embeddings.py         # TF-IDF hash embeddings
│   ├── obsidian.py           # Obsidian vault integration
│   └── storage.py            # SQLite backend
├── tests/                    # pytest test suite
├── examples/                 # Usage examples
├── pyproject.toml            # Package config
└── README.md
```

## License

MIT
