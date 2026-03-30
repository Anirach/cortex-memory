"""
CORTEX — Cognitive Orchestrated Retrieval with Temporal EXperience.

A self-improving cognitive memory architecture for AI agents featuring:
- 4-layer memory system (working, episodic, semantic, procedural)
- Ebbinghaus forgetting curve with memory consolidation
- Self-improvement engine (error tracking + correction learning)
- Self-evolution via genetic algorithm for retrieval strategies
- Meta-cognition layer (confidence scoring, knowledge gap detection)
- Hippocampal hybrid search index (vector + BM25 + temporal + graph)
"""

__version__ = "0.1.0"
__author__ = "Anirach"

from cortex.engine import CortexEngine

__all__ = ["CortexEngine", "__version__"]
