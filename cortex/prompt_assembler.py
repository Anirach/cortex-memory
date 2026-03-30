"""
Prompt Assembler — automatic prompt engineering technique selection.

Analyzes task complexity and selects the optimal combination of prompt
engineering techniques (CoT, ReAct, Few-Shot, RAG, etc.) to build
high-quality prompts using CORTEX cognitive memory.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cortex.engine import CortexEngine


# ══════════════════════════════════════════════════════════
#  Enums & Data Classes
# ══════════════════════════════════════════════════════════


class PromptTechnique(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    SELF_CRITIQUE = "self_critique"
    RAG = "rag"
    ROLE_PROMPTING = "role_prompting"
    SYSTEM_TEMPLATE = "system_template"
    STRUCTURED_OUTPUT = "structured_output"
    SELF_CONSISTENCY = "self_consistency"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    KNOWLEDGE_GAP = "gap"
    SKILL_EXECUTION = "skill"


@dataclass
class AssembledPrompt:
    system_prompt: str
    user_prompt: str
    techniques_applied: list[PromptTechnique]
    context_memories: list[Any]
    few_shot_examples: list[Any]
    complexity: TaskComplexity
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════

COMPLEXITY_PROFILES: dict[TaskComplexity, list[PromptTechnique]] = {
    TaskComplexity.SIMPLE: [
        PromptTechnique.ZERO_SHOT,
        PromptTechnique.RAG,
    ],
    TaskComplexity.MODERATE: [
        PromptTechnique.RAG,
        PromptTechnique.ROLE_PROMPTING,
        PromptTechnique.CHAIN_OF_THOUGHT,
    ],
    TaskComplexity.COMPLEX: [
        PromptTechnique.RAG,
        PromptTechnique.ROLE_PROMPTING,
        PromptTechnique.CHAIN_OF_THOUGHT,
        PromptTechnique.SELF_CRITIQUE,
        PromptTechnique.SELF_CONSISTENCY,
    ],
    TaskComplexity.KNOWLEDGE_GAP: [
        PromptTechnique.RAG,
        PromptTechnique.REACT,
        PromptTechnique.CHAIN_OF_THOUGHT,
    ],
    TaskComplexity.SKILL_EXECUTION: [
        PromptTechnique.FEW_SHOT,
        PromptTechnique.ROLE_PROMPTING,
        PromptTechnique.STRUCTURED_OUTPUT,
    ],
}

SYSTEM_TEMPLATES: dict[str, str] = {
    "default": "You are a helpful AI assistant with access to a cognitive memory system.",
    "analyst": (
        "You are an expert analyst. Think carefully, cite evidence from "
        "provided context, and acknowledge uncertainty."
    ),
    "coder": (
        "You are an expert programmer. Write clean, tested code. "
        "Explain your approach before coding."
    ),
    "researcher": (
        "You are a research assistant. Use provided sources, cite "
        "specifically, distinguish fact from inference."
    ),
    "teacher": (
        "You are a patient teacher. Explain concepts clearly, use "
        "analogies, check understanding."
    ),
}

ROLE_DETECTION: dict[str, str] = {
    r"code|program|function|bug|debug|implement": "coder",
    r"research|paper|study|literature|cite": "researcher",
    r"explain|teach|learn|understand|concept": "teacher",
    r"analyze|compare|evaluate|assess|review": "analyst",
}


# ══════════════════════════════════════════════════════════
#  PromptAssembler
# ══════════════════════════════════════════════════════════


class PromptAssembler:
    """
    Automatically selects and combines prompt engineering techniques
    based on task complexity analysis, then builds high-quality prompts
    using CORTEX cognitive memory.
    """

    def __init__(self, engine: "CortexEngine") -> None:
        self.engine = engine
        self.complexity_profiles = dict(COMPLEXITY_PROFILES)
        self.assembly_history: list[dict[str, Any]] = []
        self.technique_registry: dict[PromptTechnique, Any] = {
            PromptTechnique.RAG: self._apply_rag,
            PromptTechnique.FEW_SHOT: self._apply_few_shot,
            PromptTechnique.CHAIN_OF_THOUGHT: self._apply_chain_of_thought,
            PromptTechnique.REACT: self._apply_react,
            PromptTechnique.SELF_CRITIQUE: self._apply_self_critique,
            PromptTechnique.ROLE_PROMPTING: self._apply_role,
            PromptTechnique.STRUCTURED_OUTPUT: self._apply_structured_output,
        }

    # ── Main Entry ──────────────────────────────────────────

    def assemble(
        self,
        query: str,
        task_hint: Optional[TaskComplexity] = None,
        role: Optional[str] = None,
        output_format: Optional[str] = None,
        max_context_memories: int = 5,
    ) -> AssembledPrompt:
        """Analyze task → select techniques → build prompt."""
        complexity = task_hint if task_hint is not None else self.analyze_complexity(query)
        techniques = list(self.complexity_profiles.get(complexity, []))

        # Add structured output if requested
        if output_format and PromptTechnique.STRUCTURED_OUTPUT not in techniques:
            techniques.append(PromptTechnique.STRUCTURED_OUTPUT)

        # Add system template
        if PromptTechnique.SYSTEM_TEMPLATE not in techniques:
            techniques.append(PromptTechnique.SYSTEM_TEMPLATE)

        # Detect or use explicit role
        detected_role = role or self._detect_role(query)

        # Retrieve context memories (RAG)
        context_memories: list[Any] = []
        if PromptTechnique.RAG in techniques:
            context_memories = self._apply_rag(query, max_context_memories)

        # Retrieve few-shot examples
        few_shot_examples: list[Any] = []
        if PromptTechnique.FEW_SHOT in techniques:
            few_shot_examples = self._apply_few_shot(query)

        # Build user prompt
        user_prompt = self._build_user_prompt(query, techniques, context_memories, detected_role, output_format)

        # Build system prompt
        system_prompt = self._build_system_prompt(techniques, detected_role, context_memories)

        # Compute confidence
        confidence = self._compute_confidence(complexity, context_memories, few_shot_examples)

        assembled = AssembledPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            techniques_applied=techniques,
            context_memories=context_memories,
            few_shot_examples=few_shot_examples,
            complexity=complexity,
            confidence=confidence,
            metadata={
                "role": detected_role,
                "output_format": output_format,
                "assembled_at": time.time(),
            },
        )

        return assembled

    # ── Complexity Analysis ─────────────────────────────────

    def analyze_complexity(self, query: str) -> TaskComplexity:
        """Classify query complexity using heuristics + memory signals."""
        query_lower = query.lower()
        word_count = len(query.split())

        # Check procedural memory first (skill execution)
        try:
            proc_matches = self.engine.procedural.recall(query, top_k=1)
            if proc_matches and proc_matches[0].reliability > 0.7:
                return TaskComplexity.SKILL_EXECUTION
        except Exception:
            pass

        # Check knowledge coverage (gap detection)
        # Only flag as gap if there are enough memories to make coverage meaningful
        try:
            total_memories = (
                self.engine.episodic.count()
                + self.engine.semantic.count()
                + self.engine.procedural.count()
            )
            if total_memories >= 3:
                assessment = self.engine.metacognition.assess_confidence(query)
                if assessment.coverage <= 0.3:
                    return TaskComplexity.KNOWLEDGE_GAP
        except Exception:
            pass

        # Heuristic complexity scoring
        complexity_score = 0

        # Multi-part indicators
        if any(w in query_lower for w in ["compare", "contrast", "analyze", "evaluate", "pros and cons"]):
            complexity_score += 3
        if any(w in query_lower for w in ["how", "why", "explain", "describe"]):
            complexity_score += 2
        if any(w in query_lower for w in ["and also", "furthermore", "additionally", "step by step"]):
            complexity_score += 2
        if word_count > 30:
            complexity_score += 2
        if query.count("?") > 1:
            complexity_score += 2

        # Simple factual
        if any(w in query_lower for w in ["what is", "who is", "when did", "define"]):
            complexity_score -= 1

        if complexity_score <= 1:
            return TaskComplexity.SIMPLE
        elif complexity_score <= 4:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.COMPLEX

    # ── Role Detection ──────────────────────────────────────

    def _detect_role(self, query: str) -> str:
        """Auto-detect role from query keywords."""
        query_lower = query.lower()
        for pattern, role in ROLE_DETECTION.items():
            if re.search(pattern, query_lower):
                return role
        return "default"

    # ── Technique Applications ──────────────────────────────

    def _apply_rag(self, query: str, max_results: int = 5) -> list[Any]:
        """Retrieve relevant memories from hippocampal index."""
        try:
            return self.engine.recall(query, top_k=max_results)
        except Exception:
            return []

    def _apply_few_shot(self, query: str) -> list[Any]:
        """Find relevant procedural memories as few-shot examples."""
        try:
            return self.engine.procedural.recall(query, top_k=3)
        except Exception:
            return []

    def _apply_chain_of_thought(self, query: str, context: list[Any]) -> str:
        """Wrap query in CoT reasoning structure."""
        ctx_text = self._format_context(context)
        return (
            f"Let me work through this step-by-step:\n\n"
            f"**Given context from memory:**\n{ctx_text}\n\n"
            f"**Question:** {query}\n\n"
            f"**Step-by-step reasoning:**\n"
            f"1. First, let me identify what we know from the retrieved memories...\n"
            f"2. Next, let me identify what additional reasoning is needed...\n"
            f"3. Now, let me synthesize an answer...\n\n"
            f"**Answer:**"
        )

    def _apply_react(self, query: str) -> str:
        """Build ReAct-style Think→Act→Observe template."""
        return (
            f"Use the following format to answer:\n\n"
            f"Thought: I need to figure out {query}\n"
            f"Action: Search memory for relevant information\n"
            f"Observation: [Retrieved memories will be inserted here]\n"
            f"Thought: Based on what I found, I can reason that...\n"
            f"Action: Check if there are knowledge gaps\n"
            f"Observation: [Gap analysis results]\n"
            f"Thought: Now I can formulate my answer\n"
            f"Answer: [Final synthesized answer]"
        )

    def _apply_self_critique(self, query: str) -> str:
        """Add self-critique instructions."""
        return (
            "After generating your response, apply this self-critique:\n\n"
            "1. **Accuracy check:** Does my answer align with the provided context memories?\n"
            "2. **Completeness check:** Did I address all parts of the question?\n"
            "3. **Confidence check:** Where am I uncertain? Flag those parts explicitly.\n"
            "4. **Bias check:** Am I making unsupported assumptions?\n"
            "5. **Revision:** If any checks fail, revise before presenting the final answer.\n\n"
            "If significant issues found, prefix with [REVISED]."
        )

    def _apply_role(self, query: str, role: str) -> str:
        """Return role/persona system text."""
        return SYSTEM_TEMPLATES.get(role, SYSTEM_TEMPLATES["default"])

    def _apply_structured_output(self, format_spec: str) -> str:
        """Add output format constraints."""
        return (
            f"\n\n**Output Format:** Please structure your response as: {format_spec}\n"
            f"Follow this format strictly."
        )

    # ── Prompt Building ─────────────────────────────────────

    def _build_system_prompt(
        self,
        techniques: list[PromptTechnique],
        role: str,
        context: list[Any],
    ) -> str:
        """Assemble final system prompt from components."""
        parts: list[str] = []

        # Role / system template
        if PromptTechnique.ROLE_PROMPTING in techniques or PromptTechnique.SYSTEM_TEMPLATE in techniques:
            parts.append(SYSTEM_TEMPLATES.get(role, SYSTEM_TEMPLATES["default"]))

        # Self-consistency instruction
        if PromptTechnique.SELF_CONSISTENCY in techniques:
            parts.append(
                "Generate multiple reasoning paths and select the most "
                "consistent answer across them."
            )

        # Context memories
        if context:
            parts.append("\n**Retrieved Context Memories:**")
            for i, mem in enumerate(context, 1):
                content = mem.content if hasattr(mem, "content") else str(mem)
                parts.append(f"  [{i}] {content}")

        return "\n".join(parts)

    def _build_user_prompt(
        self,
        query: str,
        techniques: list[PromptTechnique],
        context: list[Any],
        role: str,
        output_format: Optional[str],
    ) -> str:
        """Assemble the user-facing prompt."""
        parts: list[str] = []

        # Chain-of-thought wrapping
        if PromptTechnique.CHAIN_OF_THOUGHT in techniques:
            parts.append(self._apply_chain_of_thought(query, context))
        elif PromptTechnique.REACT in techniques:
            parts.append(self._apply_react(query))
        else:
            parts.append(query)

        # Self-critique appendage
        if PromptTechnique.SELF_CRITIQUE in techniques:
            parts.append("\n\n" + self._apply_self_critique(query))

        # Structured output
        if PromptTechnique.STRUCTURED_OUTPUT in techniques and output_format:
            parts.append(self._apply_structured_output(output_format))

        return "\n".join(parts)

    # ── Outcome Recording ───────────────────────────────────

    def record_outcome(
        self,
        assembled: AssembledPrompt,
        success: bool,
        feedback: str = "",
    ) -> None:
        """Record assembly result for self-improvement."""
        record = {
            "complexity": assembled.complexity.value,
            "techniques": [t.value for t in assembled.techniques_applied],
            "success": success,
            "feedback": feedback,
            "confidence": assembled.confidence,
            "timestamp": time.time(),
        }
        self.assembly_history.append(record)

        # Also log to self-improvement engine if available
        try:
            if success:
                self.engine.improvement.log_correction(
                    correct=f"Prompt assembly successful for {assembled.complexity.value} task",
                    pattern=f"complexity:{assembled.complexity.value}",
                )
            else:
                self.engine.improvement.log_error(
                    description=f"Prompt assembly failed for {assembled.complexity.value} task: {feedback}",
                    category="prompt_assembly",
                    severity=0.4,
                )
        except Exception:
            pass

    def get_technique_stats(self) -> dict[str, Any]:
        """Return success rates per technique combination."""
        if not self.assembly_history:
            return {"total_assemblies": 0, "by_complexity": {}, "by_technique": {}}

        by_complexity: dict[str, dict[str, int]] = {}
        by_technique: dict[str, dict[str, int]] = {}

        for record in self.assembly_history:
            comp = record["complexity"]
            if comp not in by_complexity:
                by_complexity[comp] = {"success": 0, "failure": 0}
            if record["success"]:
                by_complexity[comp]["success"] += 1
            else:
                by_complexity[comp]["failure"] += 1

            for tech in record["techniques"]:
                if tech not in by_technique:
                    by_technique[tech] = {"success": 0, "failure": 0}
                if record["success"]:
                    by_technique[tech]["success"] += 1
                else:
                    by_technique[tech]["failure"] += 1

        # Compute rates
        complexity_rates = {}
        for comp, counts in by_complexity.items():
            total = counts["success"] + counts["failure"]
            complexity_rates[comp] = {
                "total": total,
                "success_rate": counts["success"] / total if total > 0 else 0.0,
            }

        technique_rates = {}
        for tech, counts in by_technique.items():
            total = counts["success"] + counts["failure"]
            technique_rates[tech] = {
                "total": total,
                "success_rate": counts["success"] / total if total > 0 else 0.0,
            }

        return {
            "total_assemblies": len(self.assembly_history),
            "by_complexity": complexity_rates,
            "by_technique": technique_rates,
        }

    # ── Helpers ─────────────────────────────────────────────

    def _format_context(self, context: list[Any]) -> str:
        """Format context memories for prompt inclusion."""
        if not context:
            return "(No relevant memories found)"
        lines = []
        for i, mem in enumerate(context, 1):
            content = mem.content if hasattr(mem, "content") else str(mem)
            lines.append(f"  [{i}] {content}")
        return "\n".join(lines)

    def _compute_confidence(
        self,
        complexity: TaskComplexity,
        context: list[Any],
        few_shot: list[Any],
    ) -> float:
        """Compute assembler confidence in this particular assembly."""
        base = {
            TaskComplexity.SIMPLE: 0.9,
            TaskComplexity.MODERATE: 0.7,
            TaskComplexity.COMPLEX: 0.5,
            TaskComplexity.KNOWLEDGE_GAP: 0.3,
            TaskComplexity.SKILL_EXECUTION: 0.8,
        }.get(complexity, 0.5)

        # Boost for available context
        if context:
            base = min(1.0, base + 0.05 * len(context))
        if few_shot:
            base = min(1.0, base + 0.1 * len(few_shot))

        return round(base, 3)
