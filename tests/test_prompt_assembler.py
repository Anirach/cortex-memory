"""Tests for the PromptAssembler module."""

import pytest

from cortex.engine import CortexEngine
from cortex.prompt_assembler import (
    AssembledPrompt,
    COMPLEXITY_PROFILES,
    PromptAssembler,
    PromptTechnique,
    ROLE_DETECTION,
    SYSTEM_TEMPLATES,
    TaskComplexity,
)


class TestComplexityAnalysis:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_simple_factual(self):
        assert self.pa.analyze_complexity("What is Python?") == TaskComplexity.SIMPLE

    def test_simple_who(self):
        assert self.pa.analyze_complexity("Who is Alan Turing?") == TaskComplexity.SIMPLE

    def test_moderate_how(self):
        assert self.pa.analyze_complexity("How does gradient descent work?") == TaskComplexity.MODERATE

    def test_moderate_why(self):
        assert self.pa.analyze_complexity("Why is the sky blue?") == TaskComplexity.MODERATE

    def test_complex_compare(self):
        result = self.pa.analyze_complexity(
            "Compare and contrast the pros and cons of microservices vs monoliths "
            "and evaluate which is better for startups. How do they differ and why "
            "would you choose one over the other? Additionally, explain the tradeoffs "
            "step by step for a growing engineering team"
        )
        assert result == TaskComplexity.COMPLEX

    def test_complex_multi_question(self):
        result = self.pa.analyze_complexity(
            "How does quantum computing work and why is it faster than classical "
            "computing? Furthermore, explain the current limitations and compare "
            "the leading approaches?"
        )
        assert result == TaskComplexity.COMPLEX

    def test_skill_execution(self):
        """Procedural memory with high reliability triggers SKILL_EXECUTION."""
        self.engine.learn_skill(
            "When generating a report, use the template",
            trigger="generate report",
        )
        # Manually boost success count to get reliability > 0.7
        skills = self.engine.procedural.get_all()
        skill_id = skills[0].id
        for _ in range(10):
            self.engine.procedural.record_outcome(skill_id, True)

        result = self.pa.analyze_complexity("generate report for Q4")
        assert result == TaskComplexity.SKILL_EXECUTION

    def test_knowledge_gap(self):
        """Topic with no coverage triggers KNOWLEDGE_GAP."""
        # Store memories about other topics so there's baseline data (>= 3 needed)
        self.engine.store("Python is a programming language", memory_type="semantic")
        self.engine.store("Java is a programming language", memory_type="semantic")
        self.engine.store("JavaScript runs in the browser", memory_type="semantic")

        # Query about a topic with zero coverage
        result = self.pa.analyze_complexity("What is the Riemann hypothesis?")
        assert result == TaskComplexity.KNOWLEDGE_GAP


class TestSimpleAssembly:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_simple_assembly_techniques(self):
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        assert PromptTechnique.ZERO_SHOT in result.techniques_applied
        assert PromptTechnique.RAG in result.techniques_applied
        assert result.complexity == TaskComplexity.SIMPLE

    def test_simple_has_system_prompt(self):
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        assert len(result.system_prompt) > 0

    def test_simple_has_user_prompt(self):
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        assert "Python" in result.user_prompt

    def test_simple_confidence_high(self):
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        assert result.confidence >= 0.8


class TestComplexAssembly:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_complex_techniques(self):
        result = self.pa.assemble(
            "Compare Python and Java",
            task_hint=TaskComplexity.COMPLEX,
        )
        assert PromptTechnique.RAG in result.techniques_applied
        assert PromptTechnique.CHAIN_OF_THOUGHT in result.techniques_applied
        assert PromptTechnique.SELF_CRITIQUE in result.techniques_applied
        assert PromptTechnique.SELF_CONSISTENCY in result.techniques_applied

    def test_complex_cot_in_prompt(self):
        result = self.pa.assemble(
            "Compare Python and Java",
            task_hint=TaskComplexity.COMPLEX,
        )
        assert "step-by-step" in result.user_prompt.lower()

    def test_complex_self_critique_in_prompt(self):
        result = self.pa.assemble(
            "Compare Python and Java",
            task_hint=TaskComplexity.COMPLEX,
        )
        assert "self-critique" in result.user_prompt.lower()

    def test_complex_lower_confidence(self):
        result = self.pa.assemble(
            "Compare Python and Java",
            task_hint=TaskComplexity.COMPLEX,
        )
        assert result.confidence <= 0.8


class TestGapAssembly:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_gap_techniques(self):
        result = self.pa.assemble(
            "Tell me about dark matter",
            task_hint=TaskComplexity.KNOWLEDGE_GAP,
        )
        assert PromptTechnique.RAG in result.techniques_applied
        assert PromptTechnique.REACT in result.techniques_applied
        assert PromptTechnique.CHAIN_OF_THOUGHT in result.techniques_applied

    def test_gap_react_in_prompt(self):
        result = self.pa.assemble(
            "Tell me about dark matter",
            task_hint=TaskComplexity.KNOWLEDGE_GAP,
        )
        # ReAct takes precedence in user prompt since no CoT for gap (REACT listed)
        # Actually both are in techniques, but REACT is checked first in _build_user_prompt
        # because CHAIN_OF_THOUGHT is checked first... let me verify
        # The build order: CoT first, then ReAct. Both are in gap profile.
        # So CoT will be applied.
        assert "step-by-step" in result.user_prompt.lower() or "thought:" in result.user_prompt.lower()

    def test_gap_low_confidence(self):
        result = self.pa.assemble(
            "Tell me about dark matter",
            task_hint=TaskComplexity.KNOWLEDGE_GAP,
        )
        assert result.confidence <= 0.5


class TestSkillAssembly:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_skill_techniques(self):
        result = self.pa.assemble(
            "How to deploy",
            task_hint=TaskComplexity.SKILL_EXECUTION,
        )
        assert PromptTechnique.FEW_SHOT in result.techniques_applied
        assert PromptTechnique.ROLE_PROMPTING in result.techniques_applied
        assert PromptTechnique.STRUCTURED_OUTPUT in result.techniques_applied

    def test_skill_with_procedural_memory(self):
        self.engine.learn_skill(
            "When deploying, first run tests, then build, then push",
            trigger="deploy",
            pattern="deploy pipeline",
        )
        result = self.pa.assemble(
            "How to deploy the app",
            task_hint=TaskComplexity.SKILL_EXECUTION,
        )
        assert len(result.few_shot_examples) >= 1

    def test_skill_confidence(self):
        result = self.pa.assemble(
            "How to deploy",
            task_hint=TaskComplexity.SKILL_EXECUTION,
        )
        assert result.confidence >= 0.7


class TestRAGIntegration:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_rag_retrieves_memories(self):
        self.engine.store("Python is a high-level programming language", memory_type="semantic")
        self.engine.store("Python was created by Guido van Rossum", memory_type="semantic")
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        assert len(result.context_memories) >= 1

    def test_rag_memories_in_system_prompt(self):
        self.engine.store("Python is a high-level programming language", memory_type="semantic")
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        assert "Python" in result.system_prompt

    def test_rag_empty_memory(self):
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        assert result.context_memories == []


class TestFewShotFromProcedural:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_few_shot_retrieves_skills(self):
        self.engine.learn_skill(
            "When writing tests, use pytest fixtures",
            trigger="write tests",
            pattern="testing pattern",
        )
        result = self.pa.assemble(
            "Write tests for the module",
            task_hint=TaskComplexity.SKILL_EXECUTION,
        )
        assert len(result.few_shot_examples) >= 1
        assert "pytest" in result.few_shot_examples[0].content

    def test_few_shot_empty_when_no_skills(self):
        result = self.pa.assemble(
            "Write tests for the module",
            task_hint=TaskComplexity.SKILL_EXECUTION,
        )
        assert result.few_shot_examples == []


class TestRoleDetection:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_detect_coder(self):
        assert self.pa._detect_role("Write a function to sort a list") == "coder"

    def test_detect_researcher(self):
        assert self.pa._detect_role("Find research papers about transformers") == "researcher"

    def test_detect_teacher(self):
        assert self.pa._detect_role("Explain how neural networks learn") == "teacher"

    def test_detect_analyst(self):
        assert self.pa._detect_role("Analyze the market trends") == "analyst"

    def test_detect_default(self):
        assert self.pa._detect_role("Hello there") == "default"

    def test_role_in_system_prompt(self):
        result = self.pa.assemble(
            "Write a function to sort data",
            task_hint=TaskComplexity.MODERATE,
        )
        assert "programmer" in result.system_prompt.lower() or "coder" in result.system_prompt.lower()


class TestCustomRoleOverride:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_explicit_role_overrides(self):
        result = self.pa.assemble(
            "Write a function to sort data",
            role="teacher",
            task_hint=TaskComplexity.MODERATE,
        )
        assert "teacher" in result.system_prompt.lower()
        assert result.metadata["role"] == "teacher"

    def test_custom_unknown_role_falls_to_default(self):
        result = self.pa.assemble(
            "Hello",
            role="astronaut",
            task_hint=TaskComplexity.SIMPLE,
        )
        assert "helpful AI assistant" in result.system_prompt


class TestStructuredOutput:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_structured_output_added(self):
        result = self.pa.assemble(
            "List the top 5 languages",
            output_format="JSON array",
            task_hint=TaskComplexity.SIMPLE,
        )
        assert PromptTechnique.STRUCTURED_OUTPUT in result.techniques_applied
        assert "JSON array" in result.user_prompt

    def test_structured_output_not_added_without_format(self):
        result = self.pa.assemble(
            "List the top 5 languages",
            task_hint=TaskComplexity.SIMPLE,
        )
        assert PromptTechnique.STRUCTURED_OUTPUT not in result.techniques_applied


class TestOutcomeRecording:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_record_success(self):
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        self.pa.record_outcome(result, success=True, feedback="Good answer")
        assert len(self.pa.assembly_history) == 1
        assert self.pa.assembly_history[0]["success"] is True

    def test_record_failure(self):
        result = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        self.pa.record_outcome(result, success=False, feedback="Wrong answer")
        assert len(self.pa.assembly_history) == 1
        assert self.pa.assembly_history[0]["success"] is False

    def test_multiple_recordings(self):
        r1 = self.pa.assemble("What is Python?", task_hint=TaskComplexity.SIMPLE)
        r2 = self.pa.assemble("Compare X and Y", task_hint=TaskComplexity.COMPLEX)
        self.pa.record_outcome(r1, success=True)
        self.pa.record_outcome(r2, success=False)
        assert len(self.pa.assembly_history) == 2


class TestTechniqueStats:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_empty_stats(self):
        stats = self.pa.get_technique_stats()
        assert stats["total_assemblies"] == 0

    def test_stats_after_recordings(self):
        r1 = self.pa.assemble("What is X?", task_hint=TaskComplexity.SIMPLE)
        r2 = self.pa.assemble("What is Y?", task_hint=TaskComplexity.SIMPLE)
        r3 = self.pa.assemble("Compare A and B", task_hint=TaskComplexity.COMPLEX)
        self.pa.record_outcome(r1, success=True)
        self.pa.record_outcome(r2, success=False)
        self.pa.record_outcome(r3, success=True)

        stats = self.pa.get_technique_stats()
        assert stats["total_assemblies"] == 3
        assert "simple" in stats["by_complexity"]
        assert stats["by_complexity"]["simple"]["total"] == 2
        assert stats["by_complexity"]["simple"]["success_rate"] == 0.5


class TestDefaultWithoutEngineData:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")
        self.pa = self.engine.prompt_assembler

    def teardown_method(self):
        self.engine.close()

    def test_assembles_with_empty_memory(self):
        result = self.pa.assemble("What is Python?")
        assert isinstance(result, AssembledPrompt)
        assert len(result.system_prompt) > 0
        assert len(result.user_prompt) > 0
        assert result.context_memories == []

    def test_all_complexity_types_work_empty(self):
        for complexity in TaskComplexity:
            result = self.pa.assemble("Test query", task_hint=complexity)
            assert isinstance(result, AssembledPrompt)

    def test_engine_assemble_prompt_convenience(self):
        result = self.engine.assemble_prompt("Hello world")
        assert isinstance(result, AssembledPrompt)


class TestAssembledPromptDataclass:
    def setup_method(self):
        self.engine = CortexEngine(db_path=":memory:")

    def teardown_method(self):
        self.engine.close()

    def test_metadata_contains_role(self):
        result = self.engine.assemble_prompt("Write code", task_hint=TaskComplexity.SIMPLE)
        assert "role" in result.metadata

    def test_metadata_contains_assembled_at(self):
        result = self.engine.assemble_prompt("Hello", task_hint=TaskComplexity.SIMPLE)
        assert "assembled_at" in result.metadata
        assert result.metadata["assembled_at"] > 0
