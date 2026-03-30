"""Tests for Obsidian vault integration."""

import os
import tempfile
import time
from pathlib import Path

import pytest

from cortex.storage import Storage
from cortex.memories.episodic import EpisodicMemory
from cortex.memories.semantic import SemanticMemory
from cortex.memories.procedural import ProceduralMemory
from cortex.hippocampus import HippocampalIndex
from cortex.obsidian import (
    ObsidianIntegration,
    _parse_frontmatter,
    _extract_wikilinks,
    _extract_tags,
    _detect_note_type,
    _detect_para_type,
)


# ── Parser unit tests ──────────────────────────────────────

class TestParsers:
    def test_parse_frontmatter(self):
        text = "---\ntitle: My Note\ntags: [ai, ml]\ndate: 2026-01-01\n---\n\nBody text here."
        meta, body = _parse_frontmatter(text)
        assert meta["title"] == "My Note"
        assert meta["tags"] == ["ai", "ml"]
        assert "Body text here" in body

    def test_parse_frontmatter_empty(self):
        meta, body = _parse_frontmatter("Just a plain note.")
        assert meta == {}
        assert "Just a plain note" in body

    def test_extract_wikilinks(self):
        text = "See [[Python]] and [[Machine Learning|ML]] for more."
        links = _extract_wikilinks(text)
        assert "Python" in links
        assert "Machine Learning" in links

    def test_extract_tags(self):
        text = "This is about #ai and #machine-learning. Also #cortex/inferred."
        tags = _extract_tags(text)
        assert "ai" in tags
        assert "machine-learning" in tags
        assert "cortex/inferred" in tags

    def test_detect_note_type_daily(self):
        assert _detect_note_type("2026-01-15", "Today I did...", {}, "") == "daily"

    def test_detect_note_type_moc(self):
        assert _detect_note_type("Map of Content - AI", "Overview of topics", {}, "") == "moc"

    def test_detect_note_type_howto(self):
        assert _detect_note_type("How to Deploy Docker", "Steps to...", {}, "") == "howto"

    def test_detect_para_type(self):
        assert _detect_para_type("Projects/My Project/notes.md") == "episodic"
        assert _detect_para_type("Areas/Health/diet.md") == "semantic"
        assert _detect_para_type("Resources/Books/review.md") == "semantic"
        assert _detect_para_type("Archive/old-stuff/note.md") == "episodic"
        assert _detect_para_type("random/note.md") is None


# ── Integration tests with temp vault ──────────────────────

class TestObsidianIntegration:
    def setup_method(self):
        self.storage = Storage(":memory:")
        self.episodic = EpisodicMemory(self.storage)
        self.semantic = SemanticMemory(self.storage)
        self.procedural = ProceduralMemory(self.storage)
        self.hippocampus = HippocampalIndex(self.storage)
        self.obsidian = ObsidianIntegration(
            self.storage,
            self.episodic, self.semantic, self.procedural,
            self.hippocampus,
        )
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_note(self, rel_path: str, content: str) -> Path:
        path = Path(self.tmpdir) / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_ingest_simple_note(self):
        self._write_note("Knowledge/Python.md", """---
tags: [programming, python]
---

# Python

Python is a versatile programming language used in AI, web development, and more.
""")
        stats = self.obsidian.ingest(self.tmpdir)
        assert stats["imported"] >= 1
        assert stats["errors"] == 0

        # Should be in semantic memory
        results = self.semantic.recall("Python programming")
        assert len(results) >= 1

    def test_ingest_daily_note(self):
        self._write_note("memory/2026-03-30.md", """# 2026-03-30

## Morning
Had a productive meeting about CORTEX architecture.

## Afternoon
Implemented gap filling engine.
""")
        stats = self.obsidian.ingest(self.tmpdir)
        assert stats["imported"] >= 1

        # Daily notes → episodic
        results = self.episodic.recall("CORTEX architecture")
        assert len(results) >= 1

    def test_ingest_howto_note(self):
        self._write_note("Guides/How to Deploy Docker.md", """# How to Deploy Docker

1. Write a Dockerfile
2. Build the image: `docker build -t myapp .`
3. Run: `docker run -p 8080:8080 myapp`
""")
        stats = self.obsidian.ingest(self.tmpdir)
        assert stats["imported"] >= 1

        # How-to → procedural
        results = self.procedural.recall("deploy docker")
        assert len(results) >= 1

    def test_ingest_para_structure(self):
        self._write_note("Projects/Alpha/notes.md", "Project Alpha meeting notes")
        self._write_note("Areas/Health/nutrition.md", "Important nutrition facts")
        self._write_note("Resources/Books/deep-work.md", "Deep Work by Cal Newport")

        stats = self.obsidian.ingest(self.tmpdir)
        assert stats["imported"] >= 3

    def test_ingest_wikilinks_build_graph(self):
        self._write_note("NoteA.md", "See also [[NoteB]] and [[NoteC]]")
        self._write_note("NoteB.md", "References [[NoteA]]")
        self._write_note("NoteC.md", "Standalone note")

        stats = self.obsidian.ingest(self.tmpdir)
        assert stats["links"] >= 2

    def test_ingest_skip_hidden_dirs(self):
        self._write_note(".obsidian/config.json", '{"key": "value"}')
        self._write_note("visible.md", "Visible note")

        stats = self.obsidian.ingest(self.tmpdir)
        assert stats["imported"] == 1  # only visible.md

    def test_ingest_idempotent(self):
        self._write_note("test.md", "Test content")
        stats1 = self.obsidian.ingest(self.tmpdir)
        assert stats1["imported"] == 1

        stats2 = self.obsidian.ingest(self.tmpdir)
        assert stats2["skipped"] == 1
        assert stats2["imported"] == 0

    def test_ingest_detects_changes(self):
        path = self._write_note("changeable.md", "Version 1")
        stats1 = self.obsidian.ingest(self.tmpdir)
        assert stats1["imported"] == 1

        path.write_text("Version 2 with updates", encoding="utf-8")
        stats2 = self.obsidian.ingest(self.tmpdir)
        assert stats2["updated"] == 1

    def test_export_creates_notes(self):
        # Store some memories
        self.semantic.store("Earth orbits the Sun", category="science", source="user")
        self.procedural.store("Run pytest to test code", trigger="testing")

        output_dir = os.path.join(self.tmpdir, "export_vault")
        stats = self.obsidian.export(output_dir)
        assert stats["notes_created"] >= 2

        # Check files exist
        cortex_dir = Path(output_dir) / "CORTEX"
        assert cortex_dir.exists()

    def test_export_status_file(self):
        self.semantic.store("Test fact", source="user")
        output_dir = os.path.join(self.tmpdir, "status_vault")
        self.obsidian.export(output_dir)

        status_file = Path(output_dir) / "CORTEX-STATUS.md"
        assert status_file.exists()
        content = status_file.read_text()
        assert "CORTEX Status" in content
        assert "Semantic" in content

    def test_export_daily_notes(self):
        self.episodic.store("Morning standup meeting", source="user")
        self.episodic.store("Afternoon code review", source="user")

        output_dir = os.path.join(self.tmpdir, "daily_vault")
        stats = self.obsidian.export(output_dir)
        assert stats["notes_created"] >= 1

        memory_dir = Path(output_dir) / "memory"
        assert memory_dir.exists()
        daily_files = list(memory_dir.glob("*.md"))
        assert len(daily_files) >= 1

    def test_export_generates_mocs(self):
        # Store enough facts in one category
        for i in range(5):
            self.semantic.store(f"Science fact {i} about physics", category="science", source="user")

        output_dir = os.path.join(self.tmpdir, "moc_vault")
        stats = self.obsidian.export(output_dir)
        assert stats["mocs_generated"] >= 1

        moc_dir = Path(output_dir) / "CORTEX" / "MOCs"
        assert moc_dir.exists()

    def test_export_skips_obsidian_sourced(self):
        # Memories from obsidian should not be re-exported
        self.semantic.store("From vault", source="obsidian")
        output_dir = os.path.join(self.tmpdir, "skip_vault")
        stats = self.obsidian.export(output_dir)
        assert stats["notes_created"] == 0

    def test_sync_bidirectional(self):
        # Create a vault note
        self._write_note("sync_test.md", "Original vault content about sync testing")

        # Store a CORTEX memory
        self.semantic.store("CORTEX-generated knowledge about syncing", source="user")

        stats = self.obsidian.sync(self.tmpdir)
        assert stats["ingest"]["imported"] >= 1
        assert stats["export"]["notes_created"] >= 1

    def test_export_inferred_memories(self):
        self.semantic.store("Inferred: Paris is in Europe", source="gap_fill",
                           tags=["cortex_inferred"])
        output_dir = os.path.join(self.tmpdir, "inferred_vault")
        stats = self.obsidian.export(output_dir)
        assert stats["notes_created"] >= 1

        inferred_dir = Path(output_dir) / "CORTEX" / "Inferred"
        assert inferred_dir.exists()

    def test_dead_link_gap_detection(self):
        from cortex.gap_filling import GapFillingEngine
        gap_engine = GapFillingEngine(
            self.storage, self.episodic, self.semantic, self.procedural
        )
        self.obsidian.gap_filling = gap_engine

        self._write_note("test_dead.md", "See [[NonExistentNote]] for more info.")
        self.obsidian.ingest(self.tmpdir)

        # Should detect dead link as gap
        dead_gaps = [g for g in gap_engine._gap_cache.values() if "dead" in g.id.lower()]
        assert len(dead_gaps) >= 1
        assert dead_gaps[0].topic == "NonExistentNote"

    def test_vault_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.obsidian.ingest("/nonexistent/vault/path")

    def test_frontmatter_tags_merged(self):
        self._write_note("tagged.md", """---
tags: [ai, research]
---

This note has #extra-tag and content about AI research.
""")
        stats = self.obsidian.ingest(self.tmpdir)
        assert stats["imported"] == 1

        # Check tags were merged
        results = self.semantic.recall("AI research")
        if results:
            fact = results[0]
            assert "ai" in fact.tags or "research" in fact.tags
