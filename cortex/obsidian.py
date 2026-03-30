"""
Obsidian Vault Integration — bidirectional sync between CORTEX memory and an
Obsidian vault on disk.

Features:
- Vault ingestion (Obsidian → CORTEX): parse .md, frontmatter, wikilinks, tags
- Vault export (CORTEX → Obsidian): write consolidated memories as notes
- Wikilink graph as memory graph
- PARA method detection
- Dead-link gap detection
- Auto-generate MOCs (Maps of Content)
- Special file handling (MEMORY.md, SOUL.md, daily notes)
"""

from __future__ import annotations

import os
import re
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from cortex.storage import Storage

if TYPE_CHECKING:
    from cortex.memories.episodic import EpisodicMemory
    from cortex.memories.semantic import SemanticMemory
    from cortex.memories.procedural import ProceduralMemory
    from cortex.hippocampus import HippocampalIndex
    from cortex.gap_filling import GapFillingEngine, Gap


# ── Frontmatter parser (YAML-lite, no dependency) ──────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]+(?:\.[a-zA-Z0-9_/-]+)*)\b", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML-like frontmatter and body from markdown text."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    raw = m.group(1)
    body = text[m.end():]
    meta: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Handle lists
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",")]
            elif val == "":
                val = True
            meta[key] = val
    return meta, body


def _extract_wikilinks(text: str) -> list[str]:
    """Extract all [[wikilink]] targets from text."""
    return _WIKILINK_RE.findall(text)


def _extract_tags(text: str) -> list[str]:
    """Extract all #tags from text."""
    return _TAG_RE.findall(text)


# ── PARA detection ──────────────────────────────────────────

_PARA_FOLDERS = {
    "projects": "episodic",
    "areas": "semantic",
    "resources": "semantic",
    "archive": "episodic",
}


def _detect_para_type(rel_path: str) -> str | None:
    """Detect PARA folder from relative path."""
    parts = Path(rel_path).parts
    if parts:
        first = parts[0].lower()
        return _PARA_FOLDERS.get(first)
    return None


# ── Note type detection ─────────────────────────────────────

_DAILY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_HOW_TO_RE = re.compile(r"(how.to|guide|tutorial|steps|workflow|recipe)", re.I)
_MOC_INDICATORS = {"map of content", "moc", "index", "overview", "hub"}


@dataclass
class VaultNote:
    """Parsed representation of an Obsidian note."""
    path: str
    rel_path: str
    title: str
    body: str
    frontmatter: dict[str, Any]
    wikilinks: list[str]
    tags: list[str]
    headings: list[str]
    note_type: str            # daily, moc, howto, zettelkasten, project, general
    para_type: str | None     # episodic, semantic, or None
    content_hash: str
    modified_at: float


def _detect_note_type(title: str, body: str, frontmatter: dict, rel_path: str) -> str:
    """Classify note type based on content signals."""
    title_lower = title.lower()

    if _DAILY_RE.match(title):
        return "daily"

    if rel_path.startswith("memory/") and _DAILY_RE.match(title):
        return "daily"

    body_lower = body.lower()
    if any(ind in title_lower or ind in body_lower[:200] for ind in _MOC_INDICATORS):
        return "moc"

    if _HOW_TO_RE.search(title) or _HOW_TO_RE.search(body[:300]):
        return "howto"

    wikilinks = _extract_wikilinks(body)
    if len(wikilinks) > 10 and len(body.split()) < 200:
        return "moc"  # Lots of links, little prose = likely MOC

    if frontmatter.get("type") == "zettelkasten" or len(body.split()) < 300:
        return "zettelkasten"

    if "project" in rel_path.lower() or frontmatter.get("type") == "project":
        return "project"

    return "general"


class ObsidianIntegration:
    """
    Bidirectional sync between an Obsidian vault and CORTEX memory.
    """

    # Table to track which files have been ingested (and their hashes)
    SYNC_DDL = """
    CREATE TABLE IF NOT EXISTS obsidian_sync (
        rel_path    TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        memory_id   TEXT,
        memory_type TEXT,
        synced_at   REAL NOT NULL,
        direction   TEXT DEFAULT 'ingest'
    );
    """

    def __init__(
        self,
        storage: Storage,
        episodic: "EpisodicMemory",
        semantic: "SemanticMemory",
        procedural: "ProceduralMemory",
        hippocampus: "HippocampalIndex",
        gap_filling: "GapFillingEngine | None" = None,
        para_enabled: bool = True,
        export_inferred: bool = True,
    ) -> None:
        self.storage = storage
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.hippocampus = hippocampus
        self.gap_filling = gap_filling
        self.para_enabled = para_enabled
        self.export_inferred = export_inferred

        # Ensure sync tracking table exists
        conn = storage._get_conn()
        conn.executescript(self.SYNC_DDL)
        conn.commit()

    # ══════════════════════════════════════════════════════════
    #  INGEST  (Obsidian → CORTEX)
    # ══════════════════════════════════════════════════════════

    def ingest(self, vault_path: str, force: bool = False) -> dict[str, Any]:
        """
        Scan vault and import notes into CORTEX memory.

        Returns stats dict with counts of imported/skipped/updated notes.
        """
        vault = Path(vault_path)
        if not vault.is_dir():
            raise FileNotFoundError(f"Vault not found: {vault_path}")

        stats = {"imported": 0, "updated": 0, "skipped": 0, "errors": 0, "links": 0}
        notes: list[VaultNote] = []

        for md_file in vault.rglob("*.md"):
            if any(p.startswith(".") for p in md_file.parts):
                continue  # Skip hidden dirs (.obsidian, .trash)

            try:
                note = self._parse_note(md_file, vault)
                notes.append(note)
            except Exception:
                stats["errors"] += 1
                continue

        # Phase 1: Import notes as memories
        for note in notes:
            result = self._import_note(note, force)
            stats[result] += 1

        # Phase 2: Build graph from wikilinks
        for note in notes:
            links_added = self._build_links(note, notes)
            stats["links"] += links_added

        # Phase 3: Detect dead links as gaps
        if self.gap_filling:
            self._detect_dead_links(notes, vault)

        return stats

    def _parse_note(self, file_path: Path, vault_root: Path) -> VaultNote:
        """Parse a single .md file into a VaultNote."""
        text = file_path.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = _parse_frontmatter(text)
        rel_path = str(file_path.relative_to(vault_root))
        title = file_path.stem
        content_hash = hashlib.md5(text.encode()).hexdigest()

        wikilinks = _extract_wikilinks(text)
        tags = _extract_tags(text)
        # Add frontmatter tags
        fm_tags = frontmatter.get("tags", [])
        if isinstance(fm_tags, str):
            fm_tags = [t.strip() for t in fm_tags.split(",")]
        elif isinstance(fm_tags, list):
            fm_tags = [str(t).strip() for t in fm_tags]
        else:
            fm_tags = []
        tags = list(set(tags + fm_tags))

        headings = [m.group(2) for m in _HEADING_RE.finditer(body)]

        note_type = _detect_note_type(title, body, frontmatter, rel_path)
        para_type = _detect_para_type(rel_path) if self.para_enabled else None

        return VaultNote(
            path=str(file_path),
            rel_path=rel_path,
            title=title,
            body=body.strip(),
            frontmatter=frontmatter,
            wikilinks=wikilinks,
            tags=tags,
            headings=headings,
            note_type=note_type,
            para_type=para_type,
            content_hash=content_hash,
            modified_at=file_path.stat().st_mtime,
        )

    def _import_note(self, note: VaultNote, force: bool) -> str:
        """Import a single note. Returns 'imported', 'updated', or 'skipped'."""
        # Check if already synced with same hash
        existing = self.storage.fetch_one("obsidian_sync", note.rel_path, "rel_path")
        if existing and not force:
            if existing["content_hash"] == note.content_hash:
                return "skipped"
            # Content changed → update
            if existing.get("memory_id"):
                self._update_memory(existing["memory_id"], existing["memory_type"], note)
                self._update_sync_record(note, existing["memory_id"], existing["memory_type"])
                return "updated"

        # Determine target memory type
        mem_type = self._note_to_memory_type(note)
        memory_id = self._store_note(note, mem_type)
        self._update_sync_record(note, memory_id, mem_type)
        return "imported"

    def _note_to_memory_type(self, note: VaultNote) -> str:
        """Decide which memory type a note maps to."""
        if note.para_type:
            return note.para_type

        type_map = {
            "daily": "episodic",
            "moc": "semantic",
            "howto": "procedural",
            "zettelkasten": "semantic",
            "project": "episodic",
            "general": "semantic",
        }
        return type_map.get(note.note_type, "semantic")

    def _store_note(self, note: VaultNote, mem_type: str) -> str:
        """Store note into the appropriate memory subsystem. Returns memory ID."""
        metadata = {
            "vault_path": note.rel_path,
            "note_type": note.note_type,
            "headings": note.headings[:10],
            "wikilinks": note.wikilinks[:20],
            **{k: v for k, v in note.frontmatter.items() if isinstance(v, (str, int, float, bool))},
        }
        importance = self._estimate_importance(note)

        if mem_type == "episodic":
            ep = self.episodic.store(
                content=f"{note.title}: {note.body[:2000]}",
                importance=importance,
                tags=note.tags,
                metadata=metadata,
                source="obsidian",
            )
            return ep.id

        elif mem_type == "procedural":
            sk = self.procedural.store(
                content=f"{note.title}: {note.body[:2000]}",
                pattern=note.note_type,
                trigger=note.title,
                importance=importance,
                tags=note.tags,
                metadata=metadata,
            )
            return sk.id

        else:  # semantic
            category = note.tags[0] if note.tags else "vault"
            fact = self.semantic.store(
                content=f"{note.title}: {note.body[:2000]}",
                importance=importance,
                category=category,
                confidence=0.9,
                tags=note.tags,
                metadata=metadata,
                source="obsidian",
            )
            return fact.id

    def _update_memory(self, memory_id: str, mem_type: str, note: VaultNote) -> None:
        """Update an existing memory with new content from the note."""
        table_map = {
            "episodic": "episodic_memory",
            "semantic": "semantic_memory",
            "procedural": "procedural_memory",
        }
        table = table_map.get(mem_type, "semantic_memory")
        from cortex.embeddings import embed_text
        embedding = embed_text(f"{note.title}: {note.body[:2000]}")
        self.storage.execute(
            f"UPDATE {table} SET content = ?, embedding = ?, last_accessed = ? WHERE id = ?",
            (f"{note.title}: {note.body[:2000]}", embedding.tobytes(), time.time(), memory_id),
        )

    def _estimate_importance(self, note: VaultNote) -> float:
        """Estimate note importance from signals."""
        score = 0.5
        # More wikilinks = more connected = more important
        score += min(0.2, len(note.wikilinks) * 0.02)
        # More tags = better categorised
        score += min(0.1, len(note.tags) * 0.02)
        # Longer content might be more substantial
        word_count = len(note.body.split())
        score += min(0.1, word_count / 2000)
        # Special files
        if note.title.lower() in ("memory", "soul", "agents"):
            score = 0.95
        return min(1.0, score)

    def _update_sync_record(self, note: VaultNote, memory_id: str, mem_type: str) -> None:
        self.storage.insert("obsidian_sync", {
            "rel_path": note.rel_path,
            "content_hash": note.content_hash,
            "memory_id": memory_id,
            "memory_type": mem_type,
            "synced_at": time.time(),
            "direction": "ingest",
        })

    def _build_links(self, note: VaultNote, all_notes: list[VaultNote]) -> int:
        """Create graph edges from wikilinks."""
        # Find source memory ID
        sync = self.storage.fetch_one("obsidian_sync", note.rel_path, "rel_path")
        if not sync or not sync.get("memory_id"):
            return 0

        source_id = sync["memory_id"]
        title_to_sync: dict[str, dict] = {}
        for n in all_notes:
            s = self.storage.fetch_one("obsidian_sync", n.rel_path, "rel_path")
            if s and s.get("memory_id"):
                title_to_sync[n.title.lower()] = s

        links_added = 0
        for link in note.wikilinks:
            target_sync = title_to_sync.get(link.lower())
            if target_sync:
                self.hippocampus.add_edge(
                    source_id, target_sync["memory_id"],
                    relation="wikilink",
                    weight=1.0,
                )
                links_added += 1

        return links_added

    def _detect_dead_links(self, notes: list[VaultNote], vault: Path) -> None:
        """Find dead wikilinks (link to non-existent notes) and register as gaps."""
        existing_titles = {n.title.lower() for n in notes}

        for note in notes:
            for link in note.wikilinks:
                if link.lower() not in existing_titles:
                    # Dead link = knowledge gap
                    from cortex.gap_filling import Gap, GapType
                    gap = Gap(
                        id=f"gap_deadlink_{hashlib.md5(link.encode()).hexdigest()[:8]}",
                        topic=link,
                        description=f"Dead wikilink [[{link}]] in {note.rel_path}",
                        gap_type=GapType.SEARCHABLE,
                        priority=0.6,
                        related_memory_ids=[],
                    )
                    if self.gap_filling:
                        self.gap_filling._gap_cache[gap.id] = gap

    # ══════════════════════════════════════════════════════════
    #  EXPORT  (CORTEX → Obsidian)
    # ══════════════════════════════════════════════════════════

    def export(self, vault_path: str) -> dict[str, Any]:
        """
        Export CORTEX memories as Obsidian notes.

        Only exports memories not already synced (or marked for export).
        """
        vault = Path(vault_path)
        vault.mkdir(parents=True, exist_ok=True)

        stats = {"notes_created": 0, "notes_updated": 0, "mocs_generated": 0}

        # Export semantic memories
        for fact in self.semantic.get_all():
            if self._should_export(fact.id, fact.source):
                self._export_semantic(fact, vault)
                stats["notes_created"] += 1

        # Export episodic memories as daily note entries
        stats["notes_created"] += self._export_episodic_daily(vault)

        # Export procedural memories
        for skill in self.procedural.get_all():
            if self._should_export(skill.id, getattr(skill, "source", "")):
                self._export_procedural(skill, vault)
                stats["notes_created"] += 1

        # Generate MOCs
        stats["mocs_generated"] = self._generate_mocs(vault)

        # Generate status file
        self._generate_status(vault)

        return stats

    def _should_export(self, memory_id: str, source: str) -> bool:
        """Check if a memory should be exported."""
        if source == "obsidian":
            return False  # Don't re-export what came from vault
        # Check if already exported
        rows = self.storage.fetch_all(
            "obsidian_sync", "memory_id = ? AND direction = 'export'", (memory_id,)
        )
        return len(rows) == 0

    def _export_semantic(self, fact: Any, vault: Path) -> None:
        """Export a semantic fact as an Obsidian note."""
        # Determine subfolder
        if fact.source == "gap_fill" and not self.export_inferred:
            return

        folder = vault / "CORTEX" / "Knowledge"
        if fact.source == "gap_fill":
            folder = vault / "CORTEX" / "Inferred"
        elif fact.source == "consolidation":
            folder = vault / "CORTEX" / "Consolidated"
        folder.mkdir(parents=True, exist_ok=True)

        title = self._sanitize_filename(fact.content[:60])
        path = folder / f"{title}.md"

        tags_str = " ".join(f"#{t}" for t in fact.tags) if fact.tags else ""
        inferred_tag = " #cortex/inferred" if fact.source == "gap_fill" else ""

        content = f"""---
id: {fact.id}
category: {fact.category}
confidence: {fact.confidence}
importance: {fact.importance}
source: {fact.source}
created: {time.strftime('%Y-%m-%d %H:%M', time.localtime(fact.created_at))}
---

# {fact.content[:80]}

{fact.content}

{tags_str}{inferred_tag}
"""
        path.write_text(content, encoding="utf-8")
        self.storage.insert("obsidian_sync", {
            "rel_path": str(path.relative_to(vault)),
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "memory_id": fact.id,
            "memory_type": "semantic",
            "synced_at": time.time(),
            "direction": "export",
        })

    def _export_episodic_daily(self, vault: Path) -> int:
        """Group episodic memories by date and append to daily notes."""
        episodes = self.episodic.get_all()
        by_date: dict[str, list] = defaultdict(list)
        exported = 0

        for ep in episodes:
            if not self._should_export(ep.id, ep.source):
                continue
            date_str = time.strftime("%Y-%m-%d", time.localtime(ep.created_at))
            by_date[date_str].append(ep)

        for date_str, eps in by_date.items():
            folder = vault / "memory"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{date_str}.md"

            existing = ""
            if path.exists():
                existing = path.read_text(encoding="utf-8")

            new_entries = []
            for ep in eps:
                entry = f"\n## {time.strftime('%H:%M', time.localtime(ep.created_at))}\n\n{ep.content}\n"
                if entry.strip() not in existing:
                    new_entries.append(entry)
                    self.storage.insert("obsidian_sync", {
                        "rel_path": str(path.relative_to(vault)),
                        "content_hash": hashlib.md5(ep.content.encode()).hexdigest(),
                        "memory_id": ep.id,
                        "memory_type": "episodic",
                        "synced_at": time.time(),
                        "direction": "export",
                    })
                    exported += 1

            if new_entries:
                with open(path, "a", encoding="utf-8") as f:
                    f.writelines(new_entries)

        return exported

    def _export_procedural(self, skill: Any, vault: Path) -> None:
        """Export a procedural skill as a how-to note."""
        folder = vault / "CORTEX" / "Skills"
        folder.mkdir(parents=True, exist_ok=True)

        title = self._sanitize_filename(skill.content[:60])
        path = folder / f"{title}.md"

        content = f"""---
id: {skill.id}
trigger: {skill.trigger}
pattern: {skill.pattern}
success_rate: {skill.success_rate:.0%}
importance: {skill.importance}
---

# {skill.content[:80]}

{skill.content}

**Trigger:** {skill.trigger or 'N/A'}
**Success Rate:** {skill.success_rate:.0%} ({skill.success_count}✓ / {skill.failure_count}✗)
"""
        path.write_text(content, encoding="utf-8")
        self.storage.insert("obsidian_sync", {
            "rel_path": str(path.relative_to(vault)),
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "memory_id": skill.id,
            "memory_type": "procedural",
            "synced_at": time.time(),
            "direction": "export",
        })

    def _generate_mocs(self, vault: Path) -> int:
        """Generate Maps of Content from memory clusters."""
        categories = self.semantic.get_categories()
        moc_folder = vault / "CORTEX" / "MOCs"
        moc_folder.mkdir(parents=True, exist_ok=True)
        count = 0

        for cat in categories:
            facts = self.semantic.recall(cat, top_k=50)
            if len(facts) < 2:
                continue

            title = self._sanitize_filename(f"MOC - {cat.title()}")
            path = moc_folder / f"{title}.md"

            lines = [
                f"---\ntype: moc\ncategory: {cat}\nupdated: {time.strftime('%Y-%m-%d')}\n---\n",
                f"# Map of Content: {cat.title()}\n\n",
                f"*Auto-generated by CORTEX — {len(facts)} memories*\n\n",
            ]

            for fact in facts:
                note_title = self._sanitize_filename(fact.content[:60])
                lines.append(f"- [[{note_title}]] (confidence: {fact.confidence:.0%})\n")

            path.write_text("".join(lines), encoding="utf-8")
            count += 1

        return count

    def _generate_status(self, vault: Path) -> None:
        """Generate CORTEX-STATUS.md with memory stats."""
        ep_count = self.episodic.count()
        sem_count = self.semantic.count()
        proc_count = self.procedural.count()
        total = ep_count + sem_count + proc_count

        synced = self.storage.count("obsidian_sync")

        lines = [
            f"---\nupdated: {time.strftime('%Y-%m-%d %H:%M')}\n---\n",
            "# 🧠 CORTEX Status\n\n",
            "## Memory Stats\n\n",
            f"| Type | Count |\n|------|-------|\n",
            f"| Episodic | {ep_count} |\n",
            f"| Semantic | {sem_count} |\n",
            f"| Procedural | {proc_count} |\n",
            f"| **Total** | **{total}** |\n\n",
            f"## Vault Sync\n\n",
            f"- Synced notes: {synced}\n",
            f"- Last sync: {time.strftime('%Y-%m-%d %H:%M')}\n\n",
        ]

        # Gap report if available
        if self.gap_filling:
            report = self.gap_filling.gap_report()
            lines.extend([
                "## Knowledge Gaps\n\n",
                f"- Total gaps: {report.total_gaps}\n",
                f"- Filled: {report.filled_gaps}\n",
                f"- Fill accuracy: {report.fill_accuracy:.0%}\n\n",
            ])
            if report.recommendations:
                lines.append("### Recommendations\n\n")
                for rec in report.recommendations:
                    lines.append(f"- {rec}\n")
                lines.append("\n")

        (vault / "CORTEX-STATUS.md").write_text("".join(lines), encoding="utf-8")

    # ══════════════════════════════════════════════════════════
    #  SYNC  (Bidirectional)
    # ══════════════════════════════════════════════════════════

    def sync(self, vault_path: str, force: bool = False) -> dict[str, Any]:
        """Bidirectional sync: ingest then export."""
        ingest_stats = self.ingest(vault_path, force=force)
        export_stats = self.export(vault_path)
        return {"ingest": ingest_stats, "export": export_stats}

    # ══════════════════════════════════════════════════════════
    #  WATCH  (File system monitoring)
    # ══════════════════════════════════════════════════════════

    def watch(self, vault_path: str, callback: Any = None) -> None:
        """
        Watch vault for changes. This is a blocking call.

        In production, run in a thread. Falls back to polling if inotify
        is not available.
        """
        vault = Path(vault_path)
        if not vault.is_dir():
            raise FileNotFoundError(f"Vault not found: {vault_path}")

        # Simple poll-based watcher (portable)
        known_hashes: dict[str, str] = {}
        for md in vault.rglob("*.md"):
            if not any(p.startswith(".") for p in md.parts):
                text = md.read_text(encoding="utf-8", errors="replace")
                known_hashes[str(md)] = hashlib.md5(text.encode()).hexdigest()

        while True:
            time.sleep(5)
            for md in vault.rglob("*.md"):
                path_str = str(md)
                if any(p.startswith(".") for p in md.parts):
                    continue
                try:
                    text = md.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                new_hash = hashlib.md5(text.encode()).hexdigest()
                old_hash = known_hashes.get(path_str)
                if old_hash != new_hash:
                    known_hashes[path_str] = new_hash
                    # Re-import changed note
                    note = self._parse_note(md, vault)
                    self._import_note(note, force=True)
                    if callback:
                        callback(note)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """Make text safe for use as a filename."""
        safe = re.sub(r'[<>:"/\\|?*]', '', text)
        safe = safe.strip(". ")
        return safe[:80] if safe else "untitled"
