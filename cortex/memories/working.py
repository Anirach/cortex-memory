"""
Working Memory — fast, limited-capacity ring buffer.

Holds the most recent N items. Oldest items are automatically evicted.
Think of it as the agent's "current context" or "scratch pad".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingItem:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    slot: int = 0


class WorkingMemory:
    """Fixed-capacity ring buffer for immediate context."""

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self._buffer: list[WorkingItem] = []
        self._next_slot = 0

    def push(self, content: str, metadata: dict[str, Any] | None = None) -> WorkingItem:
        """Add an item. Evicts oldest if at capacity."""
        item = WorkingItem(
            content=content,
            metadata=metadata or {},
            created_at=time.time(),
            slot=self._next_slot,
        )
        self._next_slot += 1

        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(item)
        return item

    def get_all(self) -> list[WorkingItem]:
        """Return all items in order (oldest first)."""
        return list(self._buffer)

    def get_recent(self, n: int = 5) -> list[WorkingItem]:
        """Return the N most recent items."""
        return self._buffer[-n:]

    def search(self, query: str) -> list[WorkingItem]:
        """Simple substring search across working memory."""
        q = query.lower()
        return [item for item in self._buffer if q in item.content.lower()]

    def clear(self) -> list[WorkingItem]:
        """Flush working memory, returning evicted items for consolidation."""
        evicted = list(self._buffer)
        self._buffer.clear()
        return evicted

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemory(capacity={self.capacity}, used={len(self._buffer)})"
