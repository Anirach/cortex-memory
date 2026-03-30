"""
SQLite storage backend for CORTEX.

All memory types share one database file. Each subsystem gets its own table(s).
Thread-safe via check_same_thread=False + connection-per-call pattern.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


class Storage:
    """SQLite-backed persistent storage for all CORTEX subsystems."""

    SCHEMA_VERSION = 1

    DDL = """
    -- Working memory (ephemeral ring buffer — persisted for crash recovery)
    CREATE TABLE IF NOT EXISTS working_memory (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        content     TEXT NOT NULL,
        metadata    TEXT DEFAULT '{}',
        created_at  REAL NOT NULL,
        slot        INTEGER NOT NULL
    );

    -- Episodic memory
    CREATE TABLE IF NOT EXISTS episodic_memory (
        id            TEXT PRIMARY KEY,
        content       TEXT NOT NULL,
        embedding     BLOB,
        metadata      TEXT DEFAULT '{}',
        created_at    REAL NOT NULL,
        last_accessed REAL NOT NULL,
        access_count  INTEGER DEFAULT 1,
        importance    REAL DEFAULT 0.5,
        decay_factor  REAL DEFAULT 1.0,
        tags          TEXT DEFAULT '[]',
        source        TEXT DEFAULT 'user'
    );

    -- Semantic memory
    CREATE TABLE IF NOT EXISTS semantic_memory (
        id            TEXT PRIMARY KEY,
        content       TEXT NOT NULL,
        embedding     BLOB,
        metadata      TEXT DEFAULT '{}',
        created_at    REAL NOT NULL,
        last_accessed REAL NOT NULL,
        access_count  INTEGER DEFAULT 1,
        importance    REAL DEFAULT 0.5,
        category      TEXT DEFAULT 'general',
        confidence    REAL DEFAULT 1.0,
        source        TEXT DEFAULT 'user',
        tags          TEXT DEFAULT '[]'
    );

    -- Procedural memory
    CREATE TABLE IF NOT EXISTS procedural_memory (
        id             TEXT PRIMARY KEY,
        content        TEXT NOT NULL,
        embedding      BLOB,
        metadata       TEXT DEFAULT '{}',
        created_at     REAL NOT NULL,
        last_accessed  REAL NOT NULL,
        access_count   INTEGER DEFAULT 1,
        importance     REAL DEFAULT 0.5,
        success_count  INTEGER DEFAULT 0,
        failure_count  INTEGER DEFAULT 0,
        pattern        TEXT DEFAULT '',
        trigger        TEXT DEFAULT '',
        tags           TEXT DEFAULT '[]'
    );

    -- Relationships (graph edges between memories)
    CREATE TABLE IF NOT EXISTS memory_edges (
        source_id   TEXT NOT NULL,
        target_id   TEXT NOT NULL,
        relation    TEXT NOT NULL,
        weight      REAL DEFAULT 1.0,
        created_at  REAL NOT NULL,
        PRIMARY KEY (source_id, target_id, relation)
    );

    -- Error log (self-improvement)
    CREATE TABLE IF NOT EXISTS error_log (
        id          TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        context     TEXT DEFAULT '',
        created_at  REAL NOT NULL,
        resolved    INTEGER DEFAULT 0,
        correction  TEXT DEFAULT '',
        category    TEXT DEFAULT 'unknown',
        severity    REAL DEFAULT 0.5
    );

    -- Correction patterns
    CREATE TABLE IF NOT EXISTS corrections (
        id          TEXT PRIMARY KEY,
        error_id    TEXT,
        wrong       TEXT NOT NULL,
        correct     TEXT NOT NULL,
        pattern     TEXT DEFAULT '',
        created_at  REAL NOT NULL,
        times_applied INTEGER DEFAULT 0
    );

    -- Evolution strategies
    CREATE TABLE IF NOT EXISTS strategies (
        id              TEXT PRIMARY KEY,
        generation      INTEGER NOT NULL,
        params          TEXT NOT NULL,
        fitness         REAL DEFAULT 0.0,
        evaluations     INTEGER DEFAULT 0,
        created_at      REAL NOT NULL,
        is_active       INTEGER DEFAULT 1
    );

    -- Evolution history
    CREATE TABLE IF NOT EXISTS evolution_history (
        generation      INTEGER PRIMARY KEY,
        best_fitness    REAL,
        avg_fitness     REAL,
        best_params     TEXT,
        population_size INTEGER,
        timestamp       REAL
    );

    -- Retrieval feedback (for evolution fitness)
    CREATE TABLE IF NOT EXISTS retrieval_feedback (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        query       TEXT NOT NULL,
        result_id   TEXT NOT NULL,
        strategy_id TEXT,
        useful      INTEGER DEFAULT 0,
        timestamp   REAL NOT NULL
    );

    -- Meta-cognition assessments
    CREATE TABLE IF NOT EXISTS meta_assessments (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        topic       TEXT NOT NULL,
        confidence  REAL NOT NULL,
        coverage    REAL NOT NULL,
        memory_count INTEGER,
        timestamp   REAL NOT NULL
    );

    -- Schema version
    CREATE TABLE IF NOT EXISTS schema_info (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self._get_conn()
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.fetchall()

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        conn = self._get_conn()
        conn.executemany(sql, params_list)
        conn.commit()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(self.DDL)
        conn.execute(
            "INSERT OR IGNORE INTO schema_info (key, value) VALUES (?, ?)",
            ("version", str(self.SCHEMA_VERSION)),
        )
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Convenience helpers ──────────────────────────────────────

    def insert(self, table: str, data: dict[str, Any]) -> None:
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
        self.execute(sql, tuple(data.values()))

    def fetch_one(self, table: str, id_val: str, id_col: str = "id") -> dict | None:
        rows = self.execute(
            f"SELECT * FROM {table} WHERE {id_col} = ?", (id_val,)
        )
        return dict(rows[0]) if rows else None

    def fetch_all(self, table: str, where: str = "", params: tuple = ()) -> list[dict]:
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return [dict(r) for r in self.execute(sql, params)]

    def delete(self, table: str, id_val: str, id_col: str = "id") -> None:
        self.execute(f"DELETE FROM {table} WHERE {id_col} = ?", (id_val,))

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        sql = f"SELECT COUNT(*) as cnt FROM {table}"
        if where:
            sql += f" WHERE {where}"
        rows = self.execute(sql, params)
        return rows[0]["cnt"] if rows else 0
