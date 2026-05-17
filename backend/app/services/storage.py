"""SQLite-backed persistence for chat memory and document workspaces.

Async wrappers around the original synchronous SQLite functions.
SQLite itself is synchronous, so we use ``asyncio.to_thread`` to avoid
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend.app.config import get_settings


def _db_path() -> str:
    return get_settings().SQLITE_DB_PATH


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or _db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> None:
    path = db_path or _db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                user_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL DEFAULT 'default',
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_documents (
                user_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                vector_index_path TEXT NOT NULL,
                ts_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY (user_id, workspace_id, doc_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY (workspace_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY (session_id)
            )
            """
        )
        # Migrations
        try:
            conn.execute("SELECT workspace_id FROM chat_messages LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE chat_messages ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages ON chat_messages(user_id, workspace_id, session_id, ts_utc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_docs ON user_documents(user_id, workspace_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workspaces ON workspaces(user_id, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions ON chat_sessions(user_id, workspace_id, updated_at)"
        )


# ---------------------------------------------------------------------------
# Synchronous helpers (called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _load_chat_messages_sync(
    user_id: str, workspace_id: str, session_id: str, limit: int = 50
) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, ts_utc
            FROM chat_messages
            WHERE user_id = ? AND workspace_id = ? AND session_id = ?
            ORDER BY ts_utc DESC
            LIMIT ?
            """,
            (user_id, workspace_id, session_id, limit),
        ).fetchall()
    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["content"], "timestamp": r["ts_utc"]} for r in rows]


def _append_chat_messages_sync(
    user_id: str,
    workspace_id: str,
    session_id: str,
    messages: List[Dict[str, Any]],
) -> None:
    if not messages:
        return
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, user_id, workspace_id, name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
            """,
            (session_id, user_id, workspace_id, "New Chat"),
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        if count == 0:
            first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if first_user_msg:
                name = first_user_msg.get("content", "")[:30] + "..."
                conn.execute(
                    "UPDATE chat_sessions SET name = ? WHERE session_id = ?",
                    (name, session_id),
                )
        conn.executemany(
            """
            INSERT INTO chat_messages(user_id, workspace_id, session_id, role, content, ts_utc)
            VALUES(?, ?, ?, ?, ?, COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')))
            """,
            [
                (
                    user_id,
                    workspace_id,
                    session_id,
                    m.get("role", "unknown"),
                    m.get("content", ""),
                    m.get("timestamp"),
                )
                for m in messages
                if m.get("content")
            ],
        )


def _upsert_document_sync(
    user_id: str,
    workspace_id: str,
    doc_id: str,
    file_path: str,
    vector_index_path: str,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_documents(user_id, workspace_id, doc_id, file_path, vector_index_path)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(user_id, workspace_id, doc_id) DO UPDATE SET
              file_path=excluded.file_path,
              vector_index_path=excluded.vector_index_path
            """,
            (user_id, workspace_id, doc_id, file_path, vector_index_path),
        )


def _list_workspace_documents_sync(
    user_id: str, workspace_id: str
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT doc_id, file_path, vector_index_path, ts_utc
            FROM user_documents
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY ts_utc ASC
            """,
            (user_id, workspace_id),
        ).fetchall()
    docs = [
        {
            "doc_id": r["doc_id"],
            "file_path": r["file_path"],
            "vector_index_path": r["vector_index_path"],
            "timestamp": r["ts_utc"],
        }
        for r in rows
    ]
    index_path = docs[-1]["vector_index_path"] if docs else None
    return docs, index_path


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

async def load_chat_messages(
    user_id: str, workspace_id: str, session_id: str, limit: int = 50
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        _load_chat_messages_sync, user_id, workspace_id, session_id, limit
    )


async def append_chat_messages(
    user_id: str,
    workspace_id: str,
    session_id: str,
    messages: List[Dict[str, Any]],
) -> None:
    await asyncio.to_thread(
        _append_chat_messages_sync, user_id, workspace_id, session_id, messages
    )


async def upsert_document(
    user_id: str,
    workspace_id: str,
    doc_id: str,
    file_path: str,
    vector_index_path: str,
) -> None:
    await asyncio.to_thread(
        _upsert_document_sync, user_id, workspace_id, doc_id, file_path, vector_index_path
    )


async def list_workspace_documents(
    user_id: str, workspace_id: str
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    return await asyncio.to_thread(
        _list_workspace_documents_sync, user_id, workspace_id
    )
