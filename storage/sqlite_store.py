"""SQLite-backed persistence for chat memory and document workspaces.

No globals: callers create a connection per node invocation.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DB_PATH = "memory.db"


def _connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with _connect(db_path) as conn:
        # Chat Messages
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
        
        # User Documents
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
        
        # Workspaces Table
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

        # Chat Sessions Table
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

        # Migrations (idempotent)
        # Check if chat_messages has workspace_id
        try:
            conn.execute("SELECT workspace_id FROM chat_messages LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'")

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


# ── Chat Session Management ──

def create_chat_session(user_id: str, workspace_id: str, session_id: str = None, name: str = "New Chat", db_path: str = DEFAULT_DB_PATH) -> str:
    """Create a new chat session."""
    import uuid
    init_db(db_path)
    if not session_id:
        session_id = str(uuid.uuid4())
    
    with _connect(db_path) as conn:
        conn.execute(
             """
             INSERT INTO chat_sessions (session_id, user_id, workspace_id, name)
             VALUES (?, ?, ?, ?)
             ON CONFLICT(session_id) DO NOTHING
             """,
             (session_id, user_id, workspace_id, name)
        )
    return session_id

def list_chat_sessions(user_id: str, workspace_id: str, limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """List chat sessions for a workspace."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT session_id, name, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, workspace_id, limit)
        ).fetchall()
    return [{"session_id": r["session_id"], "name": r["name"], "updated_at": r["updated_at"]} for r in rows]

def update_chat_session_name(session_id: str, new_name: str, db_path: str = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE chat_sessions SET name = ? WHERE session_id = ?",
            (new_name, session_id)
        )

def delete_chat_session(session_id: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """Hard delete a chat session and its messages."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))


# ── Workspace Management ──

def create_workspace(user_id: str, name: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """Create a new workspace and return its ID."""
    import uuid
    init_db(db_path)
    workspace_id = str(uuid.uuid4())[:8]  # Short ID for readability
    with _connect(db_path) as conn:
        conn.execute(
             "INSERT INTO workspaces (workspace_id, user_id, name) VALUES (?, ?, ?)",
             (workspace_id, user_id, name)
        )
    return workspace_id

def list_workspaces(user_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """List all workspaces for a user."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT workspace_id, name, created_at FROM workspaces WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    
    # Ensure a default workspace exists if none found
    if not rows:
        default_id = create_workspace(user_id, "Default Workspace", db_path)
        return [{"workspace_id": default_id, "name": "Default Workspace", "created_at": ""}]
        
    return [{"workspace_id": r["workspace_id"], "name": r["name"], "created_at": r["created_at"]} for r in rows]

def rename_workspace(workspace_id: str, new_name: str, db_path: str = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE workspaces SET name = ? WHERE workspace_id = ?",
            (new_name, workspace_id)
        )

# ── Chat & Docs ──

def load_chat_messages(
    user_id: str,
    workspace_id: str,
    session_id: str,  # Ideally thread_id, but keeping session_id for now
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    init_db(db_path)
    with _connect(db_path) as conn:
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

def append_chat_messages(
    user_id: str,
    workspace_id: str,
    session_id: str,
    messages: List[Dict[str, Any]],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    if not messages:
        return
    init_db(db_path)
    
    # Auto-create session if it doesn't exist (e.g. first message)
    # Also update updated_at timestamp
    import uuid
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, user_id, workspace_id, name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
            """,
            (session_id, user_id, workspace_id, "New Chat")
        )
        
        # If it's the very first user message, rename the session based on content
        # Check if this session has any messages yet
        count = conn.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        
        if count == 0:
             # Find first user message
             first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
             if first_user_msg:
                 # Truncate to ~30 chars
                 name = first_user_msg.get("content", "")[:30] + "..."
                 conn.execute("UPDATE chat_sessions SET name = ? WHERE session_id = ?", (name, session_id))

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

def upsert_document(
    user_id: str,
    workspace_id: str,
    doc_id: str,
    file_path: str,
    vector_index_path: str,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
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

def list_workspace_documents(
    user_id: str,
    workspace_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (docs, vector_index_path)."""
    init_db(db_path)
    with _connect(db_path) as conn:
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

