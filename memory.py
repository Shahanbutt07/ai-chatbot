import sqlite3
import os
import logging
from contextlib import contextmanager
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "chat_history.db")
MAX_TITLE_LENGTH = 60

# ── Connection manager ─────────────────────────────────────────────────────
@contextmanager
def get_db():
    """Context manager — always closes connection even if error occurs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name not index
    conn.execute("PRAGMA foreign_keys = ON")  # enforce foreign key rules
    conn.execute("PRAGMA journal_mode = WAL")  # better concurrent access
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        conn.close()

# ── Setup ──────────────────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New Chat',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                system_prompt TEXT NOT NULL DEFAULT 'You are a helpful assistant.'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL CHECK(role IN ('human', 'ai')),
                content         TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                ON messages(conversation_id);
        """)
    logger.info("Database initialized.")

# ── Conversations ──────────────────────────────────────────────────────────
def create_conversation(
    conv_id: str,
    title: str = "New Chat",
    system_prompt: str = "You are a helpful assistant."
):
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, system_prompt) VALUES (?, ?, ?, ?, ?)",
            (conv_id, title[:MAX_TITLE_LENGTH], now, now, system_prompt)
        )
    logger.info(f"Created conversation: {conv_id}")

def get_all_conversations() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, system_prompt FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]

def delete_conversation(conv_id: str):
    """Delete conversation — messages auto-deleted via CASCADE."""
    with get_db() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    logger.info(f"Deleted conversation: {conv_id}")

def rename_conversation(conv_id: str, new_title: str):
    if not new_title or not new_title.strip():
        return
    with get_db() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (new_title.strip()[:MAX_TITLE_LENGTH], datetime.now().isoformat(), conv_id)
        )

def get_conversation_info(conv_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at, system_prompt FROM conversations WHERE id = ?",
            (conv_id,)
        ).fetchone()
    return dict(row) if row else None

def update_system_prompt(conv_id: str, system_prompt: str):
    """Update system prompt for a conversation."""
    with get_db() as conn:
        conn.execute(
            "UPDATE conversations SET system_prompt = ?, updated_at = ? WHERE id = ?",
            (system_prompt, datetime.now().isoformat(), conv_id)
        )

# ── Messages ───────────────────────────────────────────────────────────────
def save_message(conv_id: str, role: str, content: str):
    if role not in ("human", "ai"):
        raise ValueError(f"Invalid role: {role}. Must be 'human' or 'ai'.")
    if not content or not content.strip():
        return
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (conv_id, role, content.strip(), now)
        )
        # Update conversation's updated_at so it floats to top of sidebar
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id)
        )

def load_messages(conv_id: str, limit: int = 100) -> list:
    """Load last N messages for a conversation."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT role, content FROM messages
               WHERE conversation_id = ?
               ORDER BY id ASC
               LIMIT ?""",
            (conv_id, limit)
        ).fetchall()
    messages = []
    for row in rows:
        if row["role"] == "human":
            messages.append(HumanMessage(content=row["content"]))
        else:
            messages.append(AIMessage(content=row["content"]))
    return messages

def get_message_count(conv_id: str) -> int:
    """Get total number of messages in a conversation."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?",
            (conv_id,)
        ).fetchone()
    return row["count"] if row else 0

# ── Auto title ─────────────────────────────────────────────────────────────
def auto_title_conversation(conv_id: str, first_message: str) -> str:
    """Set conversation title from first message."""
    first_message = first_message.strip()
    title = first_message[:MAX_TITLE_LENGTH]
    if len(first_message) > MAX_TITLE_LENGTH:
        # Cut at last word boundary instead of mid-word
        title = title[:title.rfind(" ")] + "..." if " " in title else title + "..."
    rename_conversation(conv_id, title)
    return title