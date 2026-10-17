import sqlite3
from datetime import UTC, datetime

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
)
"""


def connect() -> sqlite3.Connection:
    path = settings.data_dir / "research.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(documents)")]
    if "content_hash" in columns:
        conn.execute("ALTER TABLE documents RENAME TO documents_old")
        conn.execute(SCHEMA)
        conn.execute(
            """
            INSERT INTO documents (id, filename, status, error, created_at)
            SELECT id, filename, status, error, created_at FROM documents_old
            """
        )
        conn.execute("DROP TABLE documents_old")
        conn.commit()
    return conn


def insert_document(doc_id: str, filename: str) -> None:
    conn = connect()
    try:
        conn.execute(
            """
            INSERT INTO documents (id, filename, status, error, created_at)
            VALUES (?, ?, 'processing', NULL, ?)
            """,
            (doc_id, filename, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def mark_ready(doc_id: str) -> None:
    _set_status(doc_id, "ready", None)


def mark_failed(doc_id: str, error: str) -> None:
    _set_status(doc_id, "failed", error)


def _set_status(doc_id: str, status: str, error: str | None) -> None:
    conn = connect()
    try:
        conn.execute(
            "UPDATE documents SET status = ?, error = ? WHERE id = ?",
            (status, error, doc_id),
        )
        conn.commit()
    finally:
        conn.close()
