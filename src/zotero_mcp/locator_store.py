"""
SQLite locator store for chunk-to-source mapping.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


class LocatorStore:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path.home() / ".config" / "zotero-mcp" / "locator.db")
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _write_conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._write_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_locator (
                    chunk_id TEXT PRIMARY KEY,
                    item_key TEXT NOT NULL,
                    attachment_key TEXT NOT NULL,
                    md_store_path TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    section_path TEXT,
                    md_hash TEXT,
                    pdf_hash TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_locator_item ON chunk_locator(item_key)"
            )

    def upsert_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self._write_conn() as conn:
            conn.executemany(
                """
                INSERT INTO chunk_locator (
                    chunk_id, item_key, attachment_key, md_store_path,
                    char_start, char_end, page_start, page_end, section_path, md_hash, pdf_hash
                ) VALUES (
                    :chunk_id, :item_key, :attachment_key, :md_store_path,
                    :char_start, :char_end, :page_start, :page_end, :section_path, :md_hash, :pdf_hash
                )
                ON CONFLICT(chunk_id) DO UPDATE SET
                    item_key=excluded.item_key,
                    attachment_key=excluded.attachment_key,
                    md_store_path=excluded.md_store_path,
                    char_start=excluded.char_start,
                    char_end=excluded.char_end,
                    page_start=excluded.page_start,
                    page_end=excluded.page_end,
                    section_path=excluded.section_path,
                    md_hash=excluded.md_hash,
                    pdf_hash=excluded.pdf_hash,
                    updated_at=CURRENT_TIMESTAMP
                """,
                records,
            )

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM chunk_locator WHERE chunk_id = ?",
                (chunk_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            cur = conn.execute("SELECT COUNT(*) FROM chunk_locator")
            return int(cur.fetchone()[0])
        finally:
            conn.close()

    def delete_item(self, item_key: str) -> int:
        with self._write_conn() as conn:
            cur = conn.execute("DELETE FROM chunk_locator WHERE item_key = ?", (item_key,))
            return cur.rowcount

    def reset(self) -> None:
        with self._write_conn() as conn:
            conn.execute("DELETE FROM chunk_locator")

    def close(self) -> None:
        """No-op close.

        LocatorStore opens and closes SQLite connections per operation,
        so there is no long-lived connection to close here. This method
        exists for API compatibility and to support use as a context
        manager via __enter__/__exit__.
        """
        pass

    def __enter__(self) -> "LocatorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
