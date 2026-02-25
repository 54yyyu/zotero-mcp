"""
SQLite locator store for chunk-to-source mapping.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class LocatorStore:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path.home() / ".config" / "zotero-mcp" / "locator.db")
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # Use check_same_thread=False to allow access from asyncio worker threads
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
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
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_locator_item ON chunk_locator(item_key)"
        )
        self.conn.commit()

    def upsert_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.conn.executemany(
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
        self.conn.commit()

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        cur = self.conn.execute(
            "SELECT * FROM chunk_locator WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM chunk_locator")
        return int(cur.fetchone()[0])

    def delete_item(self, item_key: str) -> int:
        cur = self.conn.execute("DELETE FROM chunk_locator WHERE item_key = ?", (item_key,))
        self.conn.commit()
        return cur.rowcount

    def reset(self) -> None:
        self.conn.execute("DELETE FROM chunk_locator")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LocatorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
