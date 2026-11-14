"""Upgrade-path details of the item-keyed index (#826, final review I1/I2).

I1: new-shape writes send ``None`` values, which chromadb 1.0.0 and 1.0.1 reject.
I2: refreshing an unchanged file's stamp sends the whole stored metadata, so a
chromadb that replaces rather than merges does not wipe ``item_key``/``group_id``.
"""

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search

ROOT = Path(__file__).resolve().parent.parent


def test_the_chromadb_floor_accepts_none_metadata_values():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    spec = re.search(r'"chromadb>=(\d+)\.(\d+)\.(\d+)"', pyproject)

    assert spec, "chromadb has a lower bound"
    assert tuple(int(part) for part in spec.groups()) >= (1, 0, 2)


class RecordingChroma:
    """Stored metadata by id; ``update_metadatas`` keeps what it was given, as a replacing chromadb would."""

    embedding_max_tokens = 8000

    def __init__(self, stored):
        self.stored = stored
        self.updates: list[tuple[str, dict]] = []

    def get_metadatas(self, ids):
        return {doc_id: dict(self.stored[doc_id]) for doc_id in ids if doc_id in self.stored}

    def update_metadatas(self, ids, metadatas):
        for doc_id, meta in zip(ids, metadatas):
            self.updates.append((doc_id, meta))


def test_refreshing_a_stamp_sends_the_whole_stored_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: None)
    stored = {
        "item_key": "PDFAAAAA",
        "group_id": 7,
        "date_modified": "2024-01-01 00:00:00",
        "file_stamp": "1:100",
        "file_hash": "ab" * 32,
    }
    chroma = RecordingChroma({"PDFAAAAA": stored})
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=str(tmp_path / "config.json"))
    unit = semantic_search._attachment_unit(
        "PDFAAAAA", "", file_stamp="2:100", date_modified="2024-02-02 00:00:00", group_id=7
    )
    stats = {"refreshed_unchanged": 0, "processed_items": 0, "updated_items": 0}

    left = search._refresh_units([unit], stats)

    assert left == []
    assert chroma.updates == [("PDFAAAAA", {**stored, "file_stamp": "2:100", "date_modified": "2024-02-02 00:00:00"})]


def test_a_stamp_that_cannot_be_refreshed_is_left_for_a_normal_write(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: None)
    chroma = RecordingChroma({})  # the record vanished
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=str(tmp_path / "config.json"))
    unit = semantic_search._attachment_unit("PDFAAAAA", "", file_stamp="2:100", group_id=7)
    stats = {"refreshed_unchanged": 0, "processed_items": 0, "updated_items": 0}

    assert search._refresh_units([unit], stats) == [unit]
    assert chroma.updates == []
