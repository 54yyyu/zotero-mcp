"""Gap fills that wait for full text, and what a run must not lose or re-pay (#769).

An item whose other level was built from its full text is "waiting" when this
run has no text for it. These tests pin when such an item is checked again,
that a fetch error never parks it, that runs which cannot judge the gaps keep
the waiting list, and that a full scan of a split index embeds only changes.

Real ChromaDB in a temp directory, a fake Zotero, an embedding function that
records its inputs. HOME and Path.home point at tmp_path.
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from chromadb import EmbeddingFunction  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
from zotero_mcp.embeddings.base import EmbeddingAPIUnreachable  # noqa: E402

NAME = "gap_waiting"
LONG = " ".join(f"Sentence number {i} of the abstract." for i in range(100))
TEXT = " ".join(f"Body sentence {i} of the paper." for i in range(100))


class RecordingEF(EmbeddingFunction):
    def __init__(self):
        self.embedded: list[str] = []
        self.unreachable = False

    def __call__(self, input):
        if self.unreachable:
            raise EmbeddingAPIUnreachable("openai", "api.example.invalid", "connection refused")
        self.embedded.extend(input)
        return [[1.0, 0.0] for _ in input]

    def truncate(self, text, max_tokens):
        return text


class RecordingClient(ChromaClient):
    def _create_embedding_function(self):
        return RecordingEF()

    @property
    def embedded(self):
        return self.embedding_function.embedded


class FakeZotero:
    """One library. ``texts`` is what Zotero has extracted per item's PDF."""

    def __init__(self, abstracts):
        self.abstracts = dict(abstracts)
        self.texts: dict[str, str] = {}
        self.version = 5
        self.changed: set[str] = set()
        self.fetched: list[str] = []
        self.text_requests: list[str] = []
        self.fulltext_error: Exception | None = None

    def item(self, key):
        self.fetched.append(key)
        return {
            "key": key,
            "version": self.version,
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Title {key}",
                "abstractNote": self.abstracts[key],
                "creators": [],
                "dateAdded": "2024-01-01T00:00:00Z",
                "dateModified": "2024-01-01T00:00:00Z",
            },
        }

    def items(self, start=0, limit=100, **kwargs):
        if kwargs.get("itemType") == "attachment":
            return []
        return [] if start else [self.item(k) for k in self.abstracts]

    def item_versions(self, since=None, **kwargs):
        keys = self.changed if since is not None else self.abstracts
        return {k: self.version for k in keys}

    def children(self, key, *a, **kw):
        if kw.get("start"):
            return []
        return [{"key": f"{key}PDF", "data": {"itemType": "attachment", "contentType": "application/pdf"}}]

    def fulltext_item(self, attachment_key):
        parent = attachment_key[: -len("PDF")]
        self.text_requests.append(parent)
        if self.fulltext_error is not None:
            raise self.fulltext_error
        return {"content": self.texts.get(parent, "")}

    def last_modified_version(self, **kwargs):
        return self.version


@pytest.fixture()
def build(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    config_path = tmp_path / "config.json"

    def _build(zotero, *, chunking, fulltext=False):
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
        existing = json.loads(config_path.read_text()) if config_path.exists() else {"semantic_search": {}}
        existing["semantic_search"].update(
            include_fulltext=fulltext,
            index_schema_version=semantic_search._INDEX_SCHEMA_VERSION,
            chunking={"enabled": chunking, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
            openai_batch={"enabled": False},
        )
        config_path.write_text(json.dumps(existing))
        client = RecordingClient(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
        client.route_passages = True
        return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))

    return _build


def _saved(tmp_path) -> dict:
    return json.loads((tmp_path / "config.json").read_text())["semantic_search"]["update_config"]


def _passage_ids(search):
    return set(search.chroma_client.client.get_collection(f"{NAME}_passages").get(include=[])["ids"])


def _park(build, zotero):
    """Index AAA as an article built from full text, then let a run without text park its passages."""
    search = build(zotero, chunking=False)
    search.update_database()
    collection = search.chroma_client.collection
    stored = collection.get(ids=["AAA"], include=["metadatas"])["metadatas"][0]
    collection.update(ids=["AAA"], metadatas=[{**stored, "has_fulltext": True}])
    build(zotero, chunking=True).update_database()


def _age_check(tmp_path, key, days):
    """Move the stored time of ``key``'s last text check ``days`` into the past."""
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    checks = data["semantic_search"]["update_config"]["gaps_text_checked"]
    checks[key]["at"] = (datetime.now() - timedelta(days=days)).isoformat()
    config_path.write_text(json.dumps(data))


# --- fetch errors ------------------------------------------------------------------


def test_a_fetch_error_does_not_park_an_item(build, tmp_path):
    """The text may well be there; the next run should look again."""
    zotero = FakeZotero({"AAA": LONG})
    search = build(zotero, chunking=False)
    search.update_database()
    collection = search.chroma_client.collection
    stored = collection.get(ids=["AAA"], include=["metadatas"])["metadatas"][0]
    collection.update(ids=["AAA"], metadatas=[{**stored, "has_fulltext": True}])

    zotero.texts["AAA"] = TEXT
    zotero.fulltext_error = ConnectionError("connection reset")
    search = build(zotero, chunking=True, fulltext=True)
    search.update_database()

    assert search.chroma_client.embedded == [], "no passages from metadata"
    assert "AAA" not in (_saved(tmp_path).get("gaps_waiting_for_text") or [])

    zotero.fulltext_error = None
    search = build(zotero, chunking=True, fulltext=True)
    search.update_database()

    assert len(_passage_ids(search)) == 5
