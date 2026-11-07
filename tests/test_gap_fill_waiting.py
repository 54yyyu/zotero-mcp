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


def _age_tried(tmp_path, key, days):
    """Move the stored time of ``key``'s last unanswered try ``days`` into the past."""
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    checks = data["semantic_search"]["update_config"]["gaps_text_checked"]
    checks[key]["tried_at"] = (datetime.now() - timedelta(days=days)).isoformat()
    config_path.write_text(json.dumps(data))


def _age_check(tmp_path, key, days):
    """Move the stored time of ``key``'s last text check ``days`` into the past."""
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    checks = data["semantic_search"]["update_config"]["gaps_text_checked"]
    checks[key]["at"] = (datetime.now() - timedelta(days=days)).isoformat()
    config_path.write_text(json.dumps(data))


# --- re-checking with full text on ---------------------------------------------------


def test_a_waiting_item_is_filled_once_a_run_has_its_text(build, tmp_path):
    """The waiting list is for runs that can't have text, not for every run after."""
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)
    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]

    zotero.texts["AAA"] = TEXT
    search = build(zotero, chunking=True, fulltext=True)
    search.update_database()

    assert len(_passage_ids(search)) == 5
    assert search.chroma_client.embedded, "the passages were embedded from the text"
    assert _saved(tmp_path).get("gaps_waiting_for_text") == []


def test_a_run_without_text_still_skips_a_waiting_item(build):
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)

    zotero.fetched.clear()
    build(zotero, chunking=True).update_database()

    assert zotero.fetched == []


def test_an_item_still_without_text_is_checked_again_with_a_backoff(build, tmp_path):
    """One API call per item per run would add up on a large library."""
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)

    build(zotero, chunking=True, fulltext=True).update_database()
    assert zotero.text_requests == ["AAA"]
    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]
    assert _saved(tmp_path)["gaps_text_checked"]["AAA"]["checks"] == 1

    build(zotero, chunking=True, fulltext=True).update_database()
    assert zotero.text_requests == ["AAA"], "not again within a day"

    _age_check(tmp_path, "AAA", days=1.1)
    build(zotero, chunking=True, fulltext=True).update_database()
    assert zotero.text_requests == ["AAA", "AAA"]
    assert _saved(tmp_path)["gaps_text_checked"]["AAA"]["checks"] == 2

    _age_check(tmp_path, "AAA", days=1.5)
    build(zotero, chunking=True, fulltext=True).update_database()
    assert zotero.text_requests == ["AAA", "AAA"], "the wait doubles after each check"

    _age_check(tmp_path, "AAA", days=2.1)
    build(zotero, chunking=True, fulltext=True).update_database()
    assert zotero.text_requests == ["AAA", "AAA", "AAA"]


def test_the_backoff_stops_doubling_at_thirty_days(build):
    due = semantic_search.ZoteroSemanticSearch._text_recheck_due
    now = datetime.now()

    assert due(None, now)
    assert not due({"at": (now - timedelta(days=29)).isoformat(), "checks": 12}, now)
    assert due({"at": (now - timedelta(days=30, minutes=1)).isoformat(), "checks": 12}, now)
    assert due({"at": "not a time", "checks": 1}, now)


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


def test_a_fetch_error_on_a_waiting_item_does_not_lengthen_its_wait(build, tmp_path):
    """No answer is not "no text": the backoff stays, and the item is tried again the next day."""
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)
    build(zotero, chunking=True, fulltext=True).update_database()
    _age_check(tmp_path, "AAA", days=1.1)
    before = _saved(tmp_path)["gaps_text_checked"]["AAA"]

    zotero.fulltext_error = ConnectionError("connection reset")
    zotero.text_requests.clear()
    build(zotero, chunking=True, fulltext=True).update_database()
    after = _saved(tmp_path)["gaps_text_checked"]["AAA"]

    assert zotero.text_requests == ["AAA"]
    assert (after["at"], after["checks"]) == (before["at"], before["checks"])
    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]

    build(zotero, chunking=True, fulltext=True).update_database()
    assert zotero.text_requests == ["AAA"], "not on every run"

    _age_tried(tmp_path, "AAA", days=1.1)
    zotero.fulltext_error = None
    zotero.texts["AAA"] = TEXT
    search = build(zotero, chunking=True, fulltext=True)
    search.update_database()
    assert len(_passage_ids(search)) == 5


def test_a_waiting_item_zotero_never_returns_is_tried_once_a_day(build, tmp_path):
    """Otherwise it counts as work on every run, and no run is ever "library unchanged"."""
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)
    zotero.fetched.clear()

    def gone(key):
        zotero.fetched.append(key)
        raise ConnectionError("connection reset")

    zotero.item = gone
    build(zotero, chunking=True, fulltext=True).update_database()
    build(zotero, chunking=True, fulltext=True).update_database()

    assert zotero.fetched == ["AAA"]
    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]

    _age_tried(tmp_path, "AAA", days=1.1)
    build(zotero, chunking=True, fulltext=True).update_database()
    assert zotero.fetched == ["AAA", "AAA"]


def test_another_librarys_waiting_item_stays_due_for_its_own_library(build, tmp_path):
    """The waiting list is shared by every library in the index.

    A library updated many times a day must neither hold another library's
    waiting item back nor count it as work of its own.
    """
    zotero_a = FakeZotero({"AAA": LONG, "BBB": LONG})
    _park(build, zotero_a)
    search = build(zotero_a, chunking=True)
    collection = search.chroma_client.collection
    stored = collection.get(ids=["AAA"], include=["metadatas"])["metadatas"][0]
    collection.update(ids=["AAA"], metadatas=[{**stored, "group_id": 7}])
    del zotero_a.abstracts["AAA"]
    zotero_a.fetched.clear()

    for _ in range(3):
        stats = build(zotero_a, chunking=True, fulltext=True).update_database()
        assert stats.get("error") is None

    assert zotero_a.fetched == [], "library A neither fetched nor counted B's item"
    assert "AAA" not in (_saved(tmp_path).get("gaps_text_checked") or {})
    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]

    zotero_b = FakeZotero({"AAA": LONG})
    zotero_b.library_type, zotero_b.library_id = "groups", 7
    zotero_b.texts["AAA"] = TEXT
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    data["semantic_search"].setdefault("last_sync_versions", {})["7"] = zotero_b.version
    config_path.write_text(json.dumps(data))

    search = build(zotero_b, chunking=True, fulltext=True)
    search.update_database()

    assert zotero_b.text_requests == ["AAA"]
    assert len(_passage_ids(search)) == 10, "B's item has its passages next to BBB's"


@pytest.mark.parametrize(
    "record",
    [
        {"at": "0001-01-01T00:00:00+05:00", "checks": 1},
        {"at": "2026-01-01T00:00:00", "checks": float("inf")},
        {"at": "9999-01-01T00:00:00", "checks": 1},
        {"at": "2026-01-01T00:00:00", "checks": 1, "tried_at": "9999-01-01T00:00:00"},
        {"at": "2026-01-01T00:00:00+00:00", "checks": 1},
        {"at": "2026-01-01T00:00:00", "checks": "many"},
        {"at": None, "checks": None, "tried_at": "yesterday"},
        "not a record",
    ],
)
def test_a_damaged_check_record_does_not_stop_the_run(build, tmp_path, record):
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text())
    data["semantic_search"]["update_config"]["gaps_text_checked"] = {"AAA": record}
    config_path.write_text(json.dumps(data))

    stats = build(zotero, chunking=True, fulltext=True).update_database()

    assert stats.get("error") is None
    assert zotero.text_requests == ["AAA"], "a record that can't be read is due"
    assert _saved(tmp_path)["gaps_text_checked"]["AAA"]["checks"] >= 1


def test_no_text_in_zotero_is_not_a_fetch_error(build, tmp_path):
    """Zotero answers 404 for an attachment it has no text for."""
    from pyzotero.zotero_errors import ResourceNotFoundError

    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)
    zotero.fulltext_error = ResourceNotFoundError("Not found")

    build(zotero, chunking=True, fulltext=True).update_database()

    assert _saved(tmp_path)["gaps_text_checked"]["AAA"]["checks"] == 1


# --- runs that keep the waiting list -------------------------------------------------


def test_a_limited_run_keeps_the_waiting_list(build, tmp_path):
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)

    build(zotero, chunking=True).update_database(limit=1)

    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]


def test_a_run_whose_fill_is_held_back_keeps_the_waiting_list(build, tmp_path, monkeypatch):
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)
    zotero.version = 6
    zotero.changed = {"AAA"}

    search = build(zotero, chunking=True)
    monkeypatch.setattr(search, "_unfinished_batch_run", lambda: "run-1")
    search.update_database()

    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]


def test_a_run_stopped_by_an_unreachable_api_keeps_the_waiting_list(build, tmp_path):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    _park(build, zotero)
    zotero.version = 6
    zotero.changed = {"BBB"}

    search = build(zotero, chunking=True)
    search.chroma_client.embedding_function.unreachable = True
    stats = search.update_database()

    assert stats.get("embedding_api_unreachable")
    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]


# --- full scans of a split index -----------------------------------------------------


def test_a_full_scan_of_a_complete_split_index_embeds_nothing(build, tmp_path):
    """No gaps, no sync version: the scan must not pay for the library again."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=True).update_database()
    config = tmp_path / "config.json"
    data = json.loads(config.read_text())
    data["semantic_search"].pop("last_sync_version", None)
    data["semantic_search"].pop("last_sync_versions", None)
    config.write_text(json.dumps(data))

    search = build(zotero, chunking=True)
    assert search._find_index_gaps(limit=None, force_rebuild=False) == {}
    stats = search.update_database()

    assert search.chroma_client.embedded == []
    assert stats["total_items"] == 0


def test_a_full_scan_picks_up_text_for_an_item_that_had_none(build, tmp_path):
    """Attaching a PDF doesn't change the item, so a full scan is where its text comes in."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    search = build(zotero, chunking=True, fulltext=True)
    search.update_database()
    stored = search.chroma_client.get_document_metadata("AAA")
    assert stored["has_fulltext"] == "failed"
    config = tmp_path / "config.json"
    data = json.loads(config.read_text())
    data["semantic_search"].pop("last_sync_version", None)
    data["semantic_search"].pop("last_sync_versions", None)
    config.write_text(json.dumps(data))

    zotero.texts["AAA"] = TEXT
    search = build(zotero, chunking=True, fulltext=True)
    search.update_database()

    embedded = search.chroma_client.embedded
    assert embedded and any("Body sentence" in text for text in embedded)
    assert not any("Title BBB" in text for text in embedded), "BBB is unchanged"
    assert search.chroma_client.get_document_metadata("AAA")["has_fulltext"] is True


# --- the Batch API path ----------------------------------------------------------


def _batch(search):
    """Make ``search`` an OpenAI batch run; any request would hit a closed local port."""
    search.chroma_client.embedding_model = "openai"
    search.chroma_client.embedding_config = {
        "model_name": "text-embedding-3-small",
        "api_key": "test",
        "base_url": "http://127.0.0.1:9",
    }
    return search


def test_a_batch_run_with_nothing_to_embed_moves_the_sync_version(build, tmp_path):
    """Otherwise every later run fetches the same changes again, and never gets past them."""
    zotero = FakeZotero({"AAA": LONG})
    build(zotero, chunking=False).update_database()
    zotero.version = 6
    zotero.changed = set()

    search = _batch(build(zotero, chunking=False))
    stats = search.update_database(use_batch=True)

    assert stats.get("batch_submitted") is False
    assert "batch_error" not in stats
    assert list(tmp_path.glob("**/manifest.json")) == []
    assert build(zotero, chunking=False)._load_last_sync_version() == 6


def test_a_batch_run_whose_items_all_fail_keeps_the_sync_version(build, monkeypatch):
    zotero = FakeZotero({"AAA": LONG})
    build(zotero, chunking=False).update_database()
    zotero.version = 6
    zotero.changed = {"AAA"}

    search = _batch(build(zotero, chunking=False))

    def broken(*args, **kwargs):
        raise ValueError("cannot build")

    monkeypatch.setattr(search, "_build_item_records", broken)
    search.update_database(use_batch=True)

    assert build(zotero, chunking=False)._load_last_sync_version() == 5


def test_a_batch_run_saves_what_it_learned_about_waiting_items(build, tmp_path):
    """update_config is saved on the batch path too, with the waiting list in it."""
    zotero = FakeZotero({"AAA": LONG})
    search = build(zotero, chunking=True)
    search.update_database()
    search.chroma_client.delete_documents(["AAA"])
    passages = search.chroma_client.client.get_collection(f"{NAME}_passages")
    stored = passages.get(ids=["AAA#0"], include=["metadatas"])["metadatas"][0]
    passages.update(ids=["AAA#0"], metadatas=[{**stored, "has_fulltext": True}])
    zotero.version = 6

    search = _batch(build(zotero, chunking=True))
    search.update_database(use_batch=True)

    assert _saved(tmp_path)["gaps_waiting_for_text"] == ["AAA"]


# --- the waiting-list message fits the mode ------------------------------------------


def test_web_mode_message_does_not_send_the_user_to_fulltext(build, capsys):
    """'update-db --fulltext' errors outside local mode, so the message can't ask for it."""
    zotero = FakeZotero({"AAA": LONG})
    _park(build, zotero)
    capsys.readouterr()

    build(zotero, chunking=True, fulltext=True).update_database()
    err = capsys.readouterr().err
    assert "need their full text" in err
    assert "once Zotero has extracted its text" in err
    assert "run 'zotero-mcp update-db --fulltext'" not in err


def test_web_mode_message_without_fulltext_names_the_setting(build, capsys):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    _park(build, zotero)
    zotero.version = 6
    zotero.changed = {"BBB"}
    capsys.readouterr()

    build(zotero, chunking=True).update_database()
    err = capsys.readouterr().err
    assert "need their full text" in err
    assert "semantic_search.include_fulltext" in err
    assert "run 'zotero-mcp update-db --fulltext'" not in err


def test_local_mode_message_still_points_at_fulltext(monkeypatch):
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    assert semantic_search._waiting_for_text_hint(True) == "run 'zotero-mcp update-db --fulltext' to add them."
    assert semantic_search._waiting_for_text_hint(False) == "run 'zotero-mcp update-db --fulltext' to add them."
