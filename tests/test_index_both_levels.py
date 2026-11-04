"""update-db writes article and passage records, and fills in what is missing (#769).

Runs against a real ChromaDB index in a temp directory, with an embedding function
that records what it was asked to embed, because re-embedding what is already
stored is the failure that costs users money.
"""

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
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
from zotero_mcp.chroma_client import ChromaClient, read_index_layout  # noqa: E402

NAME = "both_levels"
LONG = " ".join(f"Sentence number {i} of the abstract." for i in range(100))


class RecordingEF(EmbeddingFunction):
    def __init__(self):
        self.embedded: list[str] = []

    def __call__(self, input):
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
    def __init__(self, abstracts):
        self.abstracts = dict(abstracts)
        self.version = 5
        self.changed: set[str] = set()

    def item(self, key):
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
        return [] if start else [self.item(k) for k in self.abstracts]

    def item_versions(self, since=None, **kwargs):
        keys = self.changed if since is not None else self.abstracts
        return {k: self.version for k in keys}

    def children(self, *a, **kw):
        return []

    def last_modified_version(self, **kwargs):
        return self.version


@pytest.fixture()
def build(monkeypatch, tmp_path):
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    config_path = tmp_path / "config.json"

    def _build(zotero, *, chunking, batch=False, route=True):
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
        existing = json.loads(config_path.read_text()) if config_path.exists() else {"semantic_search": {}}
        section = existing["semantic_search"]
        section.update(
            include_fulltext=False,
            index_schema_version=semantic_search._INDEX_SCHEMA_VERSION,
            chunking={"enabled": chunking, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
            openai_batch={"enabled": batch},
        )
        config_path.write_text(json.dumps(existing))
        client = RecordingClient(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
        client.route_passages = route
        return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))

    return _build


def _ids(search, name=NAME):
    return set(search.chroma_client.client.get_collection(name).get(include=[])["ids"])


def _passage_ids(search):
    return _ids(search, f"{NAME}_passages")


def test_chunking_on_writes_an_article_and_passages(build):
    search = build(FakeZotero({"AAA": LONG}), chunking=True)

    stats = search.update_database()

    assert stats.get("error") is None
    assert _ids(search) == {"AAA"}
    assert "AAA#0" in _passage_ids(search)
    assert len(_passage_ids(search)) == 5
    assert stats["added_items"] == 1


def test_chunking_off_writes_only_an_article(build):
    search = build(FakeZotero({"AAA": LONG}), chunking=False)

    search.update_database()

    assert _ids(search) == {"AAA"}
    assert read_index_layout(search.chroma_client.persist_directory, NAME)["passage_records"] == 0


def test_enabling_chunking_adds_passages_without_re_embedding_articles(build):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=False).update_database()

    search = build(zotero, chunking=True)
    stats = search.update_database()

    assert stats.get("error") is None
    assert len(search.chroma_client.embedded) == 10, "five passages each, and no article"
    assert _ids(search) == {"AAA", "BBB"}
    assert len(_passage_ids(search)) == 10


def test_a_second_run_embeds_nothing(build):
    zotero = FakeZotero({"AAA": LONG})
    build(zotero, chunking=True).update_database()

    search = build(zotero, chunking=True)
    search.update_database()

    assert search.chroma_client.embedded == []


def test_a_changed_item_re_embeds_both_levels(build):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=True).update_database()

    zotero.version = 6
    zotero.changed = {"AAA"}
    zotero.abstracts["AAA"] = "A much shorter abstract now."
    search = build(zotero, chunking=True)
    stats = search.update_database()

    assert stats["updated_items"] == 1
    assert len(search.chroma_client.embedded) == 2, "one article and one passage"
    assert _ids(search) == {"AAA", "BBB"}
    assert {p for p in _passage_ids(search) if p.startswith("AAA")} == {"AAA#0"}, "stale passages pruned"


def test_changing_an_item_with_chunking_off_drops_its_stale_passages(build):
    """Otherwise they stay searchable, and re-enabling chunking sees nothing to do."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=True).update_database()

    zotero.version = 6
    zotero.changed = {"AAA"}
    search = build(zotero, chunking=False)
    search.update_database()

    assert not [p for p in _passage_ids(search) if p.startswith("AAA")]
    assert [p for p in _passage_ids(search) if p.startswith("BBB")]

    search = build(zotero, chunking=True)
    search.update_database()
    assert [p for p in _passage_ids(search) if p.startswith("AAA")]


def test_missing_article_records_are_embedded_alone(build):
    """The state a migrated index starts in: passages kept, articles to embed."""
    zotero = FakeZotero({"AAA": LONG})
    search = build(zotero, chunking=True)
    search.update_database()
    search.chroma_client.delete_documents(["AAA"])

    search = build(zotero, chunking=True)
    search.update_database()

    assert len(search.chroma_client.embedded) == 1, "the article, and no passage"
    assert _ids(search) == {"AAA"}
    assert len(_passage_ids(search)) == 5


def test_a_limited_run_does_not_fill_gaps(build):
    zotero = FakeZotero({"AAA": LONG})
    build(zotero, chunking=False).update_database()

    search = build(zotero, chunking=True)
    assert search._find_index_gaps(limit=1, force_rebuild=False) == {}
    assert search._find_index_gaps(limit=None, force_rebuild=True) == {}
    assert search._find_index_gaps(limit=None, force_rebuild=False) == {"AAA": {"passages"}}


def test_force_rebuild_replaces_both_levels(build):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=True).update_database()

    zotero.abstracts = {"AAA": LONG}
    search = build(zotero, chunking=True)
    search.update_database(force_full_rebuild=True, allow_mass_deletion=True)

    assert _ids(search) == {"AAA"}
    assert not [p for p in _passage_ids(search) if p.startswith("BBB")]


def test_unmigrated_index_goes_on_as_it_did(build):
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=True, route=False).update_database()

    zotero.version = 6
    zotero.changed = {"BBB"}
    search = build(zotero, chunking=True)
    stats = search.update_database()

    assert stats.get("error") is None
    assert not [i for i in _ids(search) if "#" not in i], "no article records on an unmigrated index"
    assert read_index_layout(search.chroma_client.persist_directory, NAME)["passage_collection_exists"] is False
    assert len(search.chroma_client.embedded) == 5


def test_an_item_missing_a_later_passage_counts_as_a_gap(build):
    """Its first passage exists, so checking only that would hide the hole for good."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    search = build(zotero, chunking=True)
    search.update_database()
    search.chroma_client.delete_documents(["AAA#3"])

    search = build(zotero, chunking=True)
    assert search._find_index_gaps(limit=None, force_rebuild=False) == {"AAA": {"passages"}}

    search.update_database()

    assert len(search.chroma_client.embedded) == 5, "AAA's passages, and nothing else"
    assert len(_passage_ids(search)) == 10


def test_filling_articles_with_chunking_off_keeps_the_passages(build):
    """A migrated index with chunking off: those passages are paid for."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    search = build(zotero, chunking=True)
    search.update_database()
    search.chroma_client.delete_documents(["AAA", "BBB"])

    search = build(zotero, chunking=False)
    search.update_database()

    assert _ids(search) == {"AAA", "BBB"}
    assert len(_passage_ids(search)) == 10
    assert len(search.chroma_client.embedded) == 2


def test_no_fill_while_a_batch_run_is_unfinished(build, monkeypatch, capsys):
    """Paying for the same records again would also supersede the run."""
    zotero = FakeZotero({"AAA": LONG})
    build(zotero, chunking=False).update_database()
    search = build(zotero, chunking=True)
    monkeypatch.setattr(search, "_unfinished_batch_run", lambda: "run-1")

    search.update_database()

    assert search.chroma_client.embedded == []
    assert "run-1" in capsys.readouterr().err


def test_an_old_unfinished_batch_run_stops_blocking_fills(build, tmp_path):
    zotero = FakeZotero({"AAA": LONG})
    search = build(zotero, chunking=True)
    search._run_group_id = 0
    root = tmp_path / "openai_batches" / "old-run"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "run_id": "old-run",
        "created_at": "2020-01-01T00:00:00.000000Z",
        "group_id": search._pinned_group_id(),
        "batches": [{"batch_id": "b1", "imported_at": None}],
    }))
    assert search._unfinished_batch_run() is None

    (root / "manifest.json").write_text(json.dumps({
        "run_id": "new-run",
        "created_at": semantic_search.batch_common._utc_now(),
        "group_id": search._pinned_group_id(),
        "batches": [{"batch_id": "b1", "imported_at": None}],
    }))
    assert search._unfinished_batch_run() == "new-run"


def test_a_fill_without_full_text_waits_for_a_run_that_has_it(build, capsys):
    """Closing the gap from metadata would leave passages that cannot match the paper."""
    zotero = FakeZotero({"AAA": LONG})
    search = build(zotero, chunking=False)
    search.update_database()
    collection = search.chroma_client.collection
    stored = collection.get(ids=["AAA"], include=["metadatas"])["metadatas"][0]
    collection.update(ids=["AAA"], metadatas=[{**stored, "has_fulltext": True}])

    search = build(zotero, chunking=True)
    search.update_database()

    assert search.chroma_client.embedded == []
    assert "need their full text" in capsys.readouterr().err
    assert search._find_index_gaps(limit=None, force_rebuild=False) == {"AAA": {"passages"}}


def test_a_waiting_fill_is_not_fetched_again_on_every_run(build):
    """On a large --fulltext index this would be one API call per item, per run."""
    zotero = FakeZotero({"AAA": LONG})
    search = build(zotero, chunking=False)
    search.update_database()
    collection = search.chroma_client.collection
    stored = collection.get(ids=["AAA"], include=["metadatas"])["metadatas"][0]
    collection.update(ids=["AAA"], metadatas=[{**stored, "has_fulltext": True}])
    build(zotero, chunking=True).update_database()

    fetched = []
    real_item = zotero.item
    zotero.item = lambda key: fetched.append(key) or real_item(key)
    stats = build(zotero, chunking=True).update_database()

    assert fetched == []
    assert stats["total_items"] == 0


def test_timestamps_compare_across_the_local_and_api_formats():
    same = semantic_search._same_moment

    assert same("2026-07-02 01:01:48", "2026-07-02T01:01:48Z")
    assert same("2026-07-02T01:01:48Z", "2026-07-02T01:01:48Z")
    assert not same("2026-07-02 01:01:48", "2026-07-02T01:01:49Z")
    assert not same(None, "2026-07-02T01:01:48Z")
    assert not same("", "not a date")


def test_a_full_scan_with_a_fill_blocked_still_embeds_only_changes(build, tmp_path, monkeypatch):
    """A blocked fill must not turn the scan back into embedding everything."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=False).update_database()
    config = tmp_path / "config.json"
    data = json.loads(config.read_text())
    data["semantic_search"].pop("last_sync_version", None)
    data["semantic_search"].pop("last_sync_versions", None)
    config.write_text(json.dumps(data))

    search = build(zotero, chunking=True)
    monkeypatch.setattr(search, "_unfinished_batch_run", lambda: "run-1")
    search.update_database()

    assert search.chroma_client.embedded == []


def test_a_full_scan_of_a_gapped_index_embeds_only_what_is_missing(build, tmp_path):
    """With no sync watermark the API scan fetches everything, which must not re-embed it."""
    zotero = FakeZotero({"AAA": LONG, "BBB": LONG})
    build(zotero, chunking=False).update_database()
    config = tmp_path / "config.json"
    data = json.loads(config.read_text())
    data["semantic_search"].pop("last_sync_version", None)
    data["semantic_search"].pop("last_sync_versions", None)
    config.write_text(json.dumps(data))

    search = build(zotero, chunking=True)
    search.update_database()

    assert len(search.chroma_client.embedded) == 10, "ten passages, and no article"


# --- the local full-text scan -----------------------------------------------------


def test_scan_keeps_an_up_to_date_item_that_has_a_gap():
    decide = semantic_search.ZoteroSemanticSearch._scan_gap_action

    assert decide({"has_fulltext": True}, {"passages"}) == "extract"
    assert decide({"has_fulltext": "failed"}, {"passages"}) == "metadata_only"
    assert decide({"has_fulltext": True}, None) is None


def test_a_superseded_batch_run_does_not_block_fills(build, tmp_path, monkeypatch):
    """batch-import never touches a run a newer one took over, so it must not hold fills back."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    search = build(FakeZotero({"AAA": LONG}), chunking=True)
    search.chroma_client.embedding_model = "openai"
    search.chroma_client.embedding_config = {"model_name": "text-embedding-3-small", "api_key": "test"}
    group_id = search._pinned_group_id()
    now = datetime.now(timezone.utc)
    earlier = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    later = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    runs = {
        "older-run": {
            "run_id": "older-run", "created_at": earlier, "group_id": group_id,
            "since_version": 5, "target_sync_version": 6,
            "batches": [{"batch_id": "b1", "status": "in_progress", "imported_at": None}],
        },
        "rebuild-run": {
            "run_id": "rebuild-run", "created_at": later, "group_id": group_id,
            "force_full_rebuild": True, "since_version": None, "target_sync_version": 6,
            "batches": [{"batch_id": "b2", "status": "completed", "imported_at": later}],
        },
    }
    for run_id, manifest in runs.items():
        root = tmp_path / "openai_batches" / run_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(manifest))

    assert search._unfinished_batch_run() is None

    (tmp_path / "openai_batches" / "rebuild-run" / "manifest.json").unlink()
    assert search._unfinished_batch_run() == "older-run", "on its own it still holds fills back"
