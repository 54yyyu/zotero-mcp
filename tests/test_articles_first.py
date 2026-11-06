"""A run writing both levels embeds every article record before any passage (#769).

Article-level search is then complete early in a long run, and a run stopped
between the two passes leaves only passages, which the gap fill adds without
paying for the articles again. Runs against a real ChromaDB index in a temp
directory with an embedding function that records what it embeds.
"""

import importlib.util
import json
import sys
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

NAME = "articles_first"
LONG = " ".join(f"Sentence number {i} of the abstract." for i in range(100))
FULLTEXT = " ".join(f"Full text sentence {i} from the PDF." for i in range(200))


class RecordingEF(EmbeddingFunction):
    def __init__(self, parallel=1):
        self.embedded: list[str] = []
        if parallel > 1:
            # Picks the streaming pipeline.
            self.max_parallel_requests = parallel
            self.request_batch_size = 3

    def __call__(self, input):
        self.embedded.extend(input)
        return [[1.0, 0.0] for _ in input]

    def truncate(self, text, max_tokens):
        return text


class RecordingClient(ChromaClient):
    parallel = 1

    def _create_embedding_function(self):
        return RecordingEF(self.parallel)

    @property
    def embedded(self):
        return self.embedding_function.embedded


class FakeZotero:
    def __init__(self, keys, abstract=LONG):
        self.keys = list(keys)
        self.abstract = abstract
        self.version = 5
        # Items changed since the last sync: key -> new abstract.
        self.changed: dict[str, str] = {}

    def item(self, key):
        return {
            "key": key,
            "version": self.version,
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Title {key}",
                "abstractNote": self.changed.get(key, self.abstract),
                "creators": [],
                "dateAdded": "2024-01-01T00:00:00Z",
                "dateModified": "2024-02-01T00:00:00Z" if key in self.changed else "2024-01-01T00:00:00Z",
            },
        }

    def items(self, start=0, limit=100, **kwargs):
        return [] if start else [self.item(k) for k in self.keys]

    def item_versions(self, since=None, **kwargs):
        keys = self.changed if since is not None else self.keys
        return {k: self.version for k in keys}

    def children(self, *a, **kw):
        return []

    def last_modified_version(self, **kwargs):
        return self.version


@pytest.fixture()
def build(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    config_path = tmp_path / "config.json"

    def _build(zotero, *, parallel=1, route=True, chunking=True, include_fulltext=False):
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: zotero)
        existing = json.loads(config_path.read_text()) if config_path.exists() else {"semantic_search": {}}
        existing["semantic_search"].update(
            include_fulltext=include_fulltext,
            index_schema_version=semantic_search._INDEX_SCHEMA_VERSION,
            chunking={"enabled": chunking, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
            openai_batch={"enabled": False},
        )
        config_path.write_text(json.dumps(existing))
        client_cls = type("Client", (RecordingClient,), {"parallel": parallel})
        client = client_cls(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
        client.route_passages = route
        writes: list[str] = []
        for name in ("upsert_documents", "upsert_embeddings"):
            original = getattr(client, name)

            def spy(*args, _original=original, **kwargs):
                ids = kwargs.get("ids") if "ids" in kwargs else args[2]
                writes.extend(ids)
                return _original(*args, **kwargs)

            monkeypatch.setattr(client, name, spy)
        search = semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))
        search.writes = writes
        return search

    return _build


def _ids(search, name=NAME):
    return set(search.chroma_client.client.get_collection(name).get(include=[])["ids"])


def _passage_ids(search):
    names = {c.name if hasattr(c, "name") else c for c in search.chroma_client.client.list_collections()}
    return _ids(search, f"{NAME}_passages") if f"{NAME}_passages" in names else set()


def _assert_articles_first(writes, count):
    articles = [i for i in writes if "#" not in i]
    assert len(articles) == count
    last_article = max(writes.index(i) for i in articles)
    first_passage = min(n for n, i in enumerate(writes) if "#" in i)
    assert last_article < first_passage, writes


KEYS = [f"K{n:03d}" for n in range(30)]  # more than one slice of 25


@pytest.mark.parametrize("parallel", [1, 2], ids=["sequential", "streaming"])
def test_every_article_is_written_before_any_passage(build, parallel):
    search = build(FakeZotero(KEYS), parallel=parallel)

    stats = search.update_database()

    assert stats.get("error") is None
    _assert_articles_first(search.writes, len(KEYS))
    assert _ids(search) == set(KEYS)
    assert len(_passage_ids(search)) == 5 * len(KEYS)
    assert stats["added_items"] == len(KEYS), "each item counted once, not once per pass"
    assert stats["processed_items"] == len(KEYS)
    assert stats["updated_items"] == 0


@pytest.mark.parametrize("parallel", [1, 2], ids=["sequential", "streaming"])
def test_full_text_is_fetched_once_and_not_held_in_memory_between_passes(
    build, monkeypatch, parallel
):
    fetched: list[str] = []

    def fetch(self, key):
        fetched.append(key)
        return FULLTEXT, "pdf", False

    monkeypatch.setattr(semantic_search.ZoteroSemanticSearch, "_fetch_fulltext_checked", fetch)
    # Ask per item, so each fetch names its item.
    monkeypatch.setattr(semantic_search.ZoteroSemanticSearch, "_BULK_ATTACHMENT_WALK_MIN_ITEMS", 10_000)
    search = build(FakeZotero(KEYS, abstract="A short abstract."), parallel=parallel, include_fulltext=True)

    items_seen: list[list[dict]] = []
    original_source = search._get_items_from_source

    def capture(*args, **kwargs):
        items = original_source(*args, **kwargs)
        items_seen.append(items)
        return items

    monkeypatch.setattr(search, "_get_items_from_source", capture)
    in_memory_at_passage_pass: list[int] = []
    original_prepare = search._prepare_and_classify_slice

    def prepare(items, *args, **kwargs):
        if kwargs.get("level") == semantic_search.SEARCH_PASSAGES and not in_memory_at_passage_pass:
            in_memory_at_passage_pass.append(
                sum(1 for item in items_seen[0] if item["data"].get("fulltext"))
            )
        return original_prepare(items, *args, **kwargs)

    monkeypatch.setattr(search, "_prepare_and_classify_slice", prepare)

    stats = search.update_database()

    assert stats.get("error") is None
    assert sorted(fetched) == sorted(KEYS), "each item's text fetched once"
    _assert_articles_first(search.writes, len(KEYS))
    assert in_memory_at_passage_pass == [0], "text is kept on disk between the passes"
    passages = search.chroma_client.client.get_collection(f"{NAME}_passages").get(
        include=["documents"]
    )["documents"]
    assert len(passages) == 5 * len(KEYS)
    assert sum("Full text sentence" in doc for doc in passages) >= 4 * len(KEYS)


@pytest.mark.parametrize("parallel", [1, 2], ids=["sequential", "streaming"])
def test_a_stop_between_the_passes_resumes_through_the_gap_fill(build, monkeypatch, parallel):
    evicted: list[str] = []
    monkeypatch.setattr(
        semantic_search.fulltext_cache,
        "evict_many",
        lambda keys, config_path=None: evicted.extend(keys) or 0,
    )
    zotero = FakeZotero(KEYS)
    search = build(zotero, parallel=parallel)
    ef = search.chroma_client.embedding_function
    original_call = type(ef).__call__

    def refuse_passages(self, input):
        if any(len(text) < 600 for text in input):  # a passage, not a whole article
            raise EmbeddingAPIUnreachable("openai", "api.example", "connection refused")
        return original_call(self, input)

    monkeypatch.setattr(type(ef), "__call__", refuse_passages)

    stats = search.update_database()

    assert stats.get("embedding_api_unreachable") == "api.example"
    assert _ids(search) == set(KEYS), "every article written before the stop"
    assert _passage_ids(search) == set()
    assert evicted == [], "the extracted text stays cached for the passages"
    config = json.loads(Path(search.config_path).read_text())
    assert not config["semantic_search"].get("last_sync_versions"), "sync version not moved"

    monkeypatch.setattr(type(ef), "__call__", original_call)
    search = build(zotero, parallel=parallel)
    stats = search.update_database()

    assert stats.get("error") is None
    assert len(search.chroma_client.embedded) == 5 * len(KEYS), "passages only, no article again"
    assert len(_passage_ids(search)) == 5 * len(KEYS)


def test_the_text_cache_is_evicted_after_the_passage_pass(build, monkeypatch):
    evicted: list[str] = []
    monkeypatch.setattr(
        semantic_search.fulltext_cache,
        "evict_many",
        lambda keys, config_path=None: evicted.extend(keys) or 0,
    )
    search = build(FakeZotero(KEYS))
    order: list[str] = []
    original_upsert = search.chroma_client.upsert_documents

    def upsert(documents, metadatas, ids):
        order.append("passages" if any("#" in i for i in ids) else "articles")
        return original_upsert(documents, metadatas, ids)

    monkeypatch.setattr(search.chroma_client, "upsert_documents", upsert)
    real_evict = semantic_search.fulltext_cache.evict_many

    def evict(keys, config_path=None):
        order.append("evict")
        return real_evict(keys, config_path=config_path)

    monkeypatch.setattr(semantic_search.fulltext_cache, "evict_many", evict)

    search.update_database()

    assert sorted(evicted) == sorted(KEYS)
    assert order.index("evict") > order.index("passages"), order


@pytest.mark.parametrize("route", [False], ids=["unmigrated"])
def test_an_unmigrated_index_writes_in_one_pass(build, route):
    search = build(FakeZotero(["AAA", "BBB"]), route=route)

    stats = search.update_database()

    assert stats.get("error") is None
    assert search.writes == [f"AAA#{n}" for n in range(5)] + [f"BBB#{n}" for n in range(5)]


def test_an_item_level_index_writes_articles_only(build):
    search = build(FakeZotero(["AAA", "BBB"]), chunking=False)

    search.update_database()

    assert search.writes == ["AAA", "BBB"]


CHANGED = " ".join(f"Rewritten sentence {i} of the new abstract." for i in range(100))


def _stop_in_the_passage_pass(monkeypatch, search):
    ef = search.chroma_client.embedding_function
    original_call = type(ef).__call__

    def refuse_passages(self, input):
        if any(len(text) < 600 for text in input):  # a passage, not a whole article
            raise EmbeddingAPIUnreachable("openai", "api.example", "connection refused")
        return original_call(self, input)

    monkeypatch.setattr(type(ef), "__call__", refuse_passages)
    stats = search.update_database()
    monkeypatch.setattr(type(ef), "__call__", original_call)
    assert stats.get("embedding_api_unreachable") == "api.example"


def _passages_of(search, key):
    got = search.chroma_client.client.get_collection(f"{NAME}_passages").get(
        where={"parent_item_key": key}, include=["documents"]
    )
    return got["documents"]


def _changed_and_stopped(build, monkeypatch, parallel):
    zotero = FakeZotero(["AAA", "BBB"])
    build(zotero, parallel=parallel).update_database()
    zotero.version = 6
    zotero.changed = {"AAA": CHANGED}
    _stop_in_the_passage_pass(monkeypatch, build(zotero, parallel=parallel))
    return zotero


@pytest.mark.parametrize("parallel", [1, 2], ids=["sequential", "streaming"])
def test_a_stop_leaves_a_changed_item_without_its_stale_passages(build, monkeypatch, parallel):
    """Otherwise a full scan sees the new article, and the gap fill can't see the old passages."""
    zotero = _changed_and_stopped(build, monkeypatch, parallel)
    search = build(zotero, parallel=parallel)

    assert _passages_of(search, "AAA") == []
    assert len(_passages_of(search, "BBB")) == 5, "an unchanged item keeps its passages"


@pytest.mark.parametrize("parallel", [1, 2], ids=["sequential", "streaming"])
def test_a_full_scan_after_a_stop_writes_a_changed_items_new_passages(build, monkeypatch, parallel):
    zotero = _changed_and_stopped(build, monkeypatch, parallel)
    config_path = Path(build(zotero).config_path)
    config = json.loads(config_path.read_text())
    config["semantic_search"].pop("last_sync_versions", None)
    config["semantic_search"].pop("last_sync_version", None)
    config_path.write_text(json.dumps(config))  # a full scan, as after a migration

    search = build(zotero, parallel=parallel)
    stats = search.update_database()

    assert stats.get("error") is None
    passages = _passages_of(search, "AAA")
    assert len(passages) == 5
    assert all("Sentence number" not in p for p in passages)
    assert len(search.chroma_client.embedded) == 5, "AAA's passages only, not its article"



@pytest.mark.parametrize("parallel", [1, 2], ids=["sequential", "streaming"])
def test_the_next_incremental_run_after_a_stop_pays_only_for_the_passages(build, monkeypatch, parallel):
    zotero = _changed_and_stopped(build, monkeypatch, parallel)

    search = build(zotero, parallel=parallel)
    stats = search.update_database()

    assert stats.get("error") is None
    assert len(_passages_of(search, "AAA")) == 5
    assert all("Rewritten sentence" in p or "Title AAA" in p for p in _passages_of(search, "AAA"))
    assert len(search.chroma_client.embedded) == 5, "AAA's current article is not embedded again"


def test_an_incremental_change_still_rewrites_a_stale_article(build):
    zotero = FakeZotero(["AAA"])
    build(zotero).update_database()
    zotero.version = 6
    zotero.changed = {"AAA": CHANGED}

    search = build(zotero)
    search.update_database()

    assert len(search.chroma_client.embedded) == 6, "the article and five passages"


def test_an_article_written_with_full_text_is_rewritten_by_a_run_without_it(build, monkeypatch):
    """Passages-only would leave the two levels written from different text."""
    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch, "_fetch_fulltext_checked",
        lambda self, key: (FULLTEXT, "pdf", False),
    )
    zotero = FakeZotero(["AAA"])
    build(zotero, include_fulltext=True).update_database()
    zotero.version = 6
    zotero.changed = {"AAA": CHANGED}
    _stop_in_the_passage_pass(monkeypatch, build(zotero, include_fulltext=True))

    search = build(zotero, include_fulltext=False)
    search.update_database()

    assert len(search.chroma_client.embedded) == 6, "the article and five passages"


def test_the_spilled_text_is_removed_after_a_stop(build, monkeypatch, tmp_path):
    import tempfile

    spill = tmp_path / "spill"
    spill.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spill))
    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch, "_fetch_fulltext_checked",
        lambda self, key: (FULLTEXT, "pdf", False),
    )
    monkeypatch.setattr(semantic_search.ZoteroSemanticSearch, "_BULK_ATTACHMENT_WALK_MIN_ITEMS", 10_000)
    search = build(FakeZotero(KEYS), include_fulltext=True)
    held: list[str] = []
    original_hold = semantic_search._HeldText.hold

    def hold(self, item):
        original_hold(self, item)
        if item.get("_held_fulltext"):
            held.append(item["_held_fulltext"])

    monkeypatch.setattr(semantic_search._HeldText, "hold", hold)

    _stop_in_the_passage_pass(monkeypatch, search)

    assert held and all(p.startswith(str(spill)) for p in held)
    assert list(spill.iterdir()) == []


def test_held_text_after_close_stays_in_memory():
    """A producer thread still running after the run closed must not make a new directory."""
    held = semantic_search._HeldText()
    held.close()
    item = {"key": "AAA", "data": {"fulltext": "text"}}

    held.hold(item)

    assert item["data"]["fulltext"] == "text"
    assert "_held_fulltext" not in item


@pytest.mark.parametrize("parallel", [1, 2], ids=["sequential", "streaming"])
def test_old_passages_that_could_not_be_deleted_are_reported(build, monkeypatch, caplog, parallel):
    """ChromaDB's delete swallows its errors; the run must still notice."""
    import logging

    zotero = FakeZotero(["AAA", "BBB"])
    build(zotero, parallel=parallel).update_database()
    zotero.version = 6
    zotero.changed = {"AAA": CHANGED}
    search = build(zotero, parallel=parallel)
    monkeypatch.setattr(search.chroma_client, "prune_item_chunks", lambda key, keep: None)

    noted: list[str] = []
    original_note = search._note_undropped_passages

    def note(item_key):
        original_note(item_key)
        noted.append(item_key)
        assert item_key in search._undropped_passages

    monkeypatch.setattr(search, "_note_undropped_passages", note)

    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        _stop_in_the_passage_pass(monkeypatch, search)

    assert noted == ["AAA"]
    assert any("AAA" in r.getMessage() and "passages" in r.getMessage() for r in caplog.records)


def test_undropped_passages_do_not_carry_over_to_the_next_run(build, monkeypatch):
    """The server keeps one instance: a later run must not start with an earlier run's items."""
    zotero = FakeZotero(["AAA", "BBB"])
    build(zotero).update_database()
    zotero.version = 6
    zotero.changed = {"AAA": CHANGED}
    search = build(zotero)
    monkeypatch.setattr(search.chroma_client, "prune_item_chunks", lambda key, keep: None)
    _stop_in_the_passage_pass(monkeypatch, search)

    # A later run of the same instance, and a delete that fails after it (say
    # in a batch collect), must not add to the set the earlier run left.
    zotero.changed = {}
    search.update_database()
    search._note_undropped_passages("CCC")

    assert not search._undropped_passages
