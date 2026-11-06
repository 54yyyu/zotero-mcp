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

    def item(self, key):
        return {
            "key": key,
            "version": self.version,
            "data": {
                "key": key,
                "itemType": "journalArticle",
                "title": f"Title {key}",
                "abstractNote": self.abstract,
                "creators": [],
                "dateAdded": "2024-01-01T00:00:00Z",
                "dateModified": "2024-01-01T00:00:00Z",
            },
        }

    def items(self, start=0, limit=100, **kwargs):
        return [] if start else [self.item(k) for k in self.keys]

    def item_versions(self, since=None, **kwargs):
        return {} if since is not None else {k: self.version for k in self.keys}

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
