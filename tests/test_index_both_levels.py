"""update-db writes article and passage records, and fills in what is missing (#769).

Runs against a real ChromaDB index in a temp directory, with an embedding function
that records what it was asked to embed, because re-embedding what is already
stored is the failure that costs users money.
"""

import importlib.util
import json
import sys

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


def test_batch_runs_do_not_ask_for_passages_they_cannot_make_yet(build):
    zotero = FakeZotero({"AAA": LONG})
    build(zotero, chunking=False).update_database()

    search = build(zotero, chunking=True, batch=True)

    assert search._find_index_gaps(limit=None, force_rebuild=False, batch=True) == {}
    assert search._find_index_gaps(limit=None, force_rebuild=False) == {"AAA": {"passages"}}


# --- the local full-text scan -----------------------------------------------------


def test_scan_keeps_an_up_to_date_item_that_has_a_gap():
    decide = semantic_search.ZoteroSemanticSearch._scan_gap_action

    assert decide({"has_fulltext": True}, {"passages"}) == "extract"
    assert decide({"has_fulltext": "failed"}, {"passages"}) == "metadata_only"
    assert decide({"has_fulltext": True}, None) is None
