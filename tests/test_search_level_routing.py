"""Choosing between article and passage search, on a real index (#769)."""

import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

from zotero_mcp import chroma_client, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402


def _meta(key, **extra):
    return {"group_id": 0, "item_key": key, **extra}


def _passage_meta(key, index, total=2):
    return _meta(key, parent_item_key=key, chunk_index=index, n_chunks=total)


def _write(client, ids, metadatas):
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=metadatas,
        ids=ids,
        embeddings=[[1.0, 0.0] for _ in ids],
    )


@pytest.fixture()
def build(monkeypatch, tmp_path):
    """A search over a real index, embedding queries with a stub so nothing downloads."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(semantic_search, "get_zotero_client", MagicMock)
    monkeypatch.setattr(chroma_client, "CUSTOM_EMBEDDING_FUNCTIONS", (SimpleNamespace,))

    def _build(*, route=True, chunking=False, articles=(), passages=()):
        client = ChromaClient(collection_name="levels", persist_directory=str(tmp_path / "chroma"))
        if not route and passages:
            _write(client, list(passages), [_passage_meta(p.split("#")[0], int(p.split("#")[1])) for p in passages])
        client = ChromaClient(collection_name="levels", persist_directory=str(tmp_path / "chroma"))
        client.route_passages = route
        if articles:
            _write(client, list(articles), [_meta(a) for a in articles])
        if route and passages:
            _write(client, list(passages), [_passage_meta(p.split("#")[0], int(p.split("#")[1])) for p in passages])
        client.embedding_function = SimpleNamespace(embed_query_text=lambda _q: [1.0, 0.0])
        search = semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(tmp_path / "config.json"))
        search._chunking_config = {"enabled": chunking}
        search._reranker_config = {"enabled": False}
        search._attach_zotero_items = lambda enriched: None
        return search

    return _build


def _keys(result):
    assert "error" not in result, result.get("error")
    return [r["item_key"] for r in result["results"]]


def test_default_follows_the_chunking_setting(build):
    search = build(chunking=True, articles=["ART"], passages=["PAS#0", "PAS#1"])
    result = search.search("q")
    assert _keys(result) == ["PAS"]
    assert result["level"] == "passages"

    search._chunking_config = {"enabled": False}
    result = search.search("q")
    assert _keys(result) == ["ART"]
    assert result["level"] == "articles"


def test_caller_can_choose_either_level(build):
    search = build(chunking=True, articles=["ART"], passages=["PAS#0"])

    assert _keys(search.search("q", level="articles")) == ["ART"]
    assert _keys(search.search("q", level="passages")) == ["PAS"]


def test_article_search_does_not_open_the_passage_collection(build):
    search = build(chunking=False, articles=["ART"], passages=["PAS#0"])
    client = search.chroma_client
    client._passage_collection = None

    search.search("q", level="articles")

    assert client._passage_collection is None


def test_chunking_just_enabled_falls_back_to_articles(build):
    """Until passages are built, the default must still find something."""
    search = build(chunking=True, articles=["ART"])

    result = search.search("q")

    assert _keys(result) == ["ART"]
    assert result["level"] == "articles"


def test_default_is_passages_from_the_first_passage_of_a_fill(build):
    """Only the setting picks the level, so results never switch mid-fill."""
    articles = [f"A{i}" for i in range(10)]
    search = build(chunking=True, articles=articles, passages=["A0#0", "A0#1"])

    assert search.search("q", limit=20)["level"] == "passages"

    _write(search.chroma_client, [f"A{i}#0" for i in range(1, 9)], [_passage_meta(f"A{i}", 0, 1) for i in range(1, 9)])

    assert search.search("q", limit=20)["level"] == "passages"


# --- telling the caller what was searched ----------------------------------------


def test_note_points_at_the_article_level_during_a_fill(build):
    articles = [f"A{i}" for i in range(12)]
    search = build(chunking=True, articles=articles, passages=["A0#0", "A0#1"])

    note = search.search("q")["level_note"]

    assert "Searched passages (1 item in the index)" in note
    assert "article level has 12 items" in note
    assert "may find more" in note


def test_note_gives_both_counts_when_neither_level_has_more(build):
    search = build(chunking=True, articles=["A", "B"], passages=["A#0", "B#0"])

    note = search.search("q")["level_note"]

    assert "Searched passages (2 items in the index)" in note
    assert "article level has 2 items" in note
    assert "may find more" not in note


def test_note_on_an_article_search_gives_the_passage_count(build):
    search = build(chunking=False, articles=["A", "B"], passages=["A#0"])

    note = search.search("q")["level_note"]

    assert "Searched articles (2 items in the index)" in note
    assert "passage level has 1 item" in note
    assert "may find more" not in note


def test_note_says_passages_are_not_there_yet(build):
    search = build(chunking=True, articles=["ART"])

    note = search.search("q")["level_note"]

    assert "Searched articles (1 item in the index)" in note
    assert "No passages are indexed yet" in note


def test_note_says_articles_are_not_there_yet(build):
    search = build(chunking=False, passages=["PAS#0"])

    note = search.search("q")["level_note"]

    assert "Searched passages (1 item in the index)" in note
    assert "No article records are indexed yet" in note


def test_note_says_articles_are_not_there_yet_with_chunking_on(build):
    """A passage-only index tells the agent articles are missing, whatever the setting."""
    search = build(chunking=True, passages=["PAS#0"])

    note = search.search("q")["level_note"]

    assert "Searched passages (1 item in the index)" in note
    assert "No article records are indexed yet" in note


def test_passage_count_includes_an_item_missing_its_first_passage(build):
    """An item whose passage 0 failed to write still has passages to search."""
    search = build(chunking=True, articles=["A", "B"], passages=["A#0", "A#1", "B#1"])

    assert search.chroma_client.index_layout()["passage_items"] == 2
    assert "Searched passages (2 items in the index)" in search.search("q")["level_note"]


def test_passage_count_handles_keys_of_different_lengths(build):
    """Items whose keys share a prefix are each counted once."""
    passages = ["AB#0", "AB#1", "AB1#0", "AB1#1", "ABC#2", "B#0"]
    search = build(chunking=True, articles=["A"], passages=passages)

    assert search.chroma_client.index_layout()["passage_items"] == 4


def _failing_reads(monkeypatch, *fragments):
    """Make chroma_client's SQLite reads fail for queries containing any fragment.

    Stands in for a ChromaDB upgrade that changes the tables the counts read.
    """
    import sqlite3

    real_connect = sqlite3.connect

    class Connection:
        def __init__(self, con):
            self._con = con

        def execute(self, sql, *args):
            if any(f in sql for f in fragments):
                raise sqlite3.OperationalError("no such table: embeddings")
            return self._con.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._con, name)

    fake = SimpleNamespace(**{k: getattr(sqlite3, k) for k in dir(sqlite3) if not k.startswith("__")})
    fake.connect = lambda *a, **k: Connection(real_connect(*a, **k))
    monkeypatch.setattr(chroma_client, "sqlite3", fake)
    monkeypatch.setattr(chroma_client, "_passage_count_failure_logged", False, raising=False)


def test_search_works_when_the_passage_item_count_cannot_be_read(build, monkeypatch, caplog):
    search = build(chunking=True, articles=["A", "B"], passages=["A#0", "A#1", "B#0"])
    _failing_reads(monkeypatch, "WITH RECURSIVE")

    result = search.search("q")
    search.search("q")

    assert sorted(_keys(result)) == ["A", "B"]
    assert "Searched passages (2 items in the index)" in result["level_note"]
    assert sum("passage" in r.getMessage() for r in caplog.records if r.levelname == "WARNING") == 1


def test_search_works_without_any_passage_item_count(build, monkeypatch):
    search = build(chunking=True, articles=["A", "B"], passages=["A#0", "B#0"])
    _failing_reads(monkeypatch, "WITH RECURSIVE", "chunk_index")

    result = search.search("q")
    article_result = search.search("q", level="articles")

    assert sorted(_keys(result)) == ["A", "B"]
    assert result["level_note"] is None
    assert sorted(_keys(article_result)) == ["A", "B"]
    assert "passage" not in (article_result["level_note"] or "")


def test_chromadb_keeps_the_id_index_the_passage_count_seeks_through(build):
    """The item count relies on this index for speed; a ChromaDB upgrade without it should fail here."""
    import sqlite3

    search = build(chunking=True, passages=["A#0"])
    db = f"{(chroma_client.Path(search.chroma_client.persist_directory) / 'chroma.sqlite3').as_uri()}?mode=ro"
    con = sqlite3.connect(db, uri=True)
    try:
        indexes = con.execute("PRAGMA index_list(embeddings)").fetchall()
        columns = [
            [c[2] for c in con.execute(f"PRAGMA index_info('{name}')").fetchall()]
            for _, name, unique, *_ in indexes
            if unique
        ]
    finally:
        con.close()

    assert ["segment_id", "embedding_id"] in columns


def test_no_note_when_the_index_has_only_the_level_the_setting_asks_for(build):
    assert build(chunking=False, articles=["ART"]).search("q")["level_note"] is None


def test_no_note_on_an_unmigrated_index(build):
    search = build(route=False, chunking=True, passages=["PAS#0"])
    search.chroma_client.route_passages = True

    assert search.search("q")["level_note"] is None


@pytest.fixture()
def tool_output(monkeypatch, tmp_path):
    """The MCP tool's text for a search that returns these results and this note."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    from zotero_mcp import client as client_module
    from zotero_mcp.tools import search as search_tool

    def _run(results, note, load_note=None):
        fake = MagicMock()
        fake.search.return_value = {
            "results": results, "level": "passages", "level_note": note, "load_note": load_note,
        }
        monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: fake)
        monkeypatch.setattr(search_tool, "_maybe_fire_presearch_sync", lambda *a, **k: None)
        monkeypatch.setattr(client_module, "get_active_group_id", lambda: None, raising=False)
        return search_tool.semantic_search(query="q", ctx=MagicMock())

    return _run


def test_tool_prints_the_note_above_the_results(tool_output):
    hit = {"item_key": "A", "similarity_score": 0.5, "matched_text": "text", "zotero_item": {"data": {"title": "T"}}}

    text = tool_output([hit], "Searched passages (1 item in the index).")

    assert "Searched passages (1 item in the index)." in text
    assert text.index("Searched passages") < text.index("T")


def test_tool_prints_the_note_when_nothing_was_found(tool_output):
    """An empty result is when the other level matters most."""
    text = tool_output([], "The article level has 12 items, so searching it may find more.")

    assert "No semantically similar items" in text
    assert "may find more" in text


def test_migrated_index_finds_passages_before_its_articles_exist(build):
    """With chunking off, the articles the default would pick are not there yet."""
    search = build(chunking=False, passages=["PAS#0"])

    result = search.search("q")

    assert _keys(result) == ["PAS"]
    assert result["level"] == "passages"


def test_asking_for_passages_that_do_not_exist_says_so(build):
    search = build(chunking=False, articles=["ART"])

    result = search.search("q", level="passages")

    assert result["results"] == []
    assert "chunking" in result["error"]


def test_asking_for_articles_that_do_not_exist_yet_says_so(build):
    search = build(chunking=True, passages=["PAS#0"])

    result = search.search("q", level="articles")

    assert result["results"] == []
    assert "update-db" in result["error"]


def test_an_unknown_level_is_refused(build):
    search = build(articles=["ART"])

    result = search.search("q", level="sentences")

    assert result["results"] == []
    assert "sentences" in result["error"]


def test_passage_search_fetches_enough_for_grouping(build):
    search = build(chunking=False, articles=["ART"], passages=["PAS#0"])
    seen = {}
    real = search.chroma_client.search

    def spy(query_texts, n_results=10, where=None, **kwargs):
        seen["n_results"] = n_results
        return real(query_texts, n_results=n_results, where=where, **kwargs)

    search.chroma_client.search = spy

    search.search("q", limit=5, level="passages")
    assert seen["n_results"] == 20

    search.search("q", limit=5, level="articles")
    assert seen["n_results"] == 5


# --- an unmigrated index ---------------------------------------------------------


def test_unmigrated_index_searches_as_it_always_did(build):
    search = build(route=False, chunking=True, passages=["PAS#0", "PAS#1"])
    search.chroma_client.route_passages = True

    result = search.search("q")

    assert _keys(result) == ["PAS"]
    assert result["level"] == "passages"


def test_unmigrated_index_cannot_search_articles_yet(build):
    search = build(route=False, chunking=True, passages=["PAS#0"])
    search.chroma_client.route_passages = True

    result = search.search("q", level="articles")

    assert result["results"] == []
    assert "migrate" in result["error"]


def test_mixed_unmigrated_index_is_not_said_to_hold_passages_only(build):
    """Articles and passages share one ChromaDB collection until migrate-index splits them."""
    search = build(route=False, chunking=True, articles=["ART"], passages=["PAS#0"])
    search.chroma_client.route_passages = True

    result = search.search("q", level="articles")

    assert result["results"] == []
    assert "passages only" not in result["error"]
    assert "together" in result["error"]
    # migrate-index embeds nothing, so articles still missing need update-db.
    assert result["error"].rstrip(".").endswith(
        "run 'zotero-mcp migrate-index', then 'zotero-mcp update-db'"
    )


def test_a_client_double_without_levels_still_searches():
    """Fakes in this suite and old callers pass no level and know no layout."""
    search = semantic_search.ZoteroSemanticSearch.__new__(semantic_search.ZoteroSemanticSearch)
    search.chroma_client = SimpleNamespace()
    search._chunking_config = {"enabled": True}

    assert search._resolve_search_level(None) == (None, True, None, None)


# --- a passage search that had to load the passage ChromaDB collection first ---


def _slow_first_load(search, seconds, reloaded=True):
    real = search.chroma_client.search

    def search_and_report(*args, **kwargs):
        results = real(*args, **kwargs)
        results["passage_load"] = {"seconds": seconds, "reloaded": reloaded, "idle_minutes": 10}
        return results

    search.chroma_client.search = search_and_report


def test_a_slow_passage_reload_is_explained(build):
    search = build(chunking=True, articles=["ART"], passages=["PAS#0"])
    _slow_first_load(search, 42.4)

    result = search.search("q", level="passages")

    assert _keys(result) == ["PAS"]
    note = result["load_note"]
    assert "took 42 s" in note
    assert "passage ChromaDB collection" in note
    assert "10 minutes without a passage search" in note
    assert "semantic_search.idle_unload_minutes" in note


def test_a_slow_first_passage_load_is_explained(build):
    search = build(chunking=True, articles=["ART"], passages=["PAS#0"])
    _slow_first_load(search, 12.0, reloaded=False)

    note = search.search("q", level="passages")["load_note"]

    assert "took 12 s" in note
    assert "loaded into memory first" in note
    assert "semantic_search.idle_unload_minutes" in note


def test_a_quick_passage_load_needs_no_explanation(build):
    search = build(chunking=True, articles=["ART"], passages=["PAS#0"])
    _slow_first_load(search, 1.5)

    assert search.search("q", level="passages")["load_note"] is None


def test_tool_prints_why_the_search_was_slow(tool_output):
    hit = {"item_key": "A", "similarity_score": 0.5, "matched_text": "text", "zotero_item": {"data": {"title": "T"}}}

    text = tool_output([hit], None, load_note="This search took 42 s because ...")

    assert "This search took 42 s because ..." in text


def test_tool_prints_why_the_search_was_slow_when_nothing_was_found(tool_output):
    text = tool_output([], None, load_note="This search took 42 s because ...")

    assert "No semantically similar items" in text
    assert "This search took 42 s because ..." in text


def test_a_search_reports_how_long_embedding_the_query_took(build):
    """So the worker can tell a slow load from a slow embedding API."""
    import time

    search = build(chunking=True, articles=["ART"], passages=["PAS#0"])
    client = search.chroma_client

    def slow_embedding(_query):
        time.sleep(0.05)
        return [1.0, 0.0]

    client.embedding_function = SimpleNamespace(embed_query_text=slow_embedding)

    results = client.search(query_texts=["q"], n_results=1, level="passages")

    assert results["embedding_seconds"] >= 0.05
    assert search.search("q", level="passages")["results"]  # the extra key breaks nothing
