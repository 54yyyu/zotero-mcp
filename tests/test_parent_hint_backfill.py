"""Records written before the parent hint get one without being embedded (#826).

``update-db`` (an explicit run) and ``migrate-index`` set ``parent_hint`` on
attachment records and passages that lack it, from the parents Zotero knows,
as a metadata-only update. An automatic run does no such bulk work.

A real ChromaDB index in a temp directory, an embedding function that records
what it embeds, and fake readers. HOME, Path.home and the config path all
point into tmp_path; no network.
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

from _scan_fakes import target  # noqa: E402
from chromadb.api.models.CollectionCommon import CollectionCommon  # noqa: E402
from test_item_keyed_local_scan import (  # noqa: E402
    DATE,
    Item,
    LibReader,
    env,  # noqa: F401  (the fixture)
)

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient, read_index_layout  # noqa: E402
from zotero_mcp.index_migration import LocalSources, migrate_index  # noqa: E402

NAME = "item_keyed_scan"
QUIET = {"progress": lambda _m: None}


class HintReader(LibReader):
    """A library that can say each attachment's parent, as ``LocalZoteroReader`` does."""

    def get_attachment_parents(self, *, group_id=None):
        parents = {}
        for item in self.items:
            for t in self.targets.get(item.item_id, []):
                parents[t.key] = item.key
        for row in self.standalone:
            parents[row[0]] = None
        return parents

    def close(self):
        pass


def _library():
    return HintReader(
        [Item("ITEM0001", 1)],
        targets={1: [target("PDFAAAAA")]},
        standalone=[("STANDALN", "storage:s.pdf", "application/pdf", DATE, 1)],
        standalone_targets={"STANDALN": target("STANDALN", stamp="2:200")},
    )


def _records(search):
    """``{id: metadata}`` of every record, in both collections."""
    client = search.chroma_client
    out = {}
    for name in (NAME, f"{NAME}_passages"):
        got = client.client.get_collection(name).get(include=["metadatas"])
        out.update(zip(got["ids"], got["metadatas"]))
    return out


def _strip_hints(search):
    client = search.chroma_client
    records = _records(search)
    ids = [i for i, m in records.items() if "parent_hint" in m]
    client.update_metadatas(ids, [{**records[i], "parent_hint": None} for i in ids])
    assert not [i for i, m in _records(search).items() if "parent_hint" in m]
    return ids


def _hints(search):
    return {i: m["parent_hint"] for i, m in _records(search).items() if "parent_hint" in m}


@pytest.fixture()
def hintless(env):  # noqa: F811
    """A library indexed, then every hint removed: what an index from before hints looks like."""
    reader = _library()
    search, _ = env(reader)
    stripped = _strip_hints(search)
    assert stripped, "the first run wrote hints"
    return env, reader, set(stripped)


def test_a_fresh_index_has_every_hint_and_no_gap(env):  # noqa: F811
    search, _ = env(_library())

    hints = _hints(search)
    assert hints["PDFAAAAA"] == "ITEM0001" and hints["STANDALN"] == ""
    assert {h for i, h in hints.items() if i.startswith("PDFAAAAA#")} == {"ITEM0001"}
    assert "ITEM0001" not in hints, "a bibliographic record has no hint"
    layout = read_index_layout(search.chroma_client.persist_directory, NAME)
    assert layout["attachments_without_parent_hint"] == 0
    assert layout["passages_without_parent_hint"] == 0
    assert layout["attachments_without_library"] == 0
    assert layout["library_ids"] == [0]


def test_the_layout_counts_the_records_that_lack_a_hint(hintless):
    run, reader, stripped = hintless
    search, _ = run(reader, update=False)

    layout = search.chroma_client.index_layout()

    assert layout["attachments_without_parent_hint"] == 2
    assert layout["passages_without_parent_hint"] == len([i for i in stripped if "#" in i])
    assert layout["passages_without_parent_hint"] > 0


def test_an_explicit_update_sets_the_missing_hints_without_embedding(hintless):
    run, reader, stripped = hintless

    search, stats = run(reader)

    assert search.chroma_client.embedded == []
    assert reader.extracted == []
    hints = _hints(search)
    assert set(hints) == stripped
    assert hints["PDFAAAAA"] == "ITEM0001" and hints["STANDALN"] == ""
    assert {h for i, h in hints.items() if i.startswith("PDFAAAAA#")} == {"ITEM0001"}
    assert {h for i, h in hints.items() if i.startswith("STANDALN#")} == {""}
    assert stats["parent_hints"] == {"attachments": 2, "records": len(stripped), "unresolved": 0}
    layout = search.chroma_client.index_layout()
    assert layout["attachments_without_parent_hint"] == 0 and layout["passages_without_parent_hint"] == 0


def test_an_automatic_update_does_not_backfill(hintless):
    run, reader, _stripped = hintless

    search, stats = run(reader, automatic=True)

    assert "parent_hints" not in stats
    assert _hints(search) == {}
    assert search.chroma_client.embedded == []


def test_a_backfill_leaves_an_attachment_zotero_does_not_know_without_a_hint(hintless):
    run, reader, _stripped = hintless
    search, _ = run(reader, update=False)
    search.chroma_client.upsert_embeddings(
        documents=["text"], metadatas=[{"item_key": "GONE0001", "group_id": 0, "file_stamp": "9:9"}],
        ids=["GONE0001"], embeddings=[[0.0, 1.0]],
    )
    stats = {}

    search._backfill_parent_hints(stats)

    assert "parent_hint" not in _records(search)["GONE0001"]
    assert stats["parent_hints"]["unresolved"] == 1
    assert _hints(search)["PDFAAAAA"] == "ITEM0001"


def test_a_backfill_does_not_touch_an_attachment_that_is_fully_hinted(hintless):
    run, reader, _stripped = hintless
    search, _ = run(reader, update=False)
    records = _records(search)
    ids = [i for i in records if i.split("#")[0] == "PDFAAAAA"]
    search.chroma_client.update_metadatas(ids, [{**records[i], "parent_hint": "OLDPAR01"} for i in ids])

    search._backfill_parent_hints({})

    # Re-parenting is the next rewrite's business; the backfill only fills gaps.
    assert _hints(search)["PDFAAAAA"] == "OLDPAR01"


# --- migrate-index ---------------------------------------------------------------


@pytest.fixture()
def refuse_embeddings(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("the backfill asked for an embedding")

    monkeypatch.setattr(CollectionCommon, "_embed", refuse)


def _seed_hintless_index(tmp_path):
    directory = str(tmp_path / "chroma")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {"persist_directory": directory, "embedding_model": "default"}}))
    client = ChromaClient(collection_name="zotero_library", persist_directory=directory)
    ids, metas = ["PDFAAAAA", "STANDALN"], []
    for key in ids:
        metas.append({"item_key": key, "group_id": 0, "date_modified": DATE, "file_stamp": "1:100"})
    for key in list(ids):
        for n in range(2):
            ids.append(f"{key}#{n}")
            metas.append({"item_key": key, "group_id": 0, "chunk_index": n, "n_chunks": 2})
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids], metadatas=metas, ids=ids, embeddings=[[0.0, 1.0]] * len(ids)
    )
    return directory, str(config), client


def _stored(client, name):
    got = client.client.get_collection(name).get(include=["metadatas"])
    return dict(zip(got["ids"], got["metadatas"]))


def test_migrate_index_sets_the_missing_hints_without_embedding(tmp_path, monkeypatch, refuse_embeddings):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    directory, config, client = _seed_hintless_index(tmp_path)
    assert read_index_layout(directory, "zotero_library")["attachments_without_parent_hint"] == 2

    report = migrate_index(config, sources=LocalSources(_library(), config), **QUIET)

    articles = _stored(client, "zotero_library")
    passages = _stored(client, "zotero_library_passages")
    assert articles["PDFAAAAA"]["parent_hint"] == "ITEM0001" and articles["STANDALN"]["parent_hint"] == ""
    assert {m["parent_hint"] for i, m in passages.items() if i.startswith("PDFAAAAA#")} == {"ITEM0001"}
    assert {m["parent_hint"] for i, m in passages.items() if i.startswith("STANDALN#")} == {""}
    assert report.parent_hints_missing == 2 and report.parent_hints_set == 6
    layout = read_index_layout(directory, "zotero_library")
    assert layout["attachments_without_parent_hint"] == 0 and layout["passages_without_parent_hint"] == 0


def test_a_migrate_index_dry_run_only_counts_the_missing_hints(tmp_path, monkeypatch, refuse_embeddings):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    directory, config, client = _seed_hintless_index(tmp_path)

    report = migrate_index(config, dry_run=True, sources=LocalSources(_library(), config), **QUIET)

    assert report.parent_hints_missing == 2 and report.parent_hints_set == 0
    assert "parent_hint" not in _stored(client, "zotero_library")["PDFAAAAA"]


def test_migrate_index_without_zotero_leaves_the_hints_for_an_update(tmp_path, monkeypatch, refuse_embeddings):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    directory, config, client = _seed_hintless_index(tmp_path)
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no web")))

    report = migrate_index(config, sources=LocalSources(None, config), **QUIET)

    assert report.parent_hints_missing == 2 and report.parent_hints_set == 0
    assert "parent_hint" not in _stored(client, "zotero_library")["PDFAAAAA"]


# --- web mode: the parents come from the Zotero API --------------------------------


class _ApiClient:
    """pyzotero stand-in: one library's attachments, paged by ``start``/``limit``."""

    def __init__(self, library_id, rows, fail=False):
        self.library_type = "users" if library_id == "0" else "groups"
        self.library_id = library_id
        self.rows = rows
        self.fail = fail
        self.calls = []

    def items(self, start=0, limit=100, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("503")
        return self.rows[start:start + limit]


def _row(key, parent=None):
    data = {"key": key, "itemType": "attachment"}
    if parent:
        data["parentItem"] = parent
    return {"key": key, "data": data}


class _KnowsLibraries:
    def __init__(self, groups):
        self.groups = groups

    def item_group_ids(self, keys):
        return {k: self.groups[k] for k in keys if k in self.groups}


def _web_search(monkeypatch, tmp_path, personal, group, groups):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: personal)
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=_KnowsLibraries(groups), config_path=str(tmp_path / "config.json")
    )
    search._scoped_clients[777] = group
    return search


def test_web_mode_lists_each_librarys_attachments_once_and_reads_the_parents(monkeypatch, tmp_path):
    personal = _ApiClient("0", [_row("ATTPERS1", "PARPERS1"), _row("STANDALN")])
    group = _ApiClient("777", [_row("ATTGRP01", "PARGRP01")])
    search = _web_search(monkeypatch, tmp_path, personal, group, {"ATTPERS1": 0, "STANDALN": 0, "ATTGRP01": 777})

    parents = search._attachment_parents(["ATTPERS1", "STANDALN", "ATTGRP01", "GONE0001"])

    assert parents == {"ATTPERS1": "PARPERS1", "STANDALN": None, "ATTGRP01": "PARGRP01"}
    assert [c for c in personal.calls] == [{"itemType": "attachment"}] and len(group.calls) == 1


def test_web_mode_skips_a_library_it_cannot_list(monkeypatch, tmp_path):
    personal = _ApiClient("0", [_row("ATTPERS1", "PARPERS1")])
    group = _ApiClient("777", [], fail=True)
    search = _web_search(monkeypatch, tmp_path, personal, group, {"ATTPERS1": 0, "ATTGRP01": 777})

    assert search._attachment_parents(["ATTPERS1", "ATTGRP01"]) == {"ATTPERS1": "PARPERS1"}
