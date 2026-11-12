"""Vectors are reused by file hash (#495, variant A; part of #826).

A file whose bytes are already indexed under another attachment key, or come
earlier in the same run, is copied from that attachment's records: no parsing,
no embedding call. The copy is an independent record set under the new key,
with its own ``item_key``, ``group_id``, ``date_modified`` and ``file_stamp``.

A real ChromaDB index in a temp directory, an embedding function that
records what it embeds and gives each text its own vector, and fake readers
over real files in tmp_path. HOME, Path.home and the config path all point
into tmp_path; no network.
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
from test_item_keyed_local_scan import FakeZotero, Item, LibReader  # noqa: E402

from zotero_mcp import file_hash, semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
from zotero_mcp.local_db import ExtractionTarget  # noqa: E402

NAME = "item_keyed_reuse"
BODY = "Body sentence"
TEXT = " ".join(f"{BODY} {i} of the paper." for i in range(100))
PASSAGES = 5


class DistinctEF(EmbeddingFunction):
    """A different vector for every text, so a copy is told from a re-embedding."""

    def __init__(self):
        self.embedded: list[str] = []

    def __call__(self, input):
        self.embedded.extend(input)
        return [[float(len(t)), float(sum(map(ord, t)) % 1009)] for t in input]

    def truncate(self, text, max_tokens):
        return text


class Client(ChromaClient):
    def _create_embedding_function(self):
        return DistinctEF()

    @property
    def embedded(self):
        return self.embedding_function.embedded

    @property
    def bodies(self):
        return [t for t in self.embedded if BODY in t]


def real_target(tmp_path, key, content=b"the same bytes", stamp="1:100"):
    path = tmp_path / "files" / f"{key}.pdf"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return ExtractionTarget(
        key=key, path=path, content_type="application/pdf", category="pdf", file_stamp=stamp, size=len(content)
    )


class Reader(LibReader):
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups or {}

    def get_key_group_map(self):
        return ({key: self.groups.get(key, 0) for key in self.all_keys()}, set())


@pytest.fixture()
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    return tmp_path


def make_env(monkeypatch, tmp_path, chunking=True):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
                    "chunking": {
                        "enabled": chunking,
                        "chunk_size": 400,
                        "overlap": 50,
                        "max_chunks_per_item": 5,
                    },
                }
            }
        )
    )

    def run(reader, **kwargs):
        monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: reader)
        monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: FakeZotero(reader))
        client = Client(collection_name=NAME, persist_directory=str(tmp_path / "chroma"))
        search = semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))
        reader.extracted.clear()
        stats = search.update_database(extract_fulltext=True, **kwargs)
        assert stats.get("error") is None, stats
        return search, stats

    run.config_path = str(config_path)
    return run


@pytest.fixture()
def env(monkeypatch, home):
    return make_env(monkeypatch, home)


def two_items(tmp_path, **kwargs):
    return Reader(
        [Item("ITEM0001", 1), Item("ITEM0002", 2)],
        targets={1: [real_target(tmp_path, "ATT00001")], 2: [real_target(tmp_path, "ATT00002")]},
        **kwargs,
    )


def records(search, key):
    """``{id: (document, vector, metadata)}`` of a record and its passages."""
    out = {}
    existing = {c.name for c in search.chroma_client.client.list_collections()}
    for name in (NAME, f"{NAME}_passages"):
        if name not in existing:
            continue
        got = search.chroma_client.client.get_collection(name).get(
            where={"item_key": key}, include=["documents", "embeddings", "metadatas"]
        )
        for i, doc_id in enumerate(got["ids"]):
            out[doc_id] = (got["documents"][i], [float(x) for x in got["embeddings"][i]], got["metadatas"][i])
    return out


def ids(search, name=NAME):
    return set(search.chroma_client.client.get_collection(name).get(include=[])["ids"])


# --- the realtime path --------------------------------------------------------------


def test_the_same_file_under_two_items_is_parsed_and_embedded_once(env, home):
    reader = two_items(home, groups={"ITEM0002": 7})

    search, stats = env(reader)

    assert reader.extracted == ["ATT00001"], "the second copy of the file is not parsed"
    assert len(search.chroma_client.bodies) == 1 + PASSAGES, "one article and its passages, once"
    first, second = records(search, "ATT00001"), records(search, "ATT00002")
    assert set(second) == {"ATT00002"} | {f"ATT00002#{i}" for i in range(PASSAGES)}
    assert second["ATT00002"][1] == first["ATT00001"][1]
    assert second["ATT00002"][0] == first["ATT00001"][0]
    for i in range(PASSAGES):
        assert second[f"ATT00002#{i}"][1] == first[f"ATT00001#{i}"][1]
        assert second[f"ATT00002#{i}"][0] == first[f"ATT00001#{i}"][0]
    own = second["ATT00002"][2]
    assert own["item_key"] == "ATT00002" and own["group_id"] == 7
    assert own["file_hash"] == first["ATT00001"][2]["file_hash"]
    assert own["file_stamp"] == "1:100"
    passage = second["ATT00002#3"][2]
    assert passage["item_key"] == "ATT00002" and passage["group_id"] == 7
    assert (passage["chunk_index"], passage["n_chunks"]) == (3, PASSAGES)
    assert passage["char_start"] == first["ATT00001#3"][2]["char_start"]
    assert first["ATT00001#3"][2]["group_id"] == 0
    assert stats["reused_from_hash"] == {"attachments": 1, "records": 1 + PASSAGES}


def test_a_file_already_in_the_index_is_copied_in_a_later_run(env, home):
    reader = Reader([Item("ITEM0001", 1)], targets={1: [real_target(home, "ATT00001")]})
    env(reader)

    reader.items.append(Item("ITEM0002", 2))
    reader.targets[2] = [real_target(home, "ATT00002")]
    search, stats = env(reader)

    assert reader.extracted == [], "nothing is parsed"
    assert [t for t in search.chroma_client.embedded if BODY in t] == []
    assert len(records(search, "ATT00002")) == 1 + PASSAGES
    assert stats["reused_from_hash"]["attachments"] == 1
    assert stats["added_items"] >= 2  # the new item's record and its copy


def test_a_file_added_again_to_the_same_item_is_the_only_one_hashed(env, home, monkeypatch):
    reader = Reader([Item("ITEM0001", 1)], targets={1: [real_target(home, "ATT00001")]})
    env(reader)

    hashed = []
    real = file_hash.get_file_hash
    monkeypatch.setattr(file_hash, "get_file_hash", lambda key, *a, **kw: hashed.append(key) or real(key, *a, **kw))
    reader.targets[1] = [real_target(home, "ATT00001"), real_target(home, "ATT00009")]
    search, _ = env(reader)

    assert hashed == ["ATT00009"], "the file already indexed is not read again"
    assert reader.extracted == []
    assert set(records(search, "ATT00009")) == {"ATT00009"} | {f"ATT00009#{i}" for i in range(PASSAGES)}


def test_a_file_with_an_unchanged_stamp_is_not_hashed_again(env, home, monkeypatch):
    reader = Reader([Item("ITEM0001", 1)], targets={1: [real_target(home, "ATT00001")]})
    search, _ = env(reader)
    search.chroma_client.delete_item_chunks("ATT00001")
    search.chroma_client.delete_documents(["ATT00001"])  # planned again, same stamp

    opened = []
    real_open = open

    def spy(path, *args, **kwargs):
        if str(path).endswith("ATT00001.pdf"):
            opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(file_hash, "open", spy, raising=False)
    env(reader)

    assert opened == []
    assert reader.extracted == ["ATT00001"]


def test_a_source_missing_a_passage_is_not_copied(env, home):
    reader = Reader([Item("ITEM0001", 1)], targets={1: [real_target(home, "ATT00001")]})
    search, _ = env(reader)
    search.chroma_client.delete_documents(["ATT00001#2"])

    reader.items.append(Item("ITEM0002", 2))
    reader.targets[2] = [real_target(home, "ATT00002")]
    search, stats = env(reader)

    assert "ATT00002" in reader.extracted, "an incomplete source is no source"
    assert stats["reused_from_hash"]["attachments"] == 0
    assert len(records(search, "ATT00002")) == 1 + PASSAGES


def test_a_source_with_a_different_hash_is_not_copied(env, home):
    reader = two_items(home)
    reader.targets[2] = [real_target(home, "ATT00002", content=b"other bytes")]

    search, _ = env(reader)

    assert sorted(reader.extracted) == ["ATT00001", "ATT00002"]
    assert len(search.chroma_client.bodies) == 2 * (1 + PASSAGES)


def test_deleting_the_source_keeps_the_copy(env, home):
    reader = two_items(home)
    env(reader)

    reader.targets[1] = []
    reader.unselected = {"ATT00001"}
    search, stats = env(reader)

    assert stats["deleted_items"] == 1
    assert records(search, "ATT00001") == {}
    assert len(records(search, "ATT00002")) == 1 + PASSAGES


def test_with_chunking_off_only_the_attachment_record_is_copied(monkeypatch, home):
    env = make_env(monkeypatch, home, chunking=False)
    reader = two_items(home)

    search, stats = env(reader)

    assert reader.extracted == ["ATT00001"]
    assert len(search.chroma_client.bodies) == 1
    assert set(records(search, "ATT00002")) == {"ATT00002"}
    assert stats["reused_from_hash"] == {"attachments": 1, "records": 1}


def test_a_copy_replaces_the_stale_passages_of_its_key(env, home):
    reader = Reader(
        [Item("ITEM0001", 1)],
        targets={1: [real_target(home, "ATT00001", content=b"old bytes")]},
        texts={"ATT00001": (TEXT, "pdf")},
    )
    search, _ = env(reader)
    reader.items.append(Item("ITEM0002", 2))
    reader.targets[2] = [real_target(home, "ATT00002", content=b"new bytes")]
    reader.texts["ATT00002"] = (TEXT[:600], "pdf")  # a shorter text: fewer passages
    env(reader)
    short = len(records(search, "ATT00002")) - 1

    # ATT00001's file now has ATT00002's bytes: it copies, and loses its extra passages.
    reader.targets[1] = [real_target(home, "ATT00001", content=b"new bytes", stamp="2:200")]
    search, _ = env(reader)

    assert reader.extracted == []
    assert len(records(search, "ATT00001")) == 1 + short
    assert records(search, "ATT00001")["ATT00001"][2]["file_stamp"] == "2:200"


def test_no_file_hash_means_no_reuse(env, home):
    reader = two_items(home)
    for targets in reader.targets.values():
        for t in targets:
            t.path.unlink()  # unreadable: no hash, and nothing to read

    search, stats = env(reader)

    assert stats["reused_from_hash"]["attachments"] == 0
    assert sorted(reader.extracted) == ["ATT00001", "ATT00002"]


def test_a_copy_whose_source_failed_is_left_for_the_next_run(env, home):
    reader = two_items(home)
    reader.texts = {"ATT00001": None}  # the first file yields nothing

    search, stats = env(reader)

    assert records(search, "ATT00001") == {} and records(search, "ATT00002") == {}
    assert stats["reuse_left_for_next_run"] == 1

    # Next run: the failed file is not retried (marker), the other is extracted and indexed.
    search, stats = env(reader)
    assert reader.extracted == ["ATT00002"]
    assert len(records(search, "ATT00002")) == 1 + PASSAGES


def test_mark_reusable_picks_one_unit_per_hash_and_copies_the_others(env, home):
    search, _ = env(
        Reader(
            [],
        )
    )
    units = [
        semantic_search._attachment_unit(key, "text", file_stamp="1:1", file_hash=value)
        for key, value in (("AAAAAAAA", "h1"), ("BBBBBBBB", "h1"), ("CCCCCCCC", "h2"), ("DDDDDDDD", "h1"))
    ]

    search._mark_reusable(units, look_in_index=True)

    assert [u.get("reuse_from") for u in units] == [None, "AAAAAAAA", None, "AAAAAAAA"]
    assert [bool(u.get("reuse_deferred")) for u in units] == [False, True, False, True]


def test_api_units_with_a_hash_are_copied_without_embedding(env, home, monkeypatch):
    """The units of a run over the Zotero API carry a hash when the file is local."""
    reader = Reader([Item("ITEM0001", 1)], targets={1: [real_target(home, "ATT00001")]})
    search, _ = env(reader)
    sha = records(search, "ATT00001")["ATT00001"][2]["file_hash"]
    search.chroma_client.embedded.clear()
    unit = semantic_search._attachment_unit(
        "ATT00005", TEXT, file_stamp="3:3", file_hash=sha, group_id=0, date_modified="2024-01-01"
    )

    search._run_hash_first = {}
    search._mark_reusable([unit], look_in_index=True)
    stats = {
        "processed_items": 0,
        "added_items": 0,
        "updated_items": 0,
        "reused_from_hash": {"attachments": 0, "records": 0},
    }
    left = search._copy_reused_units([unit], stats)

    assert unit["reuse_from"] == "ATT00001" and left == []
    assert search.chroma_client.embedded == []
    assert len(records(search, "ATT00005")) == 1 + PASSAGES
    assert stats["added_items"] == 1


# --- the hash cache -----------------------------------------------------------------


def test_many_hashes_in_a_batched_block_write_the_cache_once(home, monkeypatch):
    config = str(home / "config.json")
    writes = []
    real = file_hash.write_json_atomic
    monkeypatch.setattr(file_hash, "write_json_atomic", lambda path, data: writes.append(path) or real(path, data))
    paths = []
    for i in range(20):
        path = home / f"f{i}.bin"
        path.write_bytes(bytes([i]))
        paths.append(path)

    with file_hash.batched(config):
        digests = [file_hash.get_file_hash(f"K{i:07d}", p, "1:1", config_path=config) for i, p in enumerate(paths)]
        with file_hash.batched(config):  # reentrant
            file_hash.get_file_hash("K9999999", paths[0], "1:1", config_path=config)
        assert writes == [], "nothing is written while the block is open"

    assert len(writes) == 1
    # A later call is served from the saved cache, without opening the file.
    paths[3].unlink()
    assert file_hash.get_file_hash("K0000003", paths[3], "1:1", config_path=config) == digests[3]


def test_a_batched_block_saves_what_it_hashed_when_it_raises(home):
    config = str(home / "config.json")
    path = home / "f.bin"
    path.write_bytes(b"x")
    with pytest.raises(RuntimeError), file_hash.batched(config):
        file_hash.get_file_hash("K0000001", path, "1:1", config_path=config)
        raise RuntimeError("stopped")

    path.unlink()
    assert file_hash.get_file_hash("K0000001", path, "1:1", config_path=config)


def test_a_scan_writes_the_hash_cache_once_per_slice(env, home, monkeypatch):
    writes = []
    real = file_hash.write_json_atomic
    monkeypatch.setattr(file_hash, "write_json_atomic", lambda path, data: writes.append(path) or real(path, data))
    items = [Item(f"ITEM{i:04d}", i + 1) for i in range(12)]
    reader = Reader(
        items,
        targets={i + 1: [real_target(home, f"ATT{i:05d}", content=bytes([i]))] for i in range(12)},
    )

    env(reader)

    assert len(writes) == 1, "12 files hashed, one write"


# --- the Batch API ------------------------------------------------------------------


class FakeBatch:
    """Stands in for the provider module: keeps the records, 'embeds' them locally."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "batches"
        self.root.mkdir()
        self.submitted: list[dict] = []
        self.manifest = None

    def submit(self, records, **kwargs):
        self.submitted.extend(records)
        records_path = self.root / "batch-001-records.jsonl"
        semantic_search.openai_batch.write_jsonl(records_path, records)
        out = self.root / "batch-001-records-output.jsonl"
        out.write_text(
            "".join(
                json.dumps(
                    {
                        "custom_id": r["id"],
                        "response": {
                            "status_code": 200,
                            "body": {"data": [{"embedding": [float(len(r["document"])), 1.0]}]},
                        },
                    }
                )
                + "\n"
                for r in records
            ),
            encoding="utf-8",
        )
        self.manifest = {
            "provider": "openai",
            "run_id": "run-1",
            "manifest_path": str(self.root / "manifest.json"),
            "force_full_rebuild": False,
            "target_sync_version": None,
            "batches": [
                {
                    "batch_id": "batch-1",
                    "status": "completed",
                    "output_file_id": "file-1",
                    "records_path": str(records_path),
                    "imported_at": None,
                }
            ],
        }
        return self.manifest


def batch_search(home, monkeypatch, fake):
    config_path = home / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
                    "chunking": {"enabled": True, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
                }
            }
        )
    )
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search.openai_batch, "submit_embedding_batches", fake.submit)
    monkeypatch.setattr(semantic_search.openai_batch, "find_manifest", lambda **kw: fake.manifest)
    monkeypatch.setattr(semantic_search.openai_batch, "refresh_manifest_status", lambda m, **kw: m)
    monkeypatch.setattr(semantic_search.openai_batch, "create_openai_client", lambda config: object())
    monkeypatch.setattr(semantic_search.openai_batch, "save_manifest", lambda m: None)
    client = Client(collection_name=NAME, persist_directory=str(home / "chroma"))
    return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))


def file_unit(key, sha, text=TEXT, stamp="1:100"):
    return semantic_search._attachment_unit(
        key, text, file_stamp=stamp, file_hash=sha, group_id=0, date_modified="2024-01-01"
    )


def stats_dict():
    return {
        "processed_items": 0,
        "added_items": 0,
        "updated_items": 0,
        "skipped_items": 0,
        "errors": 0,
        "reused_from_hash": {"attachments": 0, "records": 0},
    }


def test_batch_submits_the_first_of_identical_files_and_imports_the_copies(home, monkeypatch):
    fake = FakeBatch(home)
    search = batch_search(home, monkeypatch, fake)
    units = [file_unit("ATT00001", "h1"), file_unit("ATT00002", "h1"), file_unit("ATT00003", "h2", text=TEXT[:700])]
    search._run_hash_first = {}
    search._mark_reusable(units, look_in_index=True)
    reuse = [u for u in units if u.get("reuse_from")]
    rest = [u for u in units if not u.get("reuse_from")]

    stats = search._submit_batch_index("openai", rest, False, None, stats_dict(), reuse_units=reuse)

    assert {r["id"].split("#")[0] for r in fake.submitted} == {"ATT00001", "ATT00003"}
    assert fake.manifest["reuse"] == [
        {
            "key": "ATT00002",
            "reuse_from": "ATT00001",
            "group_id": 0,
            "date_modified": "2024-01-01",
            "file_stamp": "1:100",
            "file_hash": "h1",
        }
    ]
    assert stats["batch_reuse_entries"] == 1

    imported = search.import_openai_batch()

    assert imported["reused_from_hash"] == {"attachments": 1, "records": 1 + PASSAGES}
    assert fake.manifest["reuse"][0]["state"] == "copied"
    first, second = records(search, "ATT00001"), records(search, "ATT00002")
    assert second["ATT00002"][1] == first["ATT00001"][1]
    assert second["ATT00002#2"][1] == first["ATT00001#2"][1]
    assert second["ATT00002"][2]["item_key"] == "ATT00002"


def test_batch_does_not_submit_a_file_already_in_the_index(home, monkeypatch):
    fake = FakeBatch(home)
    search = batch_search(home, monkeypatch, fake)
    search._submit_batch_index("openai", [file_unit("ATT00001", "h1")], False, None, stats_dict())
    search.import_openai_batch()
    fake.submitted.clear()

    other = file_unit("ATT00002", "h1")
    third = file_unit("ATT00003", "h3", text=TEXT[:700])
    search._run_hash_first = {}
    search._mark_reusable([other, third], look_in_index=True)
    assert other["reuse_from"] == "ATT00001" and "reuse_deferred" not in other
    stats = search._submit_batch_index("openai", [third], False, None, stats_dict(), reuse_units=[other])

    assert {r["id"].split("#")[0] for r in fake.submitted} == {"ATT00003"}
    assert stats["batch_reuse_entries"] == 1
    search.import_openai_batch()
    assert len(records(search, "ATT00002")) == 1 + PASSAGES


def test_batch_with_only_copies_copies_them_at_once(home, monkeypatch):
    fake = FakeBatch(home)
    search = batch_search(home, monkeypatch, fake)
    search._submit_batch_index("openai", [file_unit("ATT00001", "h1")], False, None, stats_dict())
    search.import_openai_batch()
    fake.submitted.clear()
    fake.manifest = None
    other = file_unit("ATT00002", "h1")
    search._run_hash_first = {}
    search._mark_reusable([other], look_in_index=True)

    stats = search._submit_batch_index("openai", [], False, None, stats_dict(), reuse_units=[other])

    assert fake.submitted == [] and fake.manifest is None
    assert stats["batch_nothing_to_embed"] is True
    assert stats["reused_from_hash"]["attachments"] == 1
    assert len(records(search, "ATT00002")) == 1 + PASSAGES


def test_batch_import_logs_and_skips_a_copy_whose_source_never_landed(home, monkeypatch, caplog):
    fake = FakeBatch(home)
    search = batch_search(home, monkeypatch, fake)
    units = [file_unit("ATT00001", "h1"), file_unit("ATT00002", "h1")]
    search._run_hash_first = {}
    search._mark_reusable(units, look_in_index=True)
    search._submit_batch_index("openai", units[:1], False, None, stats_dict(), reuse_units=units[1:])
    # The provider returned nothing for the source's records.
    (home / "batches" / "batch-001-records-output.jsonl").write_text("", encoding="utf-8")

    with caplog.at_level("WARNING"):
        imported = search.import_openai_batch()

    assert imported["reused_from_hash"]["attachments"] == 0
    assert records(search, "ATT00002") == {}
    assert fake.manifest["reuse"][0]["state"] == "source_missing"
    assert any("ATT00002" in m and "ATT00001" in m for m in caplog.messages)
