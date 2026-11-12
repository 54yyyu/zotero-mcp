"""The API paths write the item-keyed shape (#826).

Over the Zotero API (web, or the local API) every file the attachment
priority selects becomes an attachment record under its own key, standalone
files included, in full scans and in incremental runs. Text comes from the
attachment's fulltext endpoint, and only for an attachment whose record is not
already current.

A real ChromaDB index in a temp directory, an embedding function that records
what it embeds, and a fake Zotero API client that counts its calls. HOME,
Path.home and the config path all point into tmp_path.
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
from pyzotero.zotero_errors import ResourceNotFoundError  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402

NAME = "item_keyed_api"
DATE = "2024-01-01T00:00:00Z"
TEXT = " ".join(f"Body sentence {i} of the paper." for i in range(100))


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


def paper(key, date=DATE, version=1):
    return {
        "key": key,
        "version": version,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": f"Paper {key}",
            "abstractNote": "An abstract.",
            "creators": [{"creatorType": "author", "firstName": "A", "lastName": "Author"}],
            "dateAdded": DATE,
            "dateModified": date,
        },
    }


def attachment(
    key,
    parent=None,
    ctype="application/pdf",
    filename=None,
    md5=None,
    mtime="100",
    date=DATE,
    version=1,
    deleted=False,
    link_mode="imported_file",
):
    data = {
        "key": key,
        "itemType": "attachment",
        "title": filename or key,
        "contentType": ctype,
        "filename": filename or (f"{key}.pdf" if ctype == "application/pdf" else f"{key}.html"),
        "linkMode": link_mode,
        "md5": md5 or f"md5{key}",
        "mtime": mtime,
        "dateModified": date,
    }
    if parent:
        data["parentItem"] = parent
    if deleted:
        data["deleted"] = 1
    return {"key": key, "version": version, "data": data}


class FakeZotero:
    """A library that counts the calls made to it."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.texts: dict[str, str] = {}
        self.version = 5
        self.calls: list[tuple] = []
        # Whether item_versions() still lists trashed items.
        self.trashed_in_versions = False

    def put(self, *rows, version=None):
        version = version or self.version
        for row in rows:
            row["version"] = version
            self.rows[row["key"]] = row
        self.version = max(self.version, version)

    def live(self):
        return [r for r in self.rows.values() if not r["data"].get("deleted")]

    def items(self, start=0, limit=100, **kwargs):
        self.calls.append(("items", kwargs.get("itemType")))
        rows = self.live()
        if kwargs.get("itemType") == "attachment":
            rows = [r for r in rows if r["data"]["itemType"] == "attachment"]
        return rows[start : start + limit]

    def item(self, key):
        self.calls.append(("item", key))
        return self.rows[key]

    def children(self, key, start=0, limit=25, **kwargs):
        self.calls.append(("children", key))
        kids = [r for r in self.live() if r["data"].get("parentItem") == key]
        return kids[int(start) : int(start) + int(limit)]

    def fulltext_item(self, key):
        self.calls.append(("fulltext_item", key))
        if key not in self.texts:
            raise ResourceNotFoundError("no text")
        return {"content": self.texts[key]}

    def item_versions(self, since=None, **kwargs):
        self.calls.append(("item_versions", since))
        rows = self.rows.values() if self.trashed_in_versions else self.live()
        return {r["key"]: r["version"] for r in rows if since is None or r["version"] > since}

    def last_modified_version(self, **kwargs):
        return self.version

    def fulltext_keys(self):
        return [c[1] for c in self.calls if c[0] == "fulltext_item"]


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    config_path = tmp_path / "config.json"

    def write_config(priority=None):
        extraction = {"attachment_priority": priority} if priority else {}
        config_path.write_text(
            json.dumps(
                {
                    "semantic_search": {
                        "index_schema_version": semantic_search._INDEX_SCHEMA_VERSION,
                        "chunking": {"enabled": True, "chunk_size": 400, "overlap": 50, "max_chunks_per_item": 5},
                        "extraction": extraction,
                    }
                }
            )
        )

    write_config()
    persist = str(tmp_path / "chroma")

    class Env:
        zot = FakeZotero()
        config = str(config_path)
        set_priority = staticmethod(write_config)

        def search(self):
            monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: self.zot)
            client = RecordingClient(collection_name=NAME, persist_directory=persist)
            return semantic_search.ZoteroSemanticSearch(chroma_client=client, config_path=str(config_path))

        def forget_sync_version(self):
            """The next run is a full scan again."""
            config = json.loads(Path(self.config).read_text())
            for key in ("last_sync_version", "last_sync_versions"):
                config["semantic_search"].pop(key, None)
            Path(self.config).write_text(json.dumps(config))

        def update(self, **kwargs):
            search = self.search()
            self.zot.calls.clear()
            stats = search.update_database(**kwargs)
            assert stats.get("error") is None, stats
            self.last = search
            return search, stats

    return Env()


def ids(search, name=NAME):
    try:
        return set(search.chroma_client.client.get_collection(name).get(include=[])["ids"])
    except Exception:  # no such collection: the index has no separate passages
        return set()


def all_ids(search):
    return ids(search) | ids(search, f"{NAME}_passages")


def records(search):
    """Record ids that are not passages."""
    return {i for i in all_ids(search) if "#" not in i}


def meta(search, doc_id):
    return search.chroma_client.get_document_metadata(doc_id)


def library(env, *rows, **texts):
    env.zot.put(*rows)
    env.zot.texts.update(texts)


# --- selection ----------------------------------------------------------------


def test_every_selected_pdf_is_an_attachment_record_and_the_snapshot_is_skipped(env):
    library(
        env,
        paper("ITEM0001"),
        attachment("PDFAAAAA", "ITEM0001"),
        attachment("PDFBBBBB", "ITEM0001"),
        attachment("SNAPSHOT", "ITEM0001", ctype="text/html"),
        PDFAAAAA=TEXT,
        PDFBBBBB=TEXT,
        SNAPSHOT=TEXT,
    )

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA", "PDFBBBBB"}
    assert sorted(env.zot.fulltext_keys()) == ["PDFAAAAA", "PDFBBBBB"]
    assert meta(search, "PDFAAAAA")["file_stamp"] == "api:md5PDFAAAAA:100"
    assert "file_stamp" not in meta(search, "ITEM0001")


def test_a_snapshot_is_indexed_when_the_priority_prefers_html(env):
    env.set_priority(["html", "pdf", "other"])
    library(
        env,
        paper("ITEM0001"),
        attachment("PDFAAAAA", "ITEM0001"),
        attachment("SNAPSHOT", "ITEM0001", ctype="text/html"),
        PDFAAAAA=TEXT,
        SNAPSHOT=TEXT,
    )

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "SNAPSHOT"}


def test_an_attachment_without_text_gets_no_record_and_is_asked_again_by_a_full_scan(env):
    """Zotero extracts text after the file arrives, with the stamp unchanged: no failure marker."""
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"))
    search, _stats = env.update()
    assert env.zot.fulltext_keys() == ["PDFAAAAA"]
    assert records(search) == {"ITEM0001"}
    env.zot.texts["PDFAAAAA"] = TEXT
    env.forget_sync_version()

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}


# --- standalone attachments -------------------------------------------------------


def test_a_standalone_pdf_is_indexed_in_a_full_scan(env):
    library(env, paper("ITEM0001"), attachment("LONEPDF1"), LONEPDF1=TEXT)

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "LONEPDF1"}
    assert meta(search, "LONEPDF1")["file_stamp"] == "api:md5LONEPDF1:100"


def test_a_standalone_pdf_is_indexed_by_an_incremental_run(env):
    library(env, paper("ITEM0001"))
    env.update()
    env.zot.put(attachment("LONEPDF1"), version=6)
    env.zot.texts["LONEPDF1"] = TEXT

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "LONEPDF1"}
    assert env.zot.fulltext_keys() == ["LONEPDF1"]


def test_a_full_scan_keeps_standalone_records_and_deletes_a_trashed_one(env):
    library(env, paper("ITEM0001"), attachment("LONEPDF1"), attachment("LONEPDF2"), LONEPDF1=TEXT, LONEPDF2=TEXT)
    env.update()
    env.zot.rows["LONEPDF2"]["data"]["deleted"] = 1
    env.forget_sync_version()

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "LONEPDF1"}
    assert not any(i.startswith("LONEPDF2") for i in all_ids(search))


# --- incremental runs ---------------------------------------------------------------


def test_a_new_pdf_on_an_existing_item_embeds_only_that_file(env):
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)
    env.update()
    env.zot.put(attachment("PDFBBBBB", "ITEM0001"), version=6)
    env.zot.texts["PDFBBBBB"] = TEXT + " extra"

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA", "PDFBBBBB"}
    assert env.zot.fulltext_keys() == ["PDFBBBBB"]


def test_a_replaced_pdf_is_embedded_again(env):
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)
    env.update()
    env.zot.put(attachment("PDFAAAAA", "ITEM0001", md5="newmd5", mtime="200", date="2024-02-01T00:00:00Z"), version=6)
    env.zot.texts["PDFAAAAA"] = "Replacement text " + TEXT

    search, _stats = env.update()

    assert meta(search, "PDFAAAAA")["file_stamp"] == "api:newmd5:200"
    assert env.zot.fulltext_keys() == ["PDFAAAAA"]
    stored = search.chroma_client.client.get_collection(NAME).get(ids=["PDFAAAAA"])["documents"][0]
    assert stored.startswith("Replacement text")


def test_an_outranking_pdf_removes_the_snapshot_records(env):
    library(env, paper("ITEM0001"), attachment("SNAPSHOT", "ITEM0001", ctype="text/html"), SNAPSHOT=TEXT)
    search, _stats = env.update()
    assert records(search) == {"ITEM0001", "SNAPSHOT"}
    env.zot.put(attachment("PDFAAAAA", "ITEM0001"), version=6)
    env.zot.texts["PDFAAAAA"] = TEXT
    env.forget_sync_version()

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}
    assert not any(i.startswith("SNAPSHOT") for i in all_ids(search))


def test_a_trashed_attachment_loses_its_records(env):
    library(
        env,
        paper("ITEM0001"),
        attachment("PDFAAAAA", "ITEM0001"),
        attachment("PDFBBBBB", "ITEM0001"),
        PDFAAAAA=TEXT,
        PDFBBBBB=TEXT,
    )
    env.update()
    # Trashed, but still listed by item_versions(since=...) and fetchable.
    env.zot.rows["PDFBBBBB"]["data"]["deleted"] = 1
    env.zot.rows["PDFBBBBB"]["version"] = 6
    env.zot.version = 6
    env.zot.trashed_in_versions = True

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}
    assert not any(i.startswith("PDFBBBBB") for i in all_ids(search))


def test_an_attachment_moved_between_parents_writes_nothing_when_the_selection_holds(env):
    library(
        env,
        paper("ITEM0001"),
        paper("ITEM0002"),
        attachment("PDFAAAAA", "ITEM0001"),
        attachment("PDFXXXXX", "ITEM0002"),
        PDFAAAAA=TEXT,
        PDFXXXXX=TEXT,
    )
    search, _stats = env.update()
    before = records(search)
    env.zot.put(attachment("PDFAAAAA", "ITEM0002"), version=6)

    search, _stats = env.update()

    assert records(search) == before
    assert search.chroma_client.embedded == []
    assert env.zot.fulltext_keys() == []


def test_unchanged_files_cost_no_fulltext_requests(env):
    library(
        env,
        paper("ITEM0001"),
        attachment("PDFAAAAA", "ITEM0001"),
        attachment("LONEPDF1"),
        PDFAAAAA=TEXT,
        LONEPDF1=TEXT,
    )
    env.update()
    env.forget_sync_version()

    search, _stats = env.update()

    assert env.zot.fulltext_keys() == []
    assert search.chroma_client.embedded == []


# --- legacy records ---------------------------------------------------------------------


def _legacy_record(search, key, date=DATE):
    search.chroma_client.upsert_documents(
        ["Paper with its text"],
        [{"item_key": key, "title": "Paper", "group_id": 0, "date_modified": date, "has_fulltext": True}],
        [key],
    )


def test_an_unchanged_legacy_item_writes_nothing(env):
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)
    search = env.search()
    _legacy_record(search, "ITEM0001")

    search, _stats = env.update()

    assert env.zot.fulltext_keys() == []
    assert search.chroma_client.embedded == []
    assert records(search) == {"ITEM0001"}
    assert "title" in meta(search, "ITEM0001")


def test_a_changed_legacy_item_is_rewritten_in_the_new_shape(env):
    library(env, paper("ITEM0001", date="2024-03-01T00:00:00Z"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)
    search = env.search()
    _legacy_record(search, "ITEM0001")

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}
    assert "title" not in meta(search, "ITEM0001")


# --- file stamps ----------------------------------------------------------------------------


class LocalFiles:
    """The part of LocalZoteroReader the API paths use to find a file on this machine."""

    def __init__(self, files):
        self.files = files

    def resolve_attachment_file(self, key):
        return self.files.get(key)

    def close(self):
        pass


def test_a_file_on_this_machine_gets_the_local_stamp(env, monkeypatch, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"pdf bytes")
    stat = pdf.stat()
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: LocalFiles({"PDFAAAAA": pdf}))
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)

    search, _stats = env.update()

    stored = meta(search, "PDFAAAAA")
    assert stored["file_stamp"] == f"{stat.st_mtime_ns}:{stat.st_size}"
    assert stored["file_hash"]


def test_a_replaced_local_file_is_embedded_again(env, monkeypatch, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"pdf bytes")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: LocalFiles({"PDFAAAAA": pdf}))
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)
    env.update()
    pdf.write_bytes(b"replacement pdf bytes, longer")
    env.zot.put(attachment("PDFAAAAA", "ITEM0001"), version=6)

    search, _stats = env.update()

    assert env.zot.fulltext_keys() == ["PDFAAAAA"]
    assert meta(search, "PDFAAAAA")["file_stamp"].endswith(f":{pdf.stat().st_size}")


def test_web_mode_uses_the_api_stamp(env):
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001", md5="abc", mtime="7"), PDFAAAAA=TEXT)

    search, _stats = env.update()

    assert meta(search, "PDFAAAAA")["file_stamp"] == "api:abc:7"


def test_a_stamp_of_the_other_kind_is_not_a_change_by_itself(env, monkeypatch, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"pdf bytes")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda *a, **kw: LocalFiles({"PDFAAAAA": pdf}))
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)
    search, _stats = env.update()
    local_stamp = meta(search, "PDFAAAAA")["file_stamp"]
    # The same index seen from a machine without the file: an API stamp.
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: False)
    env.forget_sync_version()

    search, _stats = env.update()

    assert env.zot.fulltext_keys() == []
    assert meta(search, "PDFAAAAA")["file_stamp"] == local_stamp

    # Its own dateModified moved: now it is a change.
    env.zot.put(attachment("PDFAAAAA", "ITEM0001", date="2024-05-05T00:00:00Z"), version=6)

    search, _stats = env.update()

    assert env.zot.fulltext_keys() == ["PDFAAAAA"]
    assert meta(search, "PDFAAAAA")["file_stamp"] == "api:md5PDFAAAAA:100"


def test_unit_currency_across_stamp_kinds():
    unit = semantic_search._attachment_unit(
        "PDFAAAAA", "t", file_stamp="api:m:1", date_modified="2024-01-01T00:00:00Z", group_id=0
    )
    current = semantic_search.ZoteroSemanticSearch._unit_is_current
    same_day = {"item_key": "PDFAAAAA", "group_id": 0, "file_stamp": "5:9", "date_modified": "2024-01-01 00:00:00"}
    assert current(unit, same_day)
    assert not current(unit, {**same_day, "date_modified": "2024-02-01 00:00:00"})
    assert not current(unit, {**same_day, "date_modified": ""})
    # Two stamps of one kind that differ are a changed file.
    assert not current(unit, {**same_day, "file_stamp": "api:m:2"})
    assert not current({**unit, "file_stamp": "6:9"}, same_day)


def test_a_changed_attachment_counts_as_a_change_of_its_legacy_parent(env):
    library(env, paper("ITEM0001"), attachment("PDFAAAAA", "ITEM0001"), PDFAAAAA=TEXT)
    search = env.search()
    _legacy_record(search, "ITEM0001")
    # The parent is unchanged, but a file under it is new: only the child is listed since the watermark.
    env.zot.put(paper("ITEM0001"), version=5)
    env.zot.put(attachment("PDFAAAAA", "ITEM0001"), version=6)
    config = json.loads(Path(env.config).read_text())
    config["semantic_search"]["last_sync_version"] = 5
    Path(env.config).write_text(json.dumps(config))

    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}
    assert "title" not in meta(search, "ITEM0001")


def test_a_trashed_standalone_attachment_is_removed_by_an_incremental_run(env):
    library(env, paper("ITEM0001"), attachment("LONEPDF1"), LONEPDF1=TEXT)
    search, _stats = env.update()
    assert "LONEPDF1" in records(search)
    env.zot.rows["LONEPDF1"]["data"]["deleted"] = 1
    env.zot.rows["LONEPDF1"]["version"] = 6
    env.zot.version = 6
    env.zot.trashed_in_versions = True

    search, _stats = env.update()

    assert records(search) == {"ITEM0001"}
    assert not any(i.startswith("LONEPDF1") for i in all_ids(search))


def _watermark(env):
    return env.search()._load_last_sync_version()


def _fail_once_then_recover(env, monkeypatch, name, only=None):
    """Make ``env.zot.<name>`` raise a non-404 error (for ``only``, if given) until the returned flag is cleared."""
    original = getattr(env.zot, name)
    broken = {"on": True}

    def flaky(*args, **kwargs):
        if broken["on"] and (only is None or args[0] == only):
            raise RuntimeError("503 service unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(env.zot, name, flaky)
    return broken


def test_a_failed_fulltext_request_holds_back_the_sync_version_and_is_retried(env, monkeypatch):
    library(env, paper("ITEM0001"))
    env.update()
    assert _watermark(env) == 5
    env.zot.put(attachment("PDFAAAAA", "ITEM0001"), version=6)
    env.zot.texts["PDFAAAAA"] = TEXT
    broken = _fail_once_then_recover(env, monkeypatch, "fulltext_item")

    search, _stats = env.update()

    assert records(search) == {"ITEM0001"}
    assert _watermark(env) == 5

    broken["on"] = False
    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}
    assert _watermark(env) == 6


def test_a_failed_children_request_holds_back_the_sync_version_and_is_retried(env, monkeypatch):
    library(env, paper("ITEM0001"))
    env.update()
    env.zot.put(attachment("PDFAAAAA", "ITEM0001"), version=6)
    env.zot.texts["PDFAAAAA"] = TEXT
    broken = _fail_once_then_recover(env, monkeypatch, "children")

    search, _stats = env.update()

    assert records(search) == {"ITEM0001"}
    assert _watermark(env) == 5

    broken["on"] = False
    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}
    assert _watermark(env) == 6


def test_a_failed_parent_request_holds_back_the_sync_version_and_is_retried(env, monkeypatch):
    library(env, paper("ITEM0001"))
    env.update()
    env.zot.put(attachment("PDFAAAAA", "ITEM0001"), version=6)
    env.zot.texts["PDFAAAAA"] = TEXT
    broken = _fail_once_then_recover(env, monkeypatch, "item", only="ITEM0001")

    search, _stats = env.update()

    assert records(search) == {"ITEM0001"}
    assert _watermark(env) == 5

    broken["on"] = False
    search, _stats = env.update()

    assert records(search) == {"ITEM0001", "PDFAAAAA"}
    assert _watermark(env) == 6


def test_zoteros_no_text_answer_does_not_hold_back_the_sync_version(env):
    library(env, paper("ITEM0001"))
    env.update()
    env.zot.put(attachment("PDFAAAAA", "ITEM0001"), version=6)

    env.update()

    assert _watermark(env) == 6
