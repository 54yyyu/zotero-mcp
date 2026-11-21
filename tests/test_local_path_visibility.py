import json
from types import SimpleNamespace

import pytest
from conftest import DummyContext, FakeZotero
from test_issue_372_pdf_tools import (
    ATTACHMENT_KEY,
    make_library,
    patch_extract,
    use_local_library,
)

from zotero_mcp import chroma_client, semantic_search, server
from zotero_mcp.tools import retrieval
from zotero_mcp.utils import EXPOSE_LOCAL_PATHS_ENV_VAR, expose_local_paths


@pytest.mark.parametrize("value", ["0", "false", "no", " FALSE "])
def test_local_paths_can_be_hidden(monkeypatch, value):
    monkeypatch.setenv(EXPOSE_LOCAL_PATHS_ENV_VAR, value)
    assert not expose_local_paths()


@pytest.mark.parametrize("value", [None, "", "1", "true", "yes"])
def test_local_paths_are_exposed_by_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(EXPOSE_LOCAL_PATHS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(EXPOSE_LOCAL_PATHS_ENV_VAR, value)
    assert expose_local_paths()


def _linked_attachment():
    return {
        "key": "ATTACH01",
        "version": 3,
        "data": {
            "key": "ATTACH01",
            "itemType": "attachment",
            "linkMode": "linked_file",
            "title": "Paper PDF",
            "filename": "paper.pdf",
            "path": "/srv/zotero/storage/papers/paper.pdf",
        },
    }


def test_metadata_omits_linked_file_path_but_keeps_filename(monkeypatch):
    item = _linked_attachment()
    backend = SimpleNamespace(name="fake", get_item=lambda _key: item)
    monkeypatch.setattr(retrieval._library, "get_library_backend", lambda: backend)
    monkeypatch.setenv(EXPOSE_LOCAL_PATHS_ENV_VAR, "0")

    result = retrieval.get_item_metadata("ATTACH01", format="json", ctx=DummyContext())

    data = json.loads(result)["data"]
    assert "path" not in data
    assert data["filename"] == "paper.pdf"
    assert item["data"]["path"] == "/srv/zotero/storage/papers/paper.pdf"


def test_metadata_keeps_linked_file_path_by_default(monkeypatch):
    backend = SimpleNamespace(name="fake", get_item=lambda _key: _linked_attachment())
    monkeypatch.setattr(retrieval._library, "get_library_backend", lambda: backend)
    monkeypatch.delenv(EXPOSE_LOCAL_PATHS_ENV_VAR, raising=False)

    result = retrieval.get_item_metadata("ATTACH01", format="json", ctx=DummyContext())

    assert json.loads(result)["data"]["path"] == "/srv/zotero/storage/papers/paper.pdf"


def test_serverless_cli_metadata_is_not_affected(monkeypatch):
    backend = SimpleNamespace(name="fake", get_item=lambda _key: _linked_attachment())
    monkeypatch.setattr(retrieval._library, "get_library_backend", lambda: backend)
    monkeypatch.setenv(EXPOSE_LOCAL_PATHS_ENV_VAR, "0")
    monkeypatch.setattr(retrieval._utils._serverless, "ENABLED", True)

    result = retrieval.get_item_metadata("ATTACH01", format="json", ctx=DummyContext())

    assert json.loads(result)["data"]["path"] == "/srv/zotero/storage/papers/paper.pdf"


def _mock_index_status(monkeypatch, directory):
    info = {
        "name": "zotero_library",
        "persist_directory": str(directory),
        "embedding_model": "default",
    }
    monkeypatch.setattr(chroma_client, "read_collection_status", lambda _path: info)
    monkeypatch.setattr(semantic_search, "load_update_config", lambda _path: {})
    monkeypatch.setattr(semantic_search, "should_update", lambda _config: False)


def test_index_status_omits_database_path(monkeypatch, tmp_path):
    directory = tmp_path / "private-index"
    _mock_index_status(monkeypatch, directory)
    monkeypatch.setenv(EXPOSE_LOCAL_PATHS_ENV_VAR, "false")

    result = server.get_search_database_status(ctx=DummyContext())

    assert str(directory) not in result
    assert "Database Path" not in result


def test_index_status_keeps_database_path_by_default(monkeypatch, tmp_path):
    directory = tmp_path / "index"
    _mock_index_status(monkeypatch, directory)
    monkeypatch.delenv(EXPOSE_LOCAL_PATHS_ENV_VAR, raising=False)

    result = server.get_search_database_status(ctx=DummyContext())

    assert f"**Database Path:** {directory}" in result


def test_hidden_paths_do_not_disable_local_pdf_reads(monkeypatch, tmp_path):
    db_path, pdf_path = make_library(tmp_path)
    use_local_library(monkeypatch, db_path, FakeZotero())
    patch_extract(monkeypatch, text="Read locally without exposing the path.")
    monkeypatch.setenv(EXPOSE_LOCAL_PATHS_ENV_VAR, "0")

    result = server.read_pdf_pages(item_key=ATTACHMENT_KEY, start_page=1, ctx=DummyContext())

    assert "Read locally without exposing the path." in result
    assert str(pdf_path) not in result
