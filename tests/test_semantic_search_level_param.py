"""The ``level`` parameter of ``zotero_semantic_search`` (#769)."""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from conftest import DummyContext  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.tools import search as search_module  # noqa: E402


class _RecordingSearch:
    def __init__(self, result=None):
        self.kwargs = None
        self.result = result or {"results": []}

    def search(self, **kwargs):
        self.kwargs = kwargs
        return self.result


@pytest.fixture()
def fake(monkeypatch):
    recording = _RecordingSearch()
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **kw: recording)
    monkeypatch.setattr(search_module, "_maybe_fire_presearch_sync", lambda *_a, **_k: None)
    return recording


@pytest.mark.parametrize("level", ["articles", "passages"])
def test_the_level_reaches_the_search(fake, level):
    search_module.semantic_search(query="q", level=level, ctx=DummyContext())

    assert fake.kwargs["level"] == level


def test_no_level_leaves_the_choice_to_the_setting(fake):
    search_module.semantic_search(query="q", ctx=DummyContext())

    assert "level" not in fake.kwargs


def test_an_error_about_the_level_reaches_the_caller(fake):
    fake.result = {"results": [], "error": "This index has no passages."}

    out = search_module.semantic_search(query="q", level="passages", ctx=DummyContext())

    assert "This index has no passages." in out
