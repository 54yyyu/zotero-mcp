"""Prompts name `zotero_search_items` when semantic search is not usable.

`zotero_literature_review` and `zotero_find_contradicting_evidence` sent the
model to `zotero_semantic_search` even when ChromaDB is not installed or the
tool is switched off with `ZOTERO_MCP_TOOLSETS`, so the first call of the
workflow failed. With semantic search usable, the text is unchanged.
"""

import importlib.util

import pytest

from zotero_mcp import prompts, toolsets

SEMANTIC = "zotero_semantic_search"
KEYWORD = "zotero_search_items"


def _fn(obj):
    # FastMCP versions differ in whether @mcp.prompt returns the function.
    return getattr(obj, "fn", obj)


RENDERERS = {
    "literature_review": lambda: _fn(prompts.literature_review)("sleep and memory"),
    "find_contradicting_evidence": lambda: _fn(prompts.find_contradicting_evidence)(
        "sleep improves memory"
    ),
}


@pytest.fixture
def chromadb_installed(monkeypatch):
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "chromadb":
            return object()
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.fixture
def chromadb_missing(monkeypatch):
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "chromadb":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.fixture
def semantic_toolset(monkeypatch):
    """The tool in its own `semantic` toolset, as #689 proposes."""
    monkeypatch.setitem(toolsets.TOOLSETS, "semantic", frozenset({SEMANTIC}))


@pytest.mark.parametrize("name", sorted(RENDERERS))
def test_semantic_available_keeps_semantic_search(name, chromadb_installed, monkeypatch):
    monkeypatch.delenv(toolsets.TOOLSETS_ENV_VAR, raising=False)
    text = RENDERERS[name]()
    assert SEMANTIC in text
    assert KEYWORD not in text


@pytest.mark.parametrize("name", sorted(RENDERERS))
def test_chromadb_missing_falls_back_to_search_items(name, chromadb_missing, monkeypatch):
    monkeypatch.delenv(toolsets.TOOLSETS_ENV_VAR, raising=False)
    text = RENDERERS[name]()
    assert SEMANTIC not in text
    assert f"`{KEYWORD}(" in text
    assert "qmode='everything'" in text


@pytest.mark.parametrize("name", sorted(RENDERERS))
def test_toolset_disabled_falls_back_to_search_items(
    name, chromadb_installed, semantic_toolset, monkeypatch
):
    monkeypatch.setenv(toolsets.TOOLSETS_ENV_VAR, "none")
    text = RENDERERS[name]()
    assert SEMANTIC not in text
    assert f"`{KEYWORD}(" in text


@pytest.mark.parametrize("name", sorted(RENDERERS))
def test_toolset_enabled_keeps_semantic_search(
    name, chromadb_installed, semantic_toolset, monkeypatch
):
    monkeypatch.setenv(toolsets.TOOLSETS_ENV_VAR, "none,semantic")
    text = RENDERERS[name]()
    assert SEMANTIC in text
    assert KEYWORD not in text


def test_available_text_is_unchanged(chromadb_installed, monkeypatch):
    monkeypatch.delenv(toolsets.TOOLSETS_ENV_VAR, raising=False)
    review = RENDERERS["literature_review"]()
    assert (
        "1. Run `zotero_semantic_search(query='sleep and memory', limit=12)` to find "
        "the most relevant papers already in the library. Note each paper's key and "
        "the matched passage."
    ) in review
    contra = RENDERERS["find_contradicting_evidence"]()
    assert (
        "2. `zotero_semantic_search` again with an INVERTED / skeptical phrasing of "
        "the claim (e.g. limitations, null results, criticisms) to surface "
        "disconfirming work."
    ) in contra


def test_other_steps_survive_the_fallback(chromadb_missing, monkeypatch):
    monkeypatch.delenv(toolsets.TOOLSETS_ENV_VAR, raising=False)
    review = RENDERERS["literature_review"]()
    assert "zotero_find_related_papers" in review
    assert review.splitlines()[3].startswith("1. ")
    contra = RENDERERS["find_contradicting_evidence"]()
    assert "3. Sort the results into SUPPORTS / CONTRADICTS / MIXED" in contra
