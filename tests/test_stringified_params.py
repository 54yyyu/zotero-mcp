"""Every list/dict-typed tool parameter tolerates a stringified value (#459).

Some MCP clients serialize array and object arguments as JSON *text* before
dispatch. Pydantic validates against the published schema before the tool body
runs, so a parameter annotated `list[...]` without `| str` is rejected at the
boundary — and any JSON-parsing the body does for that parameter is dead code
it can never reach.

#459 was the `rect` instance: area annotations were unreachable from Claude
Code entirely. Sweeping for the same shape turned up one more,
`zotero_advanced_search.conditions`, which was worse — its own description
promised "also accepts a JSON string" and its body had the `json.loads` branch
to honour that, both unreachable.

The sweep is the point of this file. A new list-typed parameter added without
string tolerance fails here rather than in someone's client.
"""

import asyncio
import json
import os
import typing

import pytest


@pytest.fixture(scope="module")
def registered_tools():
    os.environ.setdefault("ZOTERO_LOCAL", "true")
    os.environ["ZOTERO_MCP_TOOLSETS"] = "all"
    from zotero_mcp import server  # noqa: F401  (registers the tools)
    from zotero_mcp._app import mcp

    return asyncio.run(mcp.list_tools())


def _sequence_params(tool):
    """(param, annotation) for every list/dict-typed parameter of *tool*."""
    fn = getattr(tool, "fn", None)
    if fn is None:
        return []
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        return []
    out = []
    for name, annotation in hints.items():
        if name in ("return", "ctx"):
            continue
        rendered = str(annotation)
        if "list[" in rendered or "dict[" in rendered:
            out.append((name, annotation))
    return out


def _accepts_bare_str(annotation) -> bool:
    return any(arg is str for arg in typing.get_args(annotation))


class TestNoParameterIsUnreachable:
    def test_every_sequence_param_tolerates_a_json_string(self, registered_tools):
        offenders = []
        for tool in registered_tools:
            for name, annotation in _sequence_params(tool):
                if not _accepts_bare_str(annotation):
                    offenders.append(f"{tool.name}.{name}: {annotation}")

        assert not offenders, (
            "These parameters are list/dict-typed with no `| str` in the "
            "annotation. A client that stringifies arguments cannot reach "
            "them, and any json.loads in the body is dead code:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_sweep_actually_finds_parameters(self, registered_tools):
        """Guard against the check silently passing because it inspected
        nothing — a typing change upstream could empty it out."""
        total = sum(len(_sequence_params(t)) for t in registered_tools)
        assert total > 15, f"only {total} sequence params found; sweep looks broken"


class TestKnownInstances:
    """The two the sweep was written for, pinned by name."""

    def _tool(self, tools, name):
        return next(t for t in tools if t.name == name)

    def test_advanced_search_conditions_accepts_a_json_string(self, registered_tools):
        tool = self._tool(registered_tools, "zotero_advanced_search")
        annotation = dict(typing.get_type_hints(tool.fn))["conditions"]
        assert _accepts_bare_str(annotation)

    def test_advanced_search_promises_string_input_in_its_description(
        self, registered_tools
    ):
        """The description and the annotation have to agree. They did not:
        the description said "also accepts a JSON string" while the signature
        made that impossible."""
        tool = self._tool(registered_tools, "zotero_advanced_search")
        assert "JSON string" in (tool.description or "")

    def test_create_annotation_rect_accepts_a_json_string(self, registered_tools):
        tool = self._tool(registered_tools, "zotero_create_annotation")
        annotation = dict(typing.get_type_hints(tool.fn))["rect"]
        assert _accepts_bare_str(annotation)


class TestValidationLayer:
    """Annotations are the mechanism; this asserts the actual behaviour."""

    def test_stringified_conditions_passes_validation(self, registered_tools):
        tool = next(t for t in registered_tools if t.name == "zotero_advanced_search")
        payload = json.dumps([{"field": "itemType", "operation": "is", "value": "book"}])

        try:
            asyncio.run(tool.run({"conditions": payload, "limit": 1}))
        except Exception as exc:
            # Reaching the body and failing there (no Zotero running) is the
            # pass condition. A ValidationError means we never got that far.
            assert type(exc).__name__ != "ValidationError", (
                f"stringified conditions still rejected at the boundary: {exc}"
            )

    def test_stringified_rect_passes_validation(self, registered_tools):
        tool = next(t for t in registered_tools if t.name == "zotero_create_annotation")

        try:
            asyncio.run(tool.run({
                "attachment_key": "ABCD1234", "page": 1,
                "rect": "[0.1628, 0.0985, 0.7783, 0.7272]",
            }))
        except Exception as exc:
            assert type(exc).__name__ != "ValidationError", (
                f"stringified rect still rejected at the boundary: {exc}"
            )
