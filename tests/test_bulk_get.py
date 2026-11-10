"""Structured mixed reads, including errors, through the public bulk tool."""

import asyncio
import copy
import json

import pytest
import test_strict_sqlite_read_tools

from zotero_mcp import library, server
from zotero_mcp._context import ToolError

strict_library = test_strict_sqlite_read_tools.strict_library


def item(key, kind="journalArticle", **fields):
    return {"key": key, "version": 3, "data": {"itemType": kind, **fields}}


class Backend:
    def __init__(self):
        self.items = {
            "PAPER001": item(
                "PAPER001",
                title="Paper",
                abstractNote="Abstract",
                tags=[{"tag": "topic"}],
                collections=["COLL0001"],
                relations={"dc:relation": "uri"},
            ),
            "PAPER002": item("PAPER002", title="Empty parent"),
        }
        self.children = {
            "PAPER001": [
                item("NOTE0001", "note", note="<p>Full note</p>", tags=[{"tag": "note"}]),
                item("FILE0001", "attachment"),
            ],
            "PAPER002": [],
        }
        self.calls = []

    def get_items(self, keys):
        self.calls.append(("items", list(keys)))
        return {k: self.items[k] for k in keys if k in self.items}

    def get_children(self, keys):
        self.calls.append(("children", list(keys)))
        return {k: self.children[k] for k in keys if k in self.children}

    def find_by_citation_key(self, key):
        self.calls.append(("citekey", key))
        if key == "broken":
            raise RuntimeError("read failed")
        return self.items["PAPER001"] if key == "paper2026" else None


@pytest.fixture
def backend(monkeypatch):
    backend = Backend()
    monkeypatch.setattr(library, "get_library_backend", lambda: backend)
    return backend


def run(operations, dummy_ctx, **kwargs):
    return json.loads(server.bulk_get(operations, ctx=dummy_ctx, **kwargs))["results"]


def test_mixed_reads_are_ordered_and_preserve_raw_records(backend, dummy_ctx):
    before = copy.deepcopy(backend.items)
    results = run(
        [
            {"op": "notes", "parent_item_key": "PAPER001"},
            {"op": "item_metadata", "item_key": "PAPER001", "include": {"abstract": False, "tags": False}},
            {"op": "by_citation_key", "citation_key": "paper2026"},
            {"op": "item_children", "item_key": "PAPER001", "child_types": ["attachment"]},
            {"op": "item_metadata", "item_key": "PAPER001"},
        ],
        dummy_ctx,
    )
    assert [r["index"] for r in results] == list(range(5))
    assert all(r["ok"] and r["error"] is None for r in results)
    assert results[0]["notes"] == [{"note_key": "NOTE0001", "content": "<p>Full note</p>", "tags": ["note"]}]
    assert "abstractNote" not in results[1]["metadata"]["data"]
    assert "tags" not in results[1]["metadata"]["data"]
    assert results[2]["item_key"] == "PAPER001"
    assert results[2]["metadata"] == before["PAPER001"]
    assert [c["key"] for c in results[3]["children"]] == ["FILE0001"]
    assert results[4]["metadata"] == before["PAPER001"]
    assert backend.items == before
    assert backend.calls == [("items", ["PAPER001"]), ("children", ["PAPER001"]), ("citekey", "paper2026")]


def test_empty_reads_differ_from_failed_lookup(backend, dummy_ctx):
    results = run(
        [
            {"op": "item_children", "item_key": "PAPER002"},
            {"op": "notes", "parent_item_key": "PAPER002"},
            {"op": "by_citation_key", "citation_key": "missing"},
            {"op": "item_metadata", "item_key": "MISSING0"},
            {"op": "item_children", "item_key": "MISSING0"},
        ],
        dummy_ctx,
    )
    assert results[0]["children"] == [] and results[0]["ok"]
    assert results[1]["notes"] == [] and results[1]["ok"]
    assert results[2]["metadata"] is None and results[2]["item_key"] is None and results[2]["ok"]
    assert [r["error"] for r in results[3:]] == ["not_found_or_unavailable"] * 2


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "item_metadata", "item_key": "PAPER001", "include": {"abstract": "false"}},
        {"op": "item_metadata", "item_key": "PAPER001", "include": {"unknown": True}},
        {"op": "item_metadata", "item_key": "PAPER001", "extra": True},
        {"op": "item_metadata"},
        {"op": "item_metadata", "item_key": 123},
        {"op": "notes", "parent_item_key": ""},
        {"op": "item_children", "item_key": "PAPER001", "child_types": ["book"]},
        {"op": "by_citation_key", "citation_key": " "},
    ],
)
def test_bad_arguments_are_per_operation_errors(backend, dummy_ctx, operation):
    results = run([operation, {"op": "item_metadata", "item_key": "PAPER002"}], dummy_ctx)
    assert results[0]["error"] == "invalid_arguments" and not results[0]["ok"]
    assert results[1]["ok"]


def test_unknown_operation_and_runtime_failure_do_not_abort_default_batch(backend, dummy_ctx):
    results = run(
        [
            {"op": "delete_note", "note_key": "NOTE0001"},
            {"op": "by_citation_key", "citation_key": "broken"},
            {"op": "item_metadata", "item_key": "PAPER002"},
        ],
        dummy_ctx,
    )
    assert [r["error"] for r in results] == ["unknown_op", "read_failed", None]
    assert results[1]["detail"] == "read failed"


def test_fail_fast_does_not_prefetch_or_execute_later_reads(backend, dummy_ctx):
    results = run(
        [
            {"op": "item_metadata", "item_key": "MISSING0"},
            {"op": "by_citation_key", "citation_key": "paper2026"},
            {"op": "item_children", "item_key": "PAPER001"},
        ],
        dummy_ctx,
        continue_on_error=False,
    )
    assert [r["index"] for r in results] == [0, 1, 2]
    assert [r["op"] for r in results] == ["item_metadata", "by_citation_key", "item_children"]
    assert [r["error"] for r in results] == ["not_found_or_unavailable", "aborted", "aborted"]
    assert backend.calls == [("items", ["MISSING0"])]


def test_missing_children_mapping_is_a_failure(backend, dummy_ctx):
    del backend.children["PAPER001"]
    result = run([{"op": "item_children", "item_key": "PAPER001"}], dummy_ctx)[0]
    assert result["error"] == "not_found_or_unavailable"


def test_backend_failure_is_reported_for_each_affected_operation(backend, dummy_ctx, monkeypatch):
    def fail(keys):
        raise OSError("database unavailable")

    monkeypatch.setattr(backend, "get_items", fail)
    results = run(
        [
            {"op": "item_metadata", "item_key": "PAPER001"},
            {"op": "item_children", "item_key": "PAPER002"},
            {"op": "by_citation_key", "citation_key": "paper2026"},
        ],
        dummy_ctx,
    )
    assert [r["error"] for r in results] == ["read_failed", "read_failed", None]


def test_empty_and_oversized_envelopes(backend, dummy_ctx):
    assert run([], dummy_ctx) == []
    with pytest.raises(ToolError, match="50"):
        run([{"op": "item_metadata", "item_key": "PAPER001"}] * 51, dummy_ctx)
    assert backend.calls == []


def test_tool_is_registered_with_the_envelope_schema():
    tool = next(t for t in asyncio.run(server.mcp.list_tools()) if t.name == "zotero_bulk_get")
    assert "operations" in tool.parameters["required"]
    assert any(shape.get("type") == "array" for shape in tool.parameters["properties"]["operations"]["anyOf"])
    assert tool.parameters["properties"]["continue_on_error"]["default"] is True


@pytest.mark.parametrize("op", [None, [], {}, 7, "item_fulltext"])
def test_unknown_operation_shapes_are_isolated(backend, dummy_ctx, op):
    results = run([{"op": op}, {"op": "item_metadata", "item_key": "PAPER002"}], dummy_ctx)
    assert results[0]["error"] == "unknown_op" and results[0]["op"] == op
    assert results[1]["ok"]


def test_fail_fast_at_invalid_first_operation_never_opens_backend(monkeypatch, dummy_ctx):
    monkeypatch.setattr(library, "get_library_backend", lambda: pytest.fail("no reads should be attempted"))
    results = run(
        [{"op": "write"}, {"op": "item_metadata", "item_key": "PAPER001"}], dummy_ctx, continue_on_error=False
    )
    assert [r["error"] for r in results] == ["unknown_op", "aborted"]


def test_selectors_and_exact_batch_limit(backend, dummy_ctx):
    results = run(
        [
            {
                "op": "item_metadata",
                "item_key": "PAPER001",
                "include": {
                    "abstract": False,
                    "tags": False,
                    "collections": False,
                    "relations": False,
                },
            },
            {"op": "item_children", "item_key": "PAPER001", "child_types": []},
        ],
        dummy_ctx,
    )
    assert results[0]["metadata"]["data"] == {"itemType": "journalArticle", "title": "Paper"}
    assert results[1]["children"] == []
    assert len(run([{"op": "item_metadata", "item_key": "PAPER001"}] * 50, dummy_ctx)) == 50


def test_child_read_failure_does_not_fail_metadata(backend, dummy_ctx, monkeypatch):
    def fail(keys):
        raise OSError("children unavailable")

    monkeypatch.setattr(backend, "get_children", fail)
    results = run(
        [{"op": "item_metadata", "item_key": "PAPER001"}, {"op": "notes", "parent_item_key": "PAPER001"}], dummy_ctx
    )
    assert [r["error"] for r in results] == [None, "read_failed"]


def test_reads_use_real_sqlite_without_any_http(strict_library, dummy_ctx):
    results = run(
        [
            {"op": "item_metadata", "item_key": "PAPER001"},
            {"op": "item_children", "item_key": "PAPER001"},
            {"op": "notes", "parent_item_key": "PAPER001"},
            {"op": "by_citation_key", "citation_key": "vaswani2017"},
        ],
        dummy_ctx,
    )
    assert all(r["ok"] for r in results), results
    assert results[0]["metadata"]["data"]["title"] == "Attention Is All You Need"
    assert {c["key"] for c in results[1]["children"]} == {"PDFATT01", "NOTE0001"}
    assert results[2]["notes"][0]["content"] == "<p>Child note on self-attention</p>"
    assert results[3]["item_key"] == "PAPER001"
    assert strict_library == []


def test_api_backend_paginates_and_keeps_lookup_order(monkeypatch, dummy_ctx):
    class PagedZotero:
        def items(self, itemKey, start, limit):
            # The local API returns child notes before the requested parent.
            records = [item(f"NOTE{i:04d}", "note", note=f"<p>{i}</p>") for i in range(130)]
            records.append(item("PAPER001"))
            return records[start : start + limit]

        def children(self, key, start, limit):
            return [item(f"NOTE{i:04d}", "note", note=f"<p>{i}</p>") for i in range(130)][start : start + limit]

    monkeypatch.setattr(library, "get_library_backend", lambda: library.ApiBackend(PagedZotero()))
    results = run(
        [{"op": "notes", "parent_item_key": "PAPER001"}, {"op": "item_metadata", "item_key": "PAPER001"}], dummy_ctx
    )
    assert all(r["ok"] for r in results)
    assert len(results[0]["notes"]) == 130
    assert results[0]["notes"][-1]["note_key"] == "NOTE0129"
    assert results[1]["metadata"]["key"] == "PAPER001"


@pytest.mark.parametrize("encoded", [False, True])
def test_mcp_dispatch_validates_envelope_and_returns_json(backend, encoded):
    from fastmcp import Client

    from zotero_mcp._app import _ZoteroMCP

    # A fresh app exercises real MCP dispatch without the production lifespan
    # (which can refresh schema or start background indexing).
    app = _ZoteroMCP("bulk-probe")
    app.tool(name="zotero_bulk_get")(server.bulk_get)

    async def check():
        async with Client(app) as client:
            operations = [{"op": "item_metadata", "item_key": "PAPER001"}, {"op": "unknown"}]
            response = await client.call_tool(
                "zotero_bulk_get", {"operations": json.dumps(operations) if encoded else operations}
            )
            records = json.loads(response.content[0].text)["results"]
            assert records[0]["metadata"]["key"] == "PAPER001"
            assert records[1]["error"] == "unknown_op"
            assert not response.is_error
            oversized = await client.call_tool(
                "zotero_bulk_get",
                {
                    "operations": [
                        {"op": "item_metadata", "item_key": "PAPER002"},
                    ]
                    * 51
                },
                raise_on_error=False,
            )
            assert oversized.is_error
            assert "50" in oversized.content[0].text

    asyncio.run(check())
    assert backend.calls == [("items", ["PAPER001"])]


def test_json_string_operations_are_equivalent(backend, dummy_ctx):
    operations = [{"op": "item_metadata", "item_key": "PAPER001"}, {"op": "notes", "parent_item_key": "PAPER001"}]
    assert run(json.dumps(operations), dummy_ctx) == run(operations, dummy_ctx)


@pytest.mark.parametrize("operations", ["[", "{}", '"text"', "null", '["item_metadata"]', "[1]"])
def test_malformed_encoded_envelope_reads_nothing(backend, dummy_ctx, operations):
    with pytest.raises(ToolError, match="array of objects"):
        run(operations, dummy_ctx)
    assert backend.calls == []
