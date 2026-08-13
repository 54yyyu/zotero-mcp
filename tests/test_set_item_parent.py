"""Tests for the item-parent mutation tool."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from conftest import DummyContext

from zotero_mcp import server
from zotero_mcp.tools.item_parent import set_item_parent

ITEM_KEY = "CHILD001"
PARENT_KEY = "PARENT01"


def test_schema_requires_an_explicit_nullable_parent_key():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    parameters = tools["zotero_set_item_parent"].parameters

    assert "parent_key" in parameters["required"]
    assert {option["type"] for option in parameters["properties"]["parent_key"]["anyOf"]} == {
        "string",
        "null",
    }


class ParentWriteFake:
    def __init__(self, *, response=True):
        self.response = response
        self.item_calls: list[str] = []
        self.update_calls: list[dict] = []
        self.raise_on_item: Exception | None = None
        self.raise_on_update: Exception | None = None

    def item(self, key):
        self.item_calls.append(key)
        if self.raise_on_item:
            raise self.raise_on_item
        return {"key": key, "version": 7}

    def update_item(self, payload):
        self.update_calls.append(deepcopy(payload))
        if self.raise_on_update:
            raise self.raise_on_update
        return self.response


def _install(monkeypatch, fake):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client",
        lambda _ctx: (object(), fake),
    )


def _call(parent_key):
    return set_item_parent(
        item_key=ITEM_KEY,
        parent_key=parent_key,
        ctx=DummyContext(),
    )


@pytest.mark.parametrize(
    ("parent_key", "api_value", "message"),
    [
        (PARENT_KEY, PARENT_KEY, "Successfully set the parent"),
        (None, False, "Successfully cleared the parent"),
    ],
)
def test_sends_minimal_versioned_parent_update(
    monkeypatch,
    parent_key,
    api_value,
    message,
):
    fake = ParentWriteFake()
    _install(monkeypatch, fake)

    result = _call(parent_key)

    assert message in result
    assert fake.item_calls == [ITEM_KEY]
    assert fake.update_calls == [
        {
            "key": ITEM_KEY,
            "version": 7,
            "parentItem": api_value,
        }
    ]


def test_surfaces_api_exception(monkeypatch):
    fake = ParentWriteFake()
    fake.raise_on_update = RuntimeError("412 Precondition Failed")
    _install(monkeypatch, fake)

    result = _call(PARENT_KEY)

    assert result == ("Error setting parent for item 'CHILD001': 412 Precondition Failed")
    assert len(fake.update_calls) == 1


def test_reports_unsuccessful_response(monkeypatch):
    fake = ParentWriteFake(response=False)
    _install(monkeypatch, fake)

    result = _call(PARENT_KEY)

    assert result == "Failed to set parent for item 'CHILD001'."


def test_surfaces_item_fetch_error_without_writing(monkeypatch):
    fake = ParentWriteFake()
    fake.raise_on_item = RuntimeError("404 Not Found")
    _install(monkeypatch, fake)

    result = _call(PARENT_KEY)

    assert result == "Error setting parent for item 'CHILD001': 404 Not Found"
    assert fake.update_calls == []
