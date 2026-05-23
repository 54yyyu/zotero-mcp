"""Regression tests for #293: BBT JSON-RPC client.

Three bugs in ``ZoteroBetterBibTexAPI`` against current Better BibTeX:

1. ``export_bibtex`` sent positional params to ``item.citationkey`` and
   used the ``library_id:item_key`` compound form. The correct call shape
   is named params (``{"item_keys": ["KEY"]}``) with bare item keys.
2. ``item.export`` returns BibTeX entries with empty citekeys in some
   BBT versions — ``@article{`` instead of ``@article{fooBar2020``.
3. ``get_item_by_citekey`` called ``item.search``, which does not exist
   in current BBT (always returns -32601 Method not found).
"""

import json
from unittest.mock import patch

import pytest

from zotero_mcp.better_bibtex_client import (
    ZoteroBetterBibTexAPI,
    _inject_citekey,
)

# ---------------------------------------------------------------------------
# _inject_citekey helper
# ---------------------------------------------------------------------------

class TestInjectCitekey:
    def test_inserts_into_empty_atline(self):
        out = _inject_citekey("@article{\n  title = {{X}}\n}", "fooBar2020")
        assert out.startswith("@article{fooBar2020,\n")

    def test_inserts_when_only_comma_present(self):
        # BBT sometimes emits `@article{,` with a bare leading comma.
        out = _inject_citekey("@article{,\n  title = {{X}}\n}", "fooBar2020")
        assert out.startswith("@article{fooBar2020,\n")

    def test_leaves_existing_key_alone(self):
        src = "@article{existingKey2020,\n  title = {{X}}\n}"
        assert _inject_citekey(src, "differentKey") == src

    def test_only_touches_first_missing_entry(self):
        """Only the first @-line that needs a citekey gets one — never
        overwrite an entry that already has its own key."""
        src = (
            "@article{firstKey,\n  title = {{A}}\n}\n"
            "@book{,\n  title = {{B}}\n}\n"
            "@inproceedings{,\n  title = {{C}}\n}"
        )
        out = _inject_citekey(src, "newKey")
        # First entry's key preserved; first MATCH (the @book) gets the key;
        # subsequent matches untouched (count=1).
        assert "@article{firstKey," in out
        assert "@book{newKey," in out
        assert "@inproceedings{,\n" in out

    def test_empty_inputs_passthrough(self):
        assert _inject_citekey("", "key") == ""
        assert _inject_citekey("@article{}\n", "") == "@article{}\n"


# ---------------------------------------------------------------------------
# export_bibtex JSON-RPC contract
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _capture_post(captured_payloads, results_iter):
    """Build a stub for ``requests.post`` that records the request body
    and returns the next preset JSON-RPC result."""

    def _stub(url, headers=None, data=None, timeout=None):
        captured_payloads.append(json.loads(data))
        result = next(results_iter)
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)

    return _stub


class TestExportBibtex:
    def test_citationkey_uses_named_params_and_bare_key(self):
        """``item.citationkey`` must receive ``{"item_keys": ["KEY"]}`` with the
        bare item key, NOT positional ``[[\"library:KEY\"]]`` (#293 Bug 1)."""
        captured: list[dict] = []
        results = iter([
            {"jsonrpc": "2.0", "id": 1, "result": {"LLLGRWNR": "fergueneSavoirfaire2001"}},
            {"jsonrpc": "2.0", "id": 1, "result": "@article{\n  title = {{T}}\n}"},
        ])
        client = ZoteroBetterBibTexAPI()
        with patch("requests.post", side_effect=_capture_post(captured, results)):
            out = client.export_bibtex("LLLGRWNR", library_id=1)

        # First call: item.citationkey with named params, bare item key.
        first = captured[0]
        assert first["method"] == "item.citationkey"
        assert first["params"] == {"item_keys": ["LLLGRWNR"]}, (
            f"item.citationkey must use named params + bare key; got {first['params']!r}"
        )

        # Returned BibTeX must have the citekey injected (Bug 2).
        assert out.startswith("@article{fergueneSavoirfaire2001,")

    def test_export_payload_string_is_returned_with_citekey(self):
        captured: list[dict] = []
        results = iter([
            {"result": {"KEY1": "myCite2024"}},
            {"result": "@article{\n  title = {{T}}\n}"},
        ])
        client = ZoteroBetterBibTexAPI()
        with patch("requests.post", side_effect=_capture_post(captured, results)):
            out = client.export_bibtex("KEY1")
        assert "myCite2024," in out

    def test_export_payload_list_form_is_returned_with_citekey(self):
        """Some BBT versions wrap the BibTeX in a list."""
        captured: list[dict] = []
        results = iter([
            {"result": {"KEY1": "myCite2024"}},
            {"result": ["@article{\n  title = {{T}}\n}"]},
        ])
        client = ZoteroBetterBibTexAPI()
        with patch("requests.post", side_effect=_capture_post(captured, results)):
            out = client.export_bibtex("KEY1")
        assert "myCite2024," in out

    def test_missing_citekey_does_not_proceed_to_export(self):
        captured: list[dict] = []
        results = iter([{"result": {}}])  # no mapping for our key
        client = ZoteroBetterBibTexAPI()
        with patch("requests.post", side_effect=_capture_post(captured, results)):
            out = client.export_bibtex("UNKNOWN")
        # Existing contract: the function prints the error and returns "".
        assert out == ""
        # And we did NOT proceed to item.export.
        assert len(captured) == 1
        assert captured[0]["method"] == "item.citationkey"


# ---------------------------------------------------------------------------
# get_item_by_citekey no longer calls the nonexistent item.search
# ---------------------------------------------------------------------------

class TestGetItemByCitekey:
    def test_does_not_call_item_search(self):
        """Bug 3 (#293): the broken ``item.search`` probe must be gone."""
        captured: list[dict] = []
        payload = json.dumps({"items": [{"id": "myCite2020", "title": "T"}]})
        results = iter([{"result": payload}])
        client = ZoteroBetterBibTexAPI()
        with patch("requests.post", side_effect=_capture_post(captured, results)):
            item = client.get_item_by_citekey("myCite2020")

        assert item == {"id": "myCite2020", "title": "T"}
        assert all(p["method"] != "item.search" for p in captured), (
            "get_item_by_citekey must not call item.search — that BBT method "
            "does not exist in current versions (#293 Bug 3)."
        )
        assert captured[0]["method"] == "item.export"

    def test_empty_response_raises(self):
        captured: list[dict] = []
        results = iter([{"result": json.dumps({"items": []})}])
        client = ZoteroBetterBibTexAPI()
        with patch("requests.post", side_effect=_capture_post(captured, results)):
            with pytest.raises(Exception, match="No items returned"):
                client.get_item_by_citekey("missing")
