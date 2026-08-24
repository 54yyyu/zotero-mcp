"""Cross-backend parity for `zotero_advanced_search`, without a live library.

`tests/live/test_search_backend_parity.py` compares the backends against
whatever real Zotero happens to be connected. That is the stronger test, but it
needs `ZOTERO_MCP_LIVE_TESTS=1`, a running Zotero and someone's actual data, so
it does not run in CI — which is exactly where a silent divergence would be
introduced.

This module closes that gap. Every case runs the *real* `server.advanced_search`
twice over the same corpus (`tests/_search_corpus.py`), once routed to the
SQLite backend and once to the pyzotero path, and asserts the two return the
same item keys. Nothing is asserted against a hand-written expectation: the
assertion is that the backends *agree*, so adding a case to `CONDITION_CASES`
extends coverage of both paths at once and cannot go stale.

The one deliberate disagreement — `year`/`date` on multipart dates, where SQL
can read Zotero's ISO prefix and the API can only see the display text — is
asserted explicitly at the bottom rather than skipped, so that if it ever
changes, a test says so.
"""

from __future__ import annotations

import pytest
from conftest import DummyContext

import _search_corpus as corpus
from zotero_mcp import client as _client
from zotero_mcp import server
from zotero_mcp import utils as _utils
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import search as search_module

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class _RefusingZotero:
    """A pyzotero stand-in whose use fails the test.

    ``advanced_search`` builds a client before it decides which backend to
    use, so the SQL path legitimately *obtains* one — it just must never call
    it. Refusing at method level rather than at construction distinguishes
    those two things.
    """

    def add_parameters(self, **kwargs):
        raise AssertionError(
            "the SQLite backend fell back to pyzotero; parity comparison is invalid"
        )

    def items(self, *args, **kwargs):
        raise AssertionError(
            "the SQLite backend fell back to pyzotero; parity comparison is invalid"
        )


class _CorpusZotero:
    """A pyzotero stand-in serving the corpus, paged like the real client."""

    def __init__(self, items: list[dict]):
        self._items = items

    def add_parameters(self, **kwargs):
        pass

    def items(self, start: int = 0, limit: int = 100, **kwargs):
        return self._items[start : start + limit]


def _keys_from_output(text: str) -> set[str]:
    return {line[4:] for line in text.splitlines() if line.startswith("KEY:")}


def _run_raw(monkeypatch, tmp_path, backend: str, **kwargs) -> str:
    """Route ``advanced_search`` to *backend* and return its raw output.

    The caller is expected to have already installed whatever
    ``format_item_result`` stand-in it wants to read results through.
    """
    monkeypatch.setattr(_client, "get_active_group_id", lambda: 0)
    monkeypatch.setattr(_utils, "get_search_backend", lambda: backend)

    if backend == "sqlite":
        db_path = tmp_path / "zotero.sqlite"
        if not db_path.exists():
            corpus.build_sqlite(db_path)
        monkeypatch.setattr(
            search_module, "get_local_zotero_reader",
            lambda: LocalZoteroReader(db_path=str(db_path)),
        )
        # Any *use* of the API client here would mean the SQL path silently
        # fell back, invalidating the comparison.
        monkeypatch.setattr(_client, "get_zotero_client", lambda: _RefusingZotero())
    else:
        monkeypatch.setattr(search_module, "get_local_zotero_reader", lambda: None)
        monkeypatch.setattr(
            _client, "get_zotero_client", lambda: _CorpusZotero(corpus.build_api_items())
        )

    return server.advanced_search(ctx=DummyContext(), limit=100, **kwargs)


def _run(monkeypatch, tmp_path, backend: str, **kwargs) -> set[str]:
    """Run the real advanced_search tool against *backend*, return item keys."""
    # Render each result as its key alone: parity is about which items come
    # back, and the formatted output is not the thing under test.
    monkeypatch.setattr(
        _utils, "format_item_result", lambda item, index=0, **_kwargs: [f"KEY:{item['key']}"]
    )
    return _keys_from_output(_run_raw(monkeypatch, tmp_path, backend, **kwargs))


def _cond(field: str, operation: str, value: str) -> list[dict[str, str]]:
    return [{"field": field, "operation": operation, "value": value}]


# ---------------------------------------------------------------------------
# The case table. Each entry runs against both backends and must agree.
# ---------------------------------------------------------------------------

_TEXT_OPS = ["is", "isNot", "contains", "doesNotContain", "beginsWith", "endsWith"]

CONDITION_CASES: list[tuple[str, str, str]] = []

# Accented / unaccented / mixed-case creators, across every text operator.
for _op in _TEXT_OPS:
    for _value in ["Müller", "muller", "MULLER", "Hans Müller", "Ångström", "王", "Wang"]:
        CONDITION_CASES.append(("creator", _op, _value))

# Titles, including LIKE metacharacters and dash/space folding.
for _op in _TEXT_OPS:
    for _value in ["50%", "%", "_", "Underscore_Separated_Title", "Cladder-Micus",
                   "Cladder Micus", "En Dash – Study", "quantum", "QUANTUM"]:
        CONDITION_CASES.append(("title", _op, _value))

# Tags: multi-valued, so the negated operators exercise the all()/any() split.
for _op in _TEXT_OPS:
    for _value in ["méthode", "methode", "Méthode", "physics", "nonexistent"]:
        CONDITION_CASES.append(("tag", _op, _value))

# Single-valued scalar fields.
for _op in _TEXT_OPS:
    CONDITION_CASES += [
        ("itemType", _op, "journalArticle"),
        ("itemType", _op, "book"),
        ("publicationTitle", _op, "Journal of Quantum"),
        ("abstractNote", _op, "quantum"),
        ("DOI", _op, "10.1/quantum"),
    ]

# Range operators on the fields where both backends read the same value.
for _op in ["isGreaterThan", "isLessThan", "isBefore", "isAfter"]:
    CONDITION_CASES += [
        ("dateAdded", _op, "2024-01-01"),
        ("dateModified", _op, "2024-03-01"),
    ]

# Field-name aliases must resolve identically on both sides.
for _alias in ["author", "authors", "creators", "tags", "itemtype", "doi"]:
    CONDITION_CASES.append((_alias, "contains", "a"))


@pytest.mark.parametrize(
    "field,operation,value",
    CONDITION_CASES,
    ids=[f"{f}-{o}-{v}" for f, o, v in CONDITION_CASES],
)
def test_backends_agree(monkeypatch, tmp_path, field, operation, value):
    sql = _run(monkeypatch, tmp_path, "sqlite", conditions=_cond(field, operation, value))
    api = _run(monkeypatch, tmp_path, "api", conditions=_cond(field, operation, value))
    assert sql == api, (
        f"{field} {operation} {value!r}: "
        f"sqlite-only={sorted(sql - api)} api-only={sorted(api - sql)}"
    )


@pytest.mark.parametrize("join_mode", ["all", "any"])
def test_backends_agree_on_multi_condition_joins(monkeypatch, tmp_path, join_mode):
    conditions = [
        {"field": "creator", "operation": "contains", "value": "muller"},
        {"field": "tag", "operation": "contains", "value": "methode"},
    ]
    sql = _run(monkeypatch, tmp_path, "sqlite", conditions=conditions, join_mode=join_mode)
    api = _run(monkeypatch, tmp_path, "api", conditions=conditions, join_mode=join_mode)
    assert sql == api


# ---------------------------------------------------------------------------
# Properties the parity comparison itself relies on
# ---------------------------------------------------------------------------

def test_accented_and_unaccented_queries_return_the_same_items(monkeypatch, tmp_path):
    """The property the whole normalization change exists to provide."""
    accented = _run(monkeypatch, tmp_path, "sqlite", conditions=_cond("creator", "contains", "Müller"))
    plain = _run(monkeypatch, tmp_path, "sqlite", conditions=_cond("creator", "contains", "muller"))
    assert accented == plain
    assert {"ACCENT01", "ACCENT02", "ACCENT03"} <= accented


def test_like_metacharacters_are_literal_not_wildcards(monkeypatch, tmp_path):
    """A search for '%' must find the one item containing '%', not everything."""
    pct = _run(monkeypatch, tmp_path, "sqlite", conditions=_cond("title", "contains", "%"))
    assert pct == {"PCTSIGN1"}

    underscore = _run(monkeypatch, tmp_path, "sqlite", conditions=_cond("title", "contains", "_"))
    assert underscore == {"PCTSIGN2"}


def test_items_without_the_field_satisfy_no_operator(monkeypatch, tmp_path):
    """Negated operators must not sweep in items that have no value at all."""
    for backend in ("sqlite", "api"):
        keys = _run(monkeypatch, tmp_path, backend,
                    conditions=_cond("creator", "isNot", "Nobody At All"))
        assert "NOCREAT1" not in keys, f"{backend} matched a creator-less item on isNot"
        assert "BARE0001" not in keys

        tags = _run(monkeypatch, tmp_path, backend,
                    conditions=_cond("tag", "doesNotContain", "nothing"))
        assert "NOTAGS01" not in tags, f"{backend} matched a tagless item on doesNotContain"


def test_excluded_item_types_never_appear(monkeypatch, tmp_path):
    for backend in ("sqlite", "api"):
        keys = _run(monkeypatch, tmp_path, backend,
                    conditions=_cond("title", "doesNotContain", "zzzz-no-such-text"))
        assert "ATTACH01" not in keys
        assert "NOTEITM1" not in keys
        assert "DELETED1" not in keys, f"{backend} returned a trashed item"


# ---------------------------------------------------------------------------
# The one divergence that is intentional
# ---------------------------------------------------------------------------

def test_year_on_multipart_dates_diverges_by_design(monkeypatch, tmp_path):
    """SQL reads Zotero's ISO prefix; the API can only see the display text.

    Zotero stores ``"2016-10-01 October 1, 2016"`` but exposes only
    ``"October 1, 2016"`` over the API, so the API path's ``date[:4]`` yields
    ``"Octo"`` where SQL's ``SUBSTR(value, 1, 4)`` yields ``"2016"``. The SQL
    answer is the correct one — it is what Zotero itself does — so this is a
    fix rather than a divergence to reconcile, and normalizing the two would
    mean re-introducing the bug on the SQL side.

    Asserted rather than skipped so that a change in either path is noticed.
    """
    sql = _run(monkeypatch, tmp_path, "sqlite", conditions=_cond("year", "is", "2016"))
    api = _run(monkeypatch, tmp_path, "api", conditions=_cond("year", "is", "2016"))

    assert "ACCENT01" in sql, "SQL should read the ISO prefix of a multipart date"
    assert "ACCENT01" not in api, (
        "the API path cannot see the ISO prefix; if this now passes, the "
        "divergence has been fixed and this test should become a parity case"
    )


# ---------------------------------------------------------------------------
# Parity of rendered field values, not just of which items match.
# ---------------------------------------------------------------------------

def _run_dates(monkeypatch, tmp_path, backend: str, **kwargs) -> dict[str, str]:
    """Like ``_run``, but render each result's key and its ``date`` field."""
    monkeypatch.setattr(
        _utils,
        "format_item_result",
        lambda item, index=0, **_kwargs: [f"KEY:{item['key']}=DATE:{item['data'].get('date', '')}"],
    )
    out = _run_raw(monkeypatch, tmp_path, backend, **kwargs)
    pairs = {}
    for line in out.splitlines():
        if line.startswith("KEY:"):
            key, _, date = line[4:].partition("=DATE:")
            pairs[key] = date
    return pairs


def test_backends_agree_on_the_rendered_date(monkeypatch, tmp_path):
    """The hydrated ``date`` must be the API's display half, not Zotero's raw
    multipart storage form.

    Zotero stores ``"2017-00-00 2017"`` and the web API returns ``"2017"``.
    ``_DATE_DISPLAY_SQL`` strips the ISO prefix for search *conditions*, but
    the hydration projections that build the returned item selected
    ``date_val.value`` raw, so every date rendered through the SQLite backend
    carried the prefix: doubled where the user typed an ISO date, and visibly
    mangled where they did not. Verified against a live Zotero 10 library, all
    113 dated items were wrong before the fix and all 113 match the API after.

    The rest of this module compares which keys come back, which is why this
    went unnoticed: no case looked at a field value.
    """
    sql = _run_dates(monkeypatch, tmp_path, "sqlite", conditions=_cond("title", "contains", "a"))
    api = _run_dates(monkeypatch, tmp_path, "api", conditions=_cond("title", "contains", "a"))

    assert sql == api, "sqlite and api render different dates for the same items"

    expected = {i.key: i.display_date for i in corpus.CORPUS}
    for key, rendered in sql.items():
        assert rendered == expected[key], (
            f"{key}: rendered {rendered!r}, Zotero's API would say {expected[key]!r}"
        )

    # The cases that make an unstripped prefix visible rather than merely
    # doubled must actually be in the comparison, not filtered out upstream.
    assert {"PARTDT01", "PARTDT02", "PARTDT03"} <= set(sql), (
        "the partial-date corpus items did not reach the comparison"
    )
