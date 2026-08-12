"""Tests for #A5: add_by_doi fetches CrossRef metadata for a whole batch of
DOIs in one ``/works?filter=doi:...`` request per 50, rather than one request
per DOI.

The concurrent per-DOI version this replaced was measurably worse than the
sequential baseline it was meant to beat: on a real 25-DOI import, 18 of the
concurrent fetches came back HTTP 429. One batched request for the same DOIs
does not throttle at all, so these tests pin the request *shape* — how many
requests, which DOIs in each, and the params that make the response complete —
rather than timing.
"""

from unittest.mock import MagicMock

import pytest
from conftest import CROSSREF_DEFAULT_ROWS, DummyContext, FakeZotero, fake_crossref_get

from zotero_mcp.tools import write


def _message(doi):
    return {
        "type": "journal-article",
        "title": [f"Paper {doi}"],
        "DOI": doi,
        "author": [{"given": "A", "family": "Author"}],
    }


@pytest.fixture
def zot(monkeypatch):
    fake = FakeZotero()
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake, fake)
    )
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._try_attach_oa_pdf",
        lambda *a, **kw: "skipped (test)",
    )
    return fake


def _record_calls(monkeypatch, message_for):
    """Patch requests.get with the CrossRef fake, recording every call."""
    calls = []
    inner = fake_crossref_get(message_for)

    def _get(url, params=None, **kwargs):
        calls.append({"url": url, "params": params or {}})
        return inner(url, params=params, **kwargs)

    monkeypatch.setattr("requests.get", _get)
    return calls


def _filtered_dois(call):
    return [tok[4:] for tok in call["params"]["filter"].split(",")
            if tok.startswith("doi:")]


class TestBatchedFetch:
    def test_many_dois_cost_one_request(self, monkeypatch, zot):
        calls = _record_calls(monkeypatch, _message)
        dois = [f"10.1234/d{i}" for i in range(25)]

        result = write.add_by_doi(doi=dois, ctx=DummyContext())

        assert len(calls) == 1
        assert calls[0]["url"] == "https://api.crossref.org/works"
        assert _filtered_dois(calls[0]) == dois
        assert len(zot.created) == 25
        assert "# Added 25 DOIs" in result

    def test_rows_is_set_to_the_chunk_size(self, monkeypatch, zot):
        """CrossRef pages at 20 by default, so a 25-DOI filter without an
        explicit `rows` silently drops 5 — they come back looking like they
        aren't in CrossRef at all."""
        calls = _record_calls(monkeypatch, _message)
        dois = [f"10.1234/d{i}" for i in range(25)]

        write.add_by_doi(doi=dois, ctx=DummyContext())

        assert int(calls[0]["params"]["rows"]) == 25
        assert int(calls[0]["params"]["rows"]) > CROSSREF_DEFAULT_ROWS
        assert len(zot.created) == 25

    def test_chunks_at_50(self, monkeypatch, zot):
        calls = _record_calls(monkeypatch, _message)
        dois = [f"10.1234/d{i}" for i in range(120)]

        write.add_by_doi(doi=dois, ctx=DummyContext())

        assert [int(c["params"]["rows"]) for c in calls] == [50, 50, 20]
        assert [len(_filtered_dois(c)) for c in calls] == [50, 50, 20]
        # Every DOI appears exactly once across the chunks, in order.
        assert [d for c in calls for d in _filtered_dois(c)] == dois
        assert len(zot.created) == 120

    def test_mailto_always_sent(self, monkeypatch, zot):
        """The polite pool is opt-in via mailto; an unconfigured install
        must not silently fall into the anonymous pool."""
        monkeypatch.delenv("ZOTERO_MCP_CONTACT_EMAIL", raising=False)
        calls = _record_calls(monkeypatch, _message)

        write.add_by_doi(doi=["10.1234/a", "10.1234/b"], ctx=DummyContext())

        assert calls[0]["params"]["mailto"]

    def test_contact_email_overrides_the_default_mailto(self, monkeypatch, zot):
        monkeypatch.setenv("ZOTERO_MCP_CONTACT_EMAIL", "me@example.org")
        calls = _record_calls(monkeypatch, _message)

        write.add_by_doi(doi=["10.1234/a", "10.1234/b"], ctx=DummyContext())

        assert calls[0]["params"]["mailto"] == "me@example.org"

    def test_single_doi_also_sends_mailto(self, monkeypatch, zot):
        monkeypatch.delenv("ZOTERO_MCP_CONTACT_EMAIL", raising=False)
        calls = _record_calls(monkeypatch, _message)

        write.add_by_doi(doi="10.1234/solo", ctx=DummyContext())

        assert calls[0]["url"].endswith("/works/10.1234/solo")
        assert calls[0]["params"]["mailto"]

    def test_repeated_doi_is_fetched_once(self, monkeypatch, zot):
        calls = _record_calls(monkeypatch, _message)

        write.add_by_doi(doi=["10.1234/a", "10.1234/b", "10.1234/a"],
                         ctx=DummyContext())

        assert _filtered_dois(calls[0]) == ["10.1234/a", "10.1234/b"]


class TestNotFoundHandling:
    def test_missing_doi_reported_per_entry(self, monkeypatch, zot):
        """A DOI CrossRef doesn't return is absent from message.items rather
        than producing a 404, so it has to be recovered by diffing what came
        back against what was asked for."""
        known = {f"10.1234/d{i}" for i in range(4)}
        calls = _record_calls(
            monkeypatch, lambda doi: _message(doi) if doi in known else None
        )
        dois = sorted(known) + ["10.9999/missing"]

        result = write.add_by_doi(doi=dois, ctx=DummyContext())

        assert len(calls) == 1
        assert len(zot.created) == 4
        assert "DOI not found on CrossRef: 10.9999/missing" in result
        for doi in sorted(known):
            assert f"Successfully added: **Paper {doi}**" in result

    def test_matching_is_case_insensitive(self, monkeypatch, zot):
        """CrossRef echoes DOIs in canonical case, which need not match what
        the caller typed — comparing raw would report a found DOI as
        missing."""
        _record_calls(monkeypatch, lambda doi: _message(doi.upper()))

        result = write.add_by_doi(doi=["10.1234/aBc", "10.1234/dEf"],
                                  ctx=DummyContext())

        assert len(zot.created) == 2
        assert "not found on CrossRef" not in result

    def test_batch_request_failure_reported_per_doi(self, monkeypatch, zot):
        """One request covers the whole chunk, so when it fails outright the
        batch must still return one line per requested DOI."""
        def _boom(url, params=None, **kwargs):
            resp = MagicMock()
            resp.status_code = 404  # a hard 404 on the collection endpoint
            resp.json.return_value = {}
            return resp

        monkeypatch.setattr("requests.get", _boom)

        result = write.add_by_doi(doi=["10.1234/a", "10.1234/b"],
                                  ctx=DummyContext())

        assert len(zot.created) == 0
        assert "DOI not found on CrossRef: 10.1234/a" in result
        assert "DOI not found on CrossRef: 10.1234/b" in result


class TestRateLimitBackstop:
    def test_429_is_retried_then_succeeds(self, monkeypatch, zot):
        monkeypatch.setattr(write._time, "sleep", lambda _s: None)
        ok = fake_crossref_get(_message)
        seen = []

        def _get(url, params=None, **kwargs):
            seen.append(url)
            if len(seen) == 1:
                resp = MagicMock()
                resp.status_code = 429
                return resp
            return ok(url, params=params, **kwargs)

        monkeypatch.setattr("requests.get", _get)

        result = write.add_by_doi(doi=["10.1234/a", "10.1234/b"],
                                  ctx=DummyContext())

        assert len(seen) == 2
        assert len(zot.created) == 2
        assert "# Added 2 DOIs" in result

    def test_429_retry_is_bounded(self, monkeypatch, zot):
        monkeypatch.setattr(write._time, "sleep", lambda _s: None)
        seen = []

        def _always_429(url, params=None, **kwargs):
            seen.append(url)
            resp = MagicMock()
            resp.status_code = 429
            return resp

        monkeypatch.setattr("requests.get", _always_429)

        result = write.add_by_doi(doi=["10.1234/a", "10.1234/b"],
                                  ctx=DummyContext())

        assert len(seen) == write._CROSSREF_MAX_ATTEMPTS
        assert len(zot.created) == 0
        assert "HTTP 429" in result
