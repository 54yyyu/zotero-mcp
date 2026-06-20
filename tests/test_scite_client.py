"""Tests for the Scite public API client."""

from unittest.mock import MagicMock, patch

from zotero_mcp import scite_client


def _ok_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def test_get_papers_batch_sends_bare_list_payload():
    """get_papers_batch must POST a bare JSON list, not {"dois": [...]}.

    The Scite /papers batch endpoint returns HTTP 400 for a dict-wrapped
    payload, which surfaced upstream as "Could not reach Scite API" from
    scite_check_retractions. Regression test for
    https://github.com/54yyyu/zotero-mcp/issues/352
    """
    with patch.object(
        scite_client.requests,
        "post",
        return_value=_ok_response({"papers": {"10.1038/x": {"title": "T"}}}),
    ) as mock_post:
        result = scite_client.get_papers_batch(["10.1038/x"])

    _, kwargs = mock_post.call_args
    # Bare list, NOT {"dois": [...]}
    assert kwargs["json"] == ["10.1038/x"]
    assert result == {"10.1038/x": {"title": "T"}}


def test_get_papers_batch_returns_empty_on_non_200():
    """A non-200 response yields an empty dict instead of raising."""
    bad = MagicMock()
    bad.status_code = 400
    with patch.object(scite_client.requests, "post", return_value=bad):
        assert scite_client.get_papers_batch(["10.1038/x"]) == {}


def test_get_papers_batch_empty_input_skips_request():
    """An empty DOI list short-circuits without hitting the network."""
    with patch.object(scite_client.requests, "post") as mock_post:
        assert scite_client.get_papers_batch([]) == {}
    mock_post.assert_not_called()


def test_get_tallies_batch_sends_bare_list_payload():
    """Regression guard: the tallies batch endpoint already uses a bare list."""
    with patch.object(
        scite_client.requests,
        "post",
        return_value=_ok_response({"tallies": {"10.1038/x": {"total": 1}}}),
    ) as mock_post:
        scite_client.get_tallies_batch(["10.1038/x"])

    _, kwargs = mock_post.call_args
    assert kwargs["json"] == ["10.1038/x"]
