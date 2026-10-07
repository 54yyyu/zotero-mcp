"""`zotero-cli read ITEM --find TEXT`: locate a passage without returning pages."""

import argparse

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp.cli_standalone import build_parser
from zotero_mcp.tools import read_pdf as read_pdf_mod
from zotero_mcp.tools.read_pdf import PdfReadError, find_in_pages, format_find


def find(pages, query, **kwargs):
    return find_in_pages(range(len(pages)), pages, query, **kwargs)


def pages_of(result):
    return [m["page"] for m in result["matches"]]


class TestMatcher:
    def test_ignores_case_whitespace_and_punctuation(self):
        r = find(["We run a Robustness\n  Check, here."], "robustness   check")
        assert pages_of(r) == [1]

    def test_ignores_line_break_hyphenation(self):
        assert pages_of(find(["a robust-\nness check"], "robustness")) == [1]
        assert pages_of(find(["a robust­ness check"], "robustness")) == [1]

    def test_ignores_ligatures(self):
        assert pages_of(find(["the ﬁrst result"], "first")) == [1]
        assert pages_of(find(["the first result"], "ﬁrst")) == [1]

    def test_match_must_start_a_word(self):
        assert find(["we start here"], "art")["matches"] == []
        # Mid-word after a line-break hyphen, with or without a newline.
        assert find(["robust-\nness"], "ness")["matches"] == []
        assert find(["robust- ness"], "ness")["matches"] == []
        assert pages_of(find(["art at the start"], "art")) == [1]
        assert pages_of(find(["a well-known fact"], "known")) == [1]

    def test_prefix_match_shows_the_whole_word(self):
        r = find(["Statistical checks pass."], "check")
        assert r["matches"][0]["snippets"] == ["Statistical checks pass."]

    def test_ranks_by_hits_then_page(self):
        r = find(["cat", "cat cat", "dog", "cat cat"], "cat")
        assert pages_of(r) == [2, 4, 1]
        assert [m["hits"] for m in r["matches"]] == [2, 2, 1]
        assert r["hits"] == 5 and r["mode"] == "phrase"

    def test_page_numbers_are_one_based_from_the_given_indices(self):
        r = find_in_pages([6, 7], ["x", "cat"], "cat")
        assert pages_of(r) == [8]

    def test_caps_pages_and_snippets(self):
        pages = [" ".join(["cat"] * (i + 1)) for i in range(14)]
        r = find(pages, "cat")
        assert len(r["matches"]) == 10
        assert pages_of(r)[0] == 14
        assert all(len(m["snippets"]) <= 3 for m in r["matches"])
        assert r["matches"][0]["hits"] == 14
        assert r["more_pages"] == [1, 2, 3, 4]
        assert r["hits"] == sum(range(1, 15))

    def test_words_mode_when_the_phrase_is_nowhere(self):
        pages = ["alpha only", "beta only", "alpha gamma beta gamma alpha", "beta alpha"]
        r = find(pages, "alpha beta")
        assert r["mode"] == "words"
        assert pages_of(r) == [3, 4]
        assert r["matches"][0]["hits"] == 3

    def test_phrase_mode_wins_when_any_page_has_the_phrase(self):
        r = find(["alpha beta", "alpha beta x alpha beta"], "alpha beta")
        assert r["mode"] == "phrase" and pages_of(r) == [2, 1]

    def test_single_word_has_no_words_fallback(self):
        assert find(["alpha"], "beta")["matches"] == []

    @pytest.mark.parametrize("query", ["", "   ", "?! -- ."])
    def test_empty_query_is_an_error(self, query):
        with pytest.raises(PdfReadError) as err:
            find(["text"], query)
        assert err.value.code == "empty_query"

    def test_scanned_pdf_is_an_error_not_no_match(self):
        with pytest.raises(PdfReadError) as err:
            find(["", "  "], "x")
        assert err.value.code == "no_text_layer"

    def test_snippet_context_and_cleanup(self):
        words = [f"w{i}" for i in range(40)]
        text = " ".join(words[:20]) + " **TARGET** | `x` " + " ".join(words[20:])
        snippet = find([text], "target", context=3)["matches"][0]["snippets"][0]
        assert snippet == "…w17 w18 w19 TARGET x w20 w21…"

    def test_snippet_joins_hyphenation_and_has_no_ellipsis_at_page_edges(self):
        snippet = find(["a robust-\nness check"], "check")["matches"][0]["snippets"][0]
        assert snippet == "a robustness check"

    def test_context_is_clamped(self):
        text = " ".join(f"w{i}" for i in range(100)) + " target " + " ".join(f"v{i}" for i in range(100))
        snippet = find([text], "target", context=1000)["matches"][0]["snippets"][0]
        assert len(snippet.split()) == 121
        snippet = find([text], "target", context=0)["matches"][0]["snippets"][0]
        assert len(snippet.split()) == 3


class TestFormat:
    def result(self, **over):
        base = dict(item_key="K1", title="A Paper", total_pages=22,
                    **find(["x", "we run a robustness check", "robustness check"],
                           "robustness check"))
        base["pages_searched"] = 22
        base.update(over)
        return base

    def test_markdown_shape(self):
        lines = format_find(self.result(more_pages=[7, 9])).splitlines()
        assert lines[0] == '# "robustness check" in A Paper (K1): 2 hits on 4 of 22 pages'
        assert lines[1].startswith("p.2 (1 hit): we run a robustness check")
        assert "Also on pages 7, 9" in lines
        assert lines[-1] == "Read one: zotero-cli read K1 --start-page 2"

    def test_no_match_is_an_answer(self):
        r = self.result(matches=[], more_pages=[], hits=0, query="zzz")
        assert format_find(r) == 'No match for "zzz" in A Paper (22 pages).'

    def test_words_mode_is_named_in_the_header(self):
        r = self.result(mode="words")
        assert "no exact phrase; pages with all words" in format_find(r).splitlines()[0]


class TestCli:
    def test_find_needs_no_start_page(self):
        parsed = build_parser().parse_args(["read", "K1", "--find", "x"])
        assert parsed.find == "x" and parsed.start_page is None and parsed.context == 12

    def test_plain_read_still_needs_a_start_page(self, capsys):
        with pytest.raises(SystemExit):
            cli_standalone.cmd_read(build_parser().parse_args(["read", "K1"]))
        assert "start" in capsys.readouterr().err

    def test_find_with_image_is_an_error(self):
        parsed = build_parser().parse_args(["read", "K1", "--find", "x", "--format", "image"])
        with pytest.raises(cli_standalone._cli_json.CliError) as err:
            cli_standalone.cmd_read(parsed)
        assert err.value.code == "bad_find"

    def test_cmd_read_routes_find(self, monkeypatch, capsys):
        called = {}

        def fake(item_key, query, start_page=None, end_page=None, *, context, ctx):
            called.update(item_key=item_key, query=query, start=start_page, end=end_page,
                          context=context)
            return dict(item_key=item_key, title="T", total_pages=3,
                        **find(["a", "needle here", "b"], query))

        monkeypatch.setattr(read_pdf_mod, "find_in_pdf", fake)
        args = argparse.Namespace(verbose=False, json_out=False, item_key="K1", find="needle",
                                  start_page=None, end_page=None, context=5, format="text")
        from unittest.mock import patch
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            cli_standalone.cmd_read(args)
        assert called == dict(item_key="K1", query="needle", start=None, end=None, context=5)
        assert capsys.readouterr().out.startswith('# "needle" in T (K1): 1 hit on 1 of 3 pages')

    def test_cmd_read_find_json_shape(self, monkeypatch, capsys):
        import json
        from unittest.mock import patch

        monkeypatch.setattr(
            read_pdf_mod, "find_in_pdf",
            lambda item_key, query, *a, **k: dict(item_key=item_key, title="T", total_pages=2,
                                                  **find(["needle", "x"], query)))
        args = argparse.Namespace(verbose=False, json_out=True, item_key="K1", find="needle",
                                  start_page=None, end_page=None, context=12, format="text")
        with patch("zotero_mcp.cli_standalone.setup_zotero_environment"):
            cli_standalone.cmd_read(args)
        data = json.loads(capsys.readouterr().out)["data"]
        assert data["mode"] == "phrase" and data["total_pages"] == 2
        assert data["matches"] == [{"page": 1, "hits": 1, "snippets": ["needle"]}]
        assert data["more_pages"] == []
