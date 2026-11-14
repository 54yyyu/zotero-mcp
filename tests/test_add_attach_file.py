"""`zotero-cli add doi --attach-file` and `add folder`: one-step course import.

The write tools are replaced with fakes, so nothing reaches a library.
"""

import argparse
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from zotero_mcp import cli_standalone
from zotero_mcp.pdf_utils import doi_from_pdf

fitz = pytest.importorskip("fitz")


def make_pdf(path, first_page_lines=(), metadata=None, filler=0):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 60
    for _ in range(filler):                       # push the DOI well past the first 3000 characters
        page.insert_text((40, y), "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod.", fontsize=7)
        y += 8
    for line in first_page_lines:
        page.insert_text((40, y), line, fontsize=8)
        y += 10
    if metadata:
        doc.set_metadata(metadata)
    doc.save(str(path))
    doc.close()
    return str(path)


class TestDoiFromPdf:
    def test_metadata_wins_when_page_1_agrees_or_is_silent(self, tmp_path):
        meta = {"subject": "doi:10.1016/j.cell.2020.01.001"}
        path = make_pdf(tmp_path / "a.pdf", ["https://doi.org/10.1016/j.cell.2020.01.001",
                                             "see 10.9999/other.article.1 for details"], metadata=meta)
        assert doi_from_pdf(path) == ("10.1016/j.cell.2020.01.001", "metadata")
        path = make_pdf(tmp_path / "a2.pdf", ["no identifiers printed"], metadata=meta)
        assert doi_from_pdf(path) == ("10.1016/j.cell.2020.01.001", "metadata")

    def test_metadata_that_page_1_contradicts_is_ambiguous(self, tmp_path):
        path = make_pdf(tmp_path / "a3.pdf", ["https://doi.org/10.1016/j.cell.2020.02.002"],
                        metadata={"subject": "doi:10.1016/j.cell.2020.01.001"})
        doi, why = doi_from_pdf(path)
        assert doi is None and why.startswith("ambiguous") and "02.002" in why

    def test_the_whole_first_page_is_read_not_its_first_3000_characters(self, tmp_path):
        path = make_pdf(tmp_path / "b.pdf", ["DOI: https://doi.org/10.1038/s41586-021-03819-2"], filler=70)
        with fitz.open(path) as doc:
            assert len(doc[0].get_text()) > 3000 and doc[0].get_text().find("10.1038") > 3000
        assert doi_from_pdf(path)[0] == "10.1038/s41586-021-03819-2"

    def test_a_journal_spread_with_two_articles_is_ambiguous_not_a_guess(self, tmp_path):
        path = make_pdf(tmp_path / "c.pdf", ["https://doi.org/10.1016/j.cell.2020.01.001",
                                             "https://doi.org/10.1016/j.cell.2020.02.002"])
        doi, why = doi_from_pdf(path)
        assert doi is None and why.startswith("ambiguous") and "02.002" in why

    def test_naming_one_article_more_often_is_still_ambiguous(self, tmp_path):
        """A spread's neighbour can be named twice (header and footer) and this article once."""
        path = make_pdf(tmp_path / "d.pdf", ["10.1016/j.cell.2020.01.001", "10.1016/j.cell.2020.01.001",
                                             "10.1016/j.cell.2020.02.002"])
        assert doi_from_pdf(path)[0] is None

    def test_a_doi_wrapped_at_its_hyphen_is_read_whole(self, tmp_path):
        path = make_pdf(tmp_path / "w.pdf", ["https://doi.org/10.1038/s41586-", "021-03819-2"])
        assert doi_from_pdf(path) == ("10.1038/s41586-021-03819-2", "page 1")

    def test_no_doi(self, tmp_path):
        doi, why = doi_from_pdf(make_pdf(tmp_path / "e.pdf", ["nothing here"]))
        assert doi is None and "no DOI" in why


def args_for(**kwargs):
    defaults = dict(verbose=False, json_out=False, subcommand="doi", tags=None, collections=None,
                    collection=None, if_exists="file", create_collections=False, attach_mode=None,
                    attach_file=None, dry_run=False)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def run(args, write_mod):
    with patch("zotero_mcp.cli_standalone.setup_zotero_environment"), \
         patch("zotero_mcp.cli_standalone._import_tools",
               return_value=(MagicMock(), MagicMock(), MagicMock(), write_mod, MagicMock())):
        cli_standalone.cmd_add(args)


ADDED = "Successfully added: **A Paper**\n\nItem key: `ABCD1234`\nDOI: 10.1/x"


class TestAttachFile:
    def test_one_call_adds_and_attaches_without_fetching_a_second_copy(self, tmp_path, capsys):
        pdf = make_pdf(tmp_path / "paper.pdf", ["x"])
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = ADDED
        write_mod.attach_file.return_value = "File attached to `ABCD1234`: paper.pdf"
        run(args_for(doi="10.1/x", attach_file=pdf, tags="week-3", collection=["Course"]), write_mod)

        add_kwargs = write_mod.add_by_doi.call_args.kwargs
        assert add_kwargs["attach_mode"] == "none"           # the file in hand is the attachment
        assert add_kwargs["tags"] == ["week-3"] and add_kwargs["collections"] == ["Course"]
        attach_kwargs = write_mod.attach_file.call_args.kwargs
        assert attach_kwargs["item_key"] == "ABCD1234" and attach_kwargs["file_path"] == os.path.abspath(pdf)
        out = capsys.readouterr().out
        assert "Item key: `ABCD1234`" in out and "File attached" in out

    def test_an_explicit_attach_mode_is_kept(self, tmp_path):
        pdf = make_pdf(tmp_path / "paper.pdf", ["x"])
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = ADDED
        write_mod.attach_file.return_value = "File attached"
        run(args_for(doi="10.1/x", attach_file=pdf, attach_mode="import_file"), write_mod)
        assert write_mod.add_by_doi.call_args.kwargs["attach_mode"] == "import_file"

    def test_without_attach_file_the_default_is_unchanged(self):
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = ADDED
        run(args_for(doi="10.1/x"), write_mod)
        assert write_mod.add_by_doi.call_args.kwargs["attach_mode"] == "auto"
        write_mod.attach_file.assert_not_called()

    def test_a_failed_attach_is_reported_as_a_failure(self, tmp_path, capsys):
        pdf = make_pdf(tmp_path / "paper.pdf", ["x"])
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = ADDED
        write_mod.attach_file.return_value = "Error: upload failed"
        with pytest.raises(SystemExit) as exit_info:
            run(args_for(doi="10.1/x", attach_file=pdf), write_mod)
        assert exit_info.value.code == 1
        assert "ABCD1234" in capsys.readouterr().err

    def test_a_failed_add_attaches_nothing(self, tmp_path):
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = "Error: CrossRef has no record of 10.1/x"
        with pytest.raises(SystemExit):
            run(args_for(doi="10.1/x", attach_file=make_pdf(tmp_path / "p.pdf", ["x"])), write_mod)
        write_mod.attach_file.assert_not_called()

    @pytest.mark.parametrize("make", ["missing", "symlink"])
    def test_a_file_attach_would_refuse_is_caught_before_the_item_is_added(self, tmp_path, make):
        from zotero_mcp.cli_json import CliError

        path = tmp_path / "p.pdf"
        if make == "symlink":
            path.symlink_to(make_pdf(tmp_path / "real.pdf", ["x"]))
        write_mod = MagicMock()
        with pytest.raises(CliError, match="not a regular file"):
            run(args_for(doi="10.1/x", attach_file=str(path)), write_mod)
        write_mod.add_by_doi.assert_not_called()

    def test_if_exists_skip_attaches_nothing_to_an_existing_item(self, tmp_path, capsys):
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = (
            "Already in library: **A Paper** (`ABCD1234`, matched by DOI)\n\nItem key: `ABCD1234`\n"
            "No changes made (if_exists='skip').")
        run(args_for(doi="10.1/x", attach_file=make_pdf(tmp_path / "p.pdf", ["x"]), if_exists="skip"),
            write_mod)
        write_mod.attach_file.assert_not_called()
        assert "not attached" in capsys.readouterr().out


class TestAddFolder:
    def folder(self, tmp_path):
        make_pdf(tmp_path / "good.pdf", ["10.1016/j.cell.2020.01.001"])
        make_pdf(tmp_path / "spread.pdf", ["10.1016/j.cell.2020.01.001", "10.1016/j.cell.2020.02.002"])
        make_pdf(tmp_path / "none.pdf", ["nothing"])
        (tmp_path / "notes.txt").write_text("not a pdf")
        return tmp_path

    def test_adds_resolved_files_and_lists_the_rest(self, tmp_path, capsys):
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = ADDED
        write_mod.attach_file.return_value = "File attached to `ABCD1234`: good.pdf"
        with pytest.raises(SystemExit) as exit_info:         # exit 1: some need attention
            run(args_for(subcommand="folder", folder=str(self.folder(tmp_path)), tags="week-3"), write_mod)
        assert exit_info.value.code == 1

        assert write_mod.add_by_doi.call_count == 1
        kwargs = write_mod.add_by_doi.call_args.kwargs
        assert kwargs["doi"] == "10.1016/j.cell.2020.01.001" and kwargs["attach_mode"] == "none"
        assert write_mod.attach_file.call_args.kwargs["file_path"].endswith("good.pdf")
        out = capsys.readouterr().out
        assert "good.pdf" in out and "added, file attached" in out
        assert "spread.pdf" in out and "ambiguous" in out
        assert "none.pdf" in out and "no DOI" in out
        assert "notes.txt" not in out and "1/3 resolved" in out

    def test_a_symlinked_pdf_is_listed_not_followed(self, tmp_path, capsys):
        folder = tmp_path / "in"
        folder.mkdir()
        (folder / "link.pdf").symlink_to(make_pdf(tmp_path / "real.pdf", ["10.1016/j.cell.2020.01.001"]))
        write_mod = MagicMock()
        with pytest.raises(SystemExit):
            run(args_for(subcommand="folder", folder=str(folder)), write_mod)
        write_mod.add_by_doi.assert_not_called()
        assert "symlinks are not followed" in capsys.readouterr().out

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        write_mod = MagicMock()
        with pytest.raises(SystemExit):
            run(args_for(subcommand="folder", folder=str(self.folder(tmp_path)), dry_run=True), write_mod)
        write_mod.add_by_doi.assert_not_called()
        write_mod.attach_file.assert_not_called()
        assert "would add" in capsys.readouterr().out

    def test_json_lists_every_file(self, tmp_path, capsys):
        write_mod = MagicMock()
        write_mod.add_by_doi.return_value = ADDED
        write_mod.attach_file.return_value = "File attached"
        with pytest.raises(SystemExit):
            run(args_for(subcommand="folder", folder=str(self.folder(tmp_path)), json_out=True), write_mod)
        data = json.loads(capsys.readouterr().out)["data"]
        assert sorted(f["file"] for f in data["files"]) == ["good.pdf", "none.pdf", "spread.pdf"]
        assert data["unresolved"] == 2

    def test_an_empty_folder_is_a_usage_error(self, tmp_path):
        from zotero_mcp.cli_json import CliError

        with pytest.raises(CliError, match="No PDF"):
            run(args_for(subcommand="folder", folder=str(tmp_path)), MagicMock())
