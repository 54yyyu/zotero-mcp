"""YAML attachments enter the ordinary text extraction path (#703)."""

from pathlib import Path

import pytest

from zotero_mcp.extract import categorize_attachment, extract_file, is_extractable


@pytest.mark.parametrize(
    "name,ctype",
    [
        ("notes.yaml", None),
        ("notes.yml", "application/octet-stream"),
        ("notes.YAML", None),
        ("notes.YML", None),
        ("attachment", "application/yaml"),
        ("attachment", "text/yaml"),
    ],
)
def test_yaml_attachment_can_be_selected_and_read(tmp_path, name, ctype):
    attachment = tmp_path / name
    text = "title: '\u7814\u7a76'\nitems:\n  - &paper {year: 2026}\n  - *paper\n"
    attachment.write_bytes(text.encode("utf-8"))

    assert is_extractable(attachment, ctype)
    assert categorize_attachment(attachment, ctype) == "text"
    doc = extract_file(attachment, ctype)
    assert doc is not None
    assert doc.text == text
    assert doc.pages == (text,)
    assert doc.source == "text"


@pytest.mark.parametrize("name", ["paper.docx", "video.mp4"])
def test_binary_attachments_stay_unreadable(name):
    assert not is_extractable(Path(name), "application/octet-stream")
    assert categorize_attachment(Path(name), "application/octet-stream") is None
