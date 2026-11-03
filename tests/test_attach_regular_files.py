"""Attaching an existing file must not change new-item import routing (#761)."""

import mimetypes
from pathlib import Path

import pytest
from conftest import DummyContext
from test_attach_file import FakeZoteroForAttach, _patch_write_client

from zotero_mcp.tools import _helpers, write


@pytest.mark.parametrize("link_mode", ["import", "linked"])
@pytest.mark.parametrize(
    "name,content_type",
    [
        ("captions.VTT", "text/vtt"),
        ("notes.txt", "text/plain"),
        ("diagram.png", "image/png"),
        ("data.json", "application/json"),
        ("data.unknown-zotero-type", "application/octet-stream"),
        ("README", "application/octet-stream"),
    ],
)
def test_attach_regular_file(name, content_type, link_mode, tmp_path, monkeypatch):
    path = tmp_path / name
    content = b"original attachment bytes\x00\xff"
    path.write_bytes(content)
    fake = FakeZoteroForAttach()
    _patch_write_client(monkeypatch, fake)
    sent = []

    def upload(zot, filename, file_path, parent, ctx, content_type=None):
        sent.append((filename, Path(file_path).read_bytes(), parent, content_type))
        return True, "", "ATT00001"

    monkeypatch.setattr(_helpers, "_attach_and_verify", upload)
    result = write.attach_file(
        item_key="ITEM1", file_path=str(path), link_mode=link_mode, ctx=DummyContext()
    )

    assert "Error" not in result
    if link_mode == "import":
        assert sent == [(name, content, "ITEM1", content_type)]
        assert fake.created == []
    else:
        assert sent == []
        (created,) = fake.created
        assert created["linkMode"] == "linked_file"
        assert created["path"] == str(path.resolve())
        assert created["contentType"] == content_type
        assert created["parentItem"] == "ITEM1"
    assert path.read_bytes() == content


@pytest.mark.parametrize("link_mode", ["import", "linked"])
def test_rejects_windows_shortcut(link_mode, tmp_path, monkeypatch):
    path = tmp_path / "shortcut.LNK"
    path.write_bytes(b"not a regular attachment")
    fake = FakeZoteroForAttach()
    _patch_write_client(monkeypatch, fake)
    result = write.attach_file(
        item_key="ITEM1", file_path=str(path), link_mode=link_mode, ctx=DummyContext()
    )
    assert "shortcut" in result.lower()
    assert fake.created == [] and fake.attachments == []


def test_linked_mime_comes_from_file_not_title(tmp_path, monkeypatch):
    path = tmp_path / "diagram.png"
    path.write_bytes(b"image bytes")
    fake = FakeZoteroForAttach()
    _patch_write_client(monkeypatch, fake)
    result = write.attach_file(
        item_key="ITEM1", file_path=str(path), filename="Figure.pdf",
        link_mode="linked", ctx=DummyContext(),
    )
    assert "Linked file attached" in result
    assert fake.created[0]["contentType"] == "image/png"


@pytest.mark.parametrize(
    "extension,source_type", [("json", "csl_json"), ("bib", "bibtex"), ("md", "file"), ("yaml", "file")]
)
def test_add_import_routing_is_preserved(extension, source_type, tmp_path):
    path = tmp_path / f"source.{extension}"
    path.write_text("{}", encoding="utf-8")
    assert write.detect_source_type(str(path)) == source_type


def test_mime_falls_back_to_standard_database(monkeypatch):
    monkeypatch.setattr(mimetypes, "guess_type", lambda filename: ("image/png", None))
    assert _helpers._guess_content_type("figure.png") == "image/png"
    # Explicit text mappings must still win over platform MIME registrations.
    assert _helpers._guess_content_type("notes.yaml") == "text/yaml"
