"""Tests for PDF area/image annotation creation."""

import json
import sys
import types

import pytest
from conftest import DummyContext

from zotero_mcp import server


class FakePage:
    """Minimal fitz page stub with geometry and optional label."""

    def __init__(self, width=600, height=800, label="1", transformation_matrix=None):
        self.rect = types.SimpleNamespace(width=width, height=height)
        self.transformation_matrix = transformation_matrix or (
            1.0, 0.0, 0.0, -1.0, 0.0, float(height),
        )
        self._label = label

    def get_label(self):
        return self._label


class FakeDocument:
    """Minimal fitz document stub for page geometry lookups."""

    def __init__(self, pages):
        self._pages = pages
        self.is_pdf = True

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        return None


def _patch_fitz(monkeypatch, pages):
    """Patch fitz in sys.modules so imports inside pdf_utils succeed."""
    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda *_args, **_kwargs: FakeDocument(pages)
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)


def _pdf_attachment(key="ATTACH01", content_type="application/pdf"):
    return {
        "key": key,
        "data": {
            "itemType": "attachment",
            "contentType": content_type,
            "filename": "paper.pdf",
            "title": "paper.pdf",
            "parentItem": "PARENT01",
        },
    }


def test_create_area_annotation_happy_path(monkeypatch, fake_zot):
    fake_zot._items = [_pdf_attachment()]

    monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: fake_zot)
    monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)
    monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: None)
    _patch_fitz(monkeypatch, [FakePage(width=600, height=800, label="7")])

    result = server.create_area_annotation(
        attachment_key="ATTACH01",
        page=1,
        x=0.1,
        y=0.2,
        width=0.3,
        height=0.4,
        comment="Figure detail",
        ctx=DummyContext(),
    )

    assert "Successfully created area annotation" in result
    assert "**Annotation Key:** KEY0000" in result
    assert "**Page:** 7" in result
    assert "x=0.1000, y=0.2000, width=0.3000, height=0.4000" in result

    created = fake_zot.created[0]
    assert created["annotationType"] == "image"
    assert "annotationText" not in created
    assert created["annotationComment"] == "Figure detail"
    assert created["annotationPageLabel"] == "7"
    assert created["annotationSortIndex"] == "00000|000320|00060"

    position = json.loads(created["annotationPosition"])
    assert position == {
        "pageIndex": 0,
        "rects": [[60.0, 320.0, 240.0, 640.0]],
    }


def test_create_area_annotation_offset_mediabox(monkeypatch, fake_zot):
    """Rects must land in native PDF user space, not CropBox-normalized space.

    Page modeled after MediaBox [41.9, 46.5, 637.2, 840.2] with CropBox
    [41.9, 0, 637.2, 793.7]: PyMuPDF reports rect (0, 0, 595.3, 793.7) and
    transformation_matrix (1, 0, 0, -1, -41.9, 840.2).
    """
    fake_zot._items = [_pdf_attachment()]

    monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: fake_zot)
    monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)
    monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: None)
    _patch_fitz(
        monkeypatch,
        [FakePage(
            width=595.3,
            height=793.7,
            transformation_matrix=(1.0, 0.0, 0.0, -1.0, -41.9, 840.2),
        )],
    )

    result = server.create_area_annotation(
        attachment_key="ATTACH01",
        page=1,
        x=0.1,
        y=0.2,
        width=0.3,
        height=0.4,
        ctx=DummyContext(),
    )

    assert "Successfully created area annotation" in result

    created = fake_zot.created[0]
    position = json.loads(created["annotationPosition"])
    assert position["pageIndex"] == 0

    # PyMuPDF-space bbox is (59.53, 158.74, 238.12, 476.22); mapping through
    # the inverse transformation matrix adds the MediaBox origin: x + 41.9,
    # y -> 840.2 - y.
    (rect,) = position["rects"]
    assert rect == pytest.approx([101.43, 363.98, 280.02, 681.46], abs=1e-4)
    assert created["annotationSortIndex"] == "00000|000363|00101"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": 1.1, "y": 0.2, "width": 0.1, "height": 0.1}, "x must be between 0 and 1"),
        ({"x": 0.1, "y": 0.2, "width": 0.0, "height": 0.1}, "width must be greater than 0"),
        (
            {"x": 0.9, "y": 0.2, "width": 0.2, "height": 0.1},
            "Rectangle must fit within the page width",
        ),
    ],
)
def test_create_area_annotation_rejects_invalid_rectangles(kwargs, message):
    result = server.create_area_annotation(
        attachment_key="ATTACH01",
        page=1,
        comment=None,
        color="#ffd400",
        ctx=DummyContext(),
        **kwargs,
    )

    assert message in result


def test_create_area_annotation_rejects_non_pdf_attachment(monkeypatch, fake_zot):
    fake_zot._items = [_pdf_attachment(content_type="text/html")]

    monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: fake_zot)
    monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)
    monkeypatch.setattr("zotero_mcp.client.get_active_library", lambda: None)

    result = server.create_area_annotation(
        attachment_key="ATTACH01",
        page=1,
        x=0.1,
        y=0.2,
        width=0.3,
        height=0.4,
        ctx=DummyContext(),
    )

    assert "not a PDF attachment" in result


def test_create_area_annotation_requires_web_api(monkeypatch):
    monkeypatch.setattr("zotero_mcp.client.get_web_zotero_client", lambda: None)
    monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)

    result = server.create_area_annotation(
        attachment_key="ATTACH01",
        page=1,
        x=0.1,
        y=0.2,
        width=0.3,
        height=0.4,
        ctx=DummyContext(),
    )

    assert "Web API credentials required for creating annotations" in result

