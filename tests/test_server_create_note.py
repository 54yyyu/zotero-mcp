from zotero_mcp import server
from zotero_mcp.tools.annotations import _looks_like_markdown


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZotero:
    def __init__(self):
        self.created = []

    def item(self, _item_key):
        return {"data": {"title": "Parent Item"}}

    def create_items(self, items):
        self.created.extend(items)
        return {"success": {"0": "NOTEKEY01"}}


def test_create_note_includes_title_heading(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

    result = server.create_note(
        item_key="ITEM0001",
        note_title="<Unsafe Title>",
        note_text="Line one\n\nLine two",
        tags=["t1"],
        ctx=DummyContext(),
    )

    assert "Successfully created note" in result
    assert len(fake_zot.created) == 1
    note_html = fake_zot.created[0]["note"]
    assert note_html.startswith("<h1>&lt;Unsafe Title&gt;</h1>")
    assert "<p>Line one</p>" in note_html


def test_create_note_markdown_input_warns_and_stores_verbatim(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

    result = server.create_note(
        item_key="ITEM0001",
        note_title="",
        note_text="## Heading\n\n**bold** and a - list item",
        tags=None,
        ctx=DummyContext(),
    )

    assert "Successfully created note" in result
    assert "looks like Markdown" in result
    note_html = fake_zot.created[0]["note"]
    assert "## Heading" in note_html
    assert "<h2>" not in note_html


def test_create_note_plain_text_has_no_markdown_warning(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

    result = server.create_note(
        item_key="ITEM0001",
        note_title="",
        note_text="Line one\n\nLine two",
        tags=None,
        ctx=DummyContext(),
    )

    assert "Successfully created note" in result
    assert "Markdown" not in result


def test_create_note_html_passthrough_has_no_markdown_warning(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

    result = server.create_note(
        item_key="ITEM0001",
        note_title="",
        note_text="<p>2*3=6 and a-b are fine</p>",
        tags=None,
        ctx=DummyContext(),
    )

    assert "Successfully created note" in result
    assert "Markdown" not in result


def test_looks_like_markdown_cases():
    assert _looks_like_markdown("## Heading")
    assert _looks_like_markdown("- item\n- item")
    assert _looks_like_markdown("1. first\n2. second")
    assert _looks_like_markdown("**bold** and `code`")
    assert _looks_like_markdown("> quote")
    assert _looks_like_markdown("[text](https://example.com)")
    assert _looks_like_markdown("~~struck~~")
    assert not _looks_like_markdown("Line one\n\nLine two")
    assert not _looks_like_markdown("2*3=6 and file_name here")
    assert not _looks_like_markdown("a-b and 10.1007/s11142-021-09582-z")
