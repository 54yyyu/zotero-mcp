from zotero_mcp import server


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
    assert note_html.startswith('<div data-schema-version="8"><h1>&lt;Unsafe Title&gt;</h1>')
    assert "<p>Line one</p>" in note_html


def test_create_note_markdown_becomes_note_html(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)

    result = server.create_note(
        item_key="ITEM0001",
        note_title="",
        note_text="## Heading\n\n**bold**, <u>under</u>, <span style=\"color: red\">red</span> and $x^2$\n\n- a list item",
        tags=None,
        ctx=DummyContext(),
    )

    assert "Successfully created note" in result
    assert "Markdown" not in result
    note_html = fake_zot.created[0]["note"]
    assert note_html == (
        '<div data-schema-version="9"><h2>Heading</h2>\n<p><strong>bold</strong>, <u>under</u>, '
        '<span style="color: #ff2020">red</span> and <span class="math">$x^2$</span></p>\n'
        "<ul>\n<li>a list item</li>\n</ul></div>"
    )


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
