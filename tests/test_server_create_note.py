from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
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
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

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
