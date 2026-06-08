class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class FakeFeedReader:
    def get_feeds(self):
        return [{
            "libraryID": 10,
            "name": "Example Feed",
            "url": "https://example.test/feed.xml",
        }]

    def get_feed_items(self, library_id, limit=20):
        assert library_id == 10
        assert limit == 20
        return [{
            "title": "Feed Item Title",
            "readTime": None,
            "creators": "Lovelace, Ada",
            "url": "https://example.test/item",
            "date": "2024-05-15",
            "DOI": "10.1234/example.doi",
            "dateAdded": "2026-06-01 10:00:00",
            "abstract": None,
        }]

    def close(self):
        return None


def test_get_feed_items_output_includes_publication_date(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")

    import zotero_mcp.local_db
    from zotero_mcp.tools import retrieval

    monkeypatch.setattr(zotero_mcp.local_db, "LocalZoteroReader", FakeFeedReader)
    result = retrieval.get_feed_items(library_id=10, limit=20, ctx=DummyContext())

    assert "- **Date:** 2024-05-15" in result
    assert "- **DOI:** 10.1234/example.doi" in result
    assert "- **Added:** 2026-06-01 10:00:00" in result
