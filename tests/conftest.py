"""Shared test fixtures for Zotero MCP tests."""

import pytest


class DummyContext:
    """No-op MCP context for unit tests."""

    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZotero:
    """Minimal pyzotero client stub. Extend per test file as needed."""

    def __init__(self):
        self.created = []
        self.updated = []
        self._items = []
        self._collections = []
        self._children = {}
        self.library_id = "12345"
        self.library_type = "user"

    def item(self, item_key):
        for it in self._items:
            if it.get("key") == item_key:
                return it
        return {"key": item_key, "version": 1, "data": {"title": "Item " + item_key}}

    def items(self, **kwargs):
        return self._items

    def collections(self, **kwargs):
        return self._collections

    def children(self, item_key, **kwargs):
        return self._children.get(item_key, [])

    def create_items(self, items, **kwargs):
        self.created.extend(items)
        result = {}
        for i, item in enumerate(items):
            result[str(i)] = f"KEY{i:04d}"
        return {"success": result, "successful": {}, "failed": {}}

    def create_collections(self, colls, **kwargs):
        result = {}
        for i, c in enumerate(colls):
            result[str(i)] = f"COL{i:04d}"
        return {"success": result, "successful": {}, "failed": {}}

    def update_item(self, item, **kwargs):
        self.updated.append(item)
        # Simulate httpx.Response
        return _FakeResponse(204)

    def item_template(self, item_type):
        """Return a minimal Zotero item template."""
        base = {
            "itemType": item_type,
            "title": "",
            "creators": [],
            "tags": [],
            "collections": [],
            "relations": {},
            "date": "",
            "abstractNote": "",
            "url": "",
            "DOI": "",
            "extra": "",
        }
        if item_type in ("journalArticle", "preprint"):
            base.update({
                "publicationTitle": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "ISSN": "",
                "publisher": "",
                "language": "",
                "shortTitle": "",
            })
        if item_type == "book":
            base.update({
                "publisher": "",
                "place": "",
                "ISBN": "",
                "numPages": "",
                "edition": "",
                "volume": "",
                "ISSN": "",
                "language": "",
                "shortTitle": "",
            })
        if item_type == "bookSection":
            base.update({
                "bookTitle": "",
                "publisher": "",
                "place": "",
                "ISBN": "",
                "pages": "",
                "edition": "",
                "volume": "",
                "ISSN": "",
                "language": "",
                "shortTitle": "",
            })
        return base

    def addto_collection(self, collection_key, items, **kwargs):
        return _FakeResponse(204)

    def deletefrom_collection(self, collection_key, item, **kwargs):
        return _FakeResponse(204)

    def everything(self, method, *args, **kwargs):
        if callable(method):
            return method(*args, **kwargs)
        return method

    def collection_items(self, key, **kwargs):
        return [it for it in self._items
                if key in it.get("data", {}).get("collections", [])]

    def file(self, key, **kwargs):
        return b""

    def dump(self, key, filename=None, path=None):
        """Create a dummy file so code that checks os.path.exists passes."""
        if path and filename:
            import os
            filepath = os.path.join(path, filename)
            with open(filepath, "wb") as f:
                f.write(b"%PDF-1.4 fake")
        return None

    def num_collectionitems(self, key):
        return len(self.collection_items(key))


class _FakeResponse:
    """Minimal httpx.Response stub."""

    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text

    @property
    def is_success(self):
        return 200 <= self.status_code < 300


@pytest.fixture
def dummy_ctx():
    return DummyContext()


@pytest.fixture
def fake_zot():
    return FakeZotero()
