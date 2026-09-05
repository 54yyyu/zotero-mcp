"""Shared test fixtures for Zotero MCP tests."""

import os
import sys
from pathlib import Path

import pytest
from pyzotero.errors import CallDoesNotExistError

# Always exercise the source tree that these tests live in, not whatever
# `zotero_mcp` an editable install happens to resolve to. Without this, running
# the suite from a git worktree silently imports the *main* checkout's package,
# so edits under test are never executed and results are meaningless.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
    for _name in [n for n in sys.modules if n == "zotero_mcp" or n.startswith("zotero_mcp.")]:
        del sys.modules[_name]

# Marker for tests that use tmp_path and fail on GitHub Actions
skip_on_ci = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="tmp_path fixture unreliable on GitHub Actions"
)

# Marker for tests whose *fixtures* hardcode POSIX paths ("/Users/test/x.pdf",
# "file:///…"). The code under test is cross-platform; the assertions are not,
# because Windows resolves those strings to "D:\Users\test\x.pdf". Skipped
# rather than deleted so the coverage stays real on Linux and macOS —
# rewriting them platform-neutrally is worth doing but is not this change.
skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="fixture hardcodes POSIX paths; the behaviour under test is not platform-specific",
)


@pytest.fixture(autouse=True)
def isolate_local_write_state(monkeypatch, tmp_path):
    """Keep local-write config and capability probing out of the tests.

    Two things would otherwise leak the developer's machine into the results:
    the local API key is read from ~/.config/zotero-mcp/config.json, so anyone
    who has authorized local writes would resolve a different write client
    than CI does; and the capability probe makes a real request to
    localhost:23119, so the outcome would depend on whether Zotero happens to
    be running. Both are pinned here; tests that exercise them override.
    """
    import time

    from zotero_mcp import client as _client

    monkeypatch.setattr(_client, "ZOTERO_MCP_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(_client, "_local_write_state", {})
    # Seed the probe cache with a negative answer rather than stubbing the
    # function, so tests can opt into either behaviour by editing the cache:
    # clear() to exercise the probe itself, or write a server_id to pretend
    # Zotero 10 is running.
    monkeypatch.setattr(
        _client,
        "_local_probe_cache",
        {"server_id": None, "checked_at": time.monotonic()},
    )
    for var in ("ZOTERO_LOCAL_API_KEY", "ZOTERO_LOCAL_SERVER_ID", "ZOTERO_LOCAL_WRITE"):
        monkeypatch.delenv(var, raising=False)


class DummyContext:
    """No-op MCP context for unit tests."""

    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZotero:
    """Minimal pyzotero client stub. Extend per test file as needed."""

    # pyzotero sets this on the instance; several helpers branch on it to tell
    # a local client from a web one.
    local = False
    endpoint = "https://api.zotero.org"

    def __init__(self):
        self.created = []
        self.updated = []
        self._items = []
        self._collections = []
        self._children = {}
        self.attached = []
        self.uploaded = []
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

    def item_template(self, item_type, linkmode=None):
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

    def attachment_both(self, pairs, parentid=None):
        self.attached.append((list(pairs), parentid))
        return {"success": [{"key": "ATT00001"}], "unchanged": [], "failure": []}

    def upload_attachments(self, attachments, parentid=None, basedir=None):
        self.uploaded.append((list(attachments), parentid))
        return {"success": [{"key": "ATT00001"}], "unchanged": [], "failure": []}

    def item_types(self):
        return [{"itemType": "journalArticle", "localized": "Journal Article"}]

    def item_type_fields(self, item_type):
        return [{"field": "title"}, {"field": "date"}, {"field": "abstractNote"}]

    def item_creator_types(self, item_type):
        return [{"creatorType": "author", "localized": "Author"}]


class FakeLocalZotero(FakeZotero):
    """A FakeZotero that behaves like a client talking to the local API.

    The local API has no /items/new, so pyzotero raises CallDoesNotExistError
    from item_template() — and, transitively, from attachment_both(). Writes
    go through _write(), which is what attaches the server-id and key headers.
    """

    local = True
    endpoint = "http://localhost:23119/api"

    def __init__(self, write_status=204):
        super().__init__()
        self.library_id = "0"
        self.library_type = "users"
        self.server_id = "test-server-id"
        self.local_api_key = "test-key"
        self.writes = []
        self.authorized = []
        self._write_status = write_status

    def item_template(self, item_type, linkmode=None):
        raise CallDoesNotExistError(
            "The local Zotero API doesn't implement the /items/new template endpoint"
        )

    def attachment_both(self, pairs, parentid=None):
        # Fails transitively through item_template, exactly as pyzotero does.
        return self.item_template("attachment", "imported_file")

    def _write(self, method, url, **kwargs):
        self.writes.append((method, url, kwargs))
        return _FakeResponse(self._write_status)

    def authorize_local(self, app_name):
        self.authorized.append(app_name)
        return {"key": "granted-key", "remember": True}


class _FakeResponse:
    """Minimal httpx.Response stub."""

    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text

    @property
    def is_success(self):
        return 200 <= self.status_code < 300


class _FakeCrossrefResponse:
    """Minimal requests.Response stub for the CrossRef API."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# CrossRef's real default page size. The fake honors it so that dropping the
# explicit `rows` param from a batched request fails the suite the same way it
# failed in production: the first 20 DOIs resolve and the rest look absent.
CROSSREF_DEFAULT_ROWS = 20


def fake_crossref_get(message_for):
    """Build a ``requests.get`` replacement serving both CrossRef shapes.

    ``message_for(doi)`` returns the ``/works`` message dict for a DOI, or
    None if CrossRef doesn't have it. Dispatches on the URL: ``/works/<doi>``
    is the single-DOI endpoint, bare ``/works`` is the batched
    ``filter=doi:...`` query, whose response is paged by ``rows`` exactly as
    the real API pages it.
    """
    def _get(url, params=None, **_kwargs):
        params = params or {}

        if "/works/" in url:
            doi = url.split("/works/", 1)[1]
            msg = message_for(doi)
            if msg is None:
                return _FakeCrossrefResponse(404)
            return _FakeCrossrefResponse(200, {"status": "ok", "message": msg})

        dois = [tok[4:] for tok in (params.get("filter") or "").split(",")
                if tok.startswith("doi:")]
        found = [msg for msg in (message_for(d) for d in dois) if msg is not None]
        rows = int(params.get("rows", CROSSREF_DEFAULT_ROWS))
        return _FakeCrossrefResponse(200, {
            "status": "ok",
            "message": {"total-results": len(found), "items": found[:rows]},
        })

    return _get


@pytest.fixture
def dummy_ctx():
    return DummyContext()


@pytest.fixture
def fake_zot():
    return FakeZotero()


@pytest.fixture
def fake_local_zot():
    return FakeLocalZotero()
