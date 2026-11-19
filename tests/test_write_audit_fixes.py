"""Regressions from the 0.17.3 write-path audit."""

from conftest import DummyContext

from zotero_mcp.tools import _helpers, write


class _VersionedZotero:
    """One item; like the real API, a write with a stale version is refused."""

    def __init__(self, collections=(), tags=(), extra=""):
        self.version = 1
        self.data = {"key": "ITEM0001", "itemType": "journalArticle",
                     "collections": list(collections),
                     "tags": [{"tag": t} for t in tags], "extra": extra}
        self.gets = 0

    def item(self, key):
        self.gets += 1
        return {"key": key, "version": self.version,
                "data": {**self.data, "version": self.version}}

    def addto_collection(self, coll_key, item):
        if item["version"] != self.version:
            raise RuntimeError("412 Precondition Failed")
        self.data["collections"] = item["data"]["collections"] + [coll_key]
        self.version += 1
        return True


def test_filing_into_several_collections_survives_the_version_bump():
    zot = _VersionedZotero(collections=["AAAA1111"])
    failed = _helpers.ensure_collection_membership(
        zot, "ITEM0001", ["AAAA1111", "BBBB2222", "CCCC3333"], DummyContext())
    assert failed == []
    assert zot.data["collections"] == ["AAAA1111", "BBBB2222", "CCCC3333"]
    assert zot.gets == 2  # one read up front, one after the first add


def _hybrid(monkeypatch, write_data):
    """Local reads, web writes: the web copy holds what the desktop has not synced."""
    from test_batch_write_requests import CountingZot, _install, _items

    read_zot = CountingZot(_items(1))
    web = _items(1)
    web[0]["version"] = 77
    web[0]["data"].update(write_data)
    write_zot = CountingZot(web)
    _install(monkeypatch, read_zot, write_zot)
    return write_zot


def test_batch_tag_edit_keeps_tags_only_the_write_side_has(monkeypatch):
    write_zot = _hybrid(monkeypatch, {"tags": [{"tag": "old"}, {"tag": "x", "type": 1}]})

    result = write.batch_update_tags(
        item_keys=["K0000000"], add_tags=["new"], remove_tags=["old"], ctx=DummyContext())

    assert "Items updated: 1" in result
    written, = write_zot.batches[0]
    assert written["data"]["tags"] == [{"tag": "x", "type": 1}, {"tag": "new"}]


def test_batch_extra_edit_keeps_lines_only_the_write_side_has(monkeypatch):
    write_zot = _hybrid(monkeypatch, {"extra": "PMID: 123"})

    write.batch_update_extra(
        item_keys=["K0000000"], set_keys={"tex.x": "1"}, ctx=DummyContext())

    written, = write_zot.batches[0]
    assert written["data"]["extra"].splitlines() == ["PMID: 123", "tex.x: 1"]


# --- zotero_set_item_collections: one read and one write per 50 items ---


def _collections_zot(n, monkeypatch, **kwargs):
    from test_batch_write_requests import CountingZot, _install, _items

    items = _items(n)
    for it in items:
        it["data"]["collections"] = ["OLDCOLL1"]
    zot = CountingZot(items, **kwargs)
    zot.collections = lambda **kw: [] if kw.get("start") else [
        {"key": k, "data": {"name": k, "parentCollection": False}}
        for k in ("OLDCOLL1", "NEWCOLL1")]
    _install(monkeypatch, zot)
    return zot


def test_set_item_collections_batches_reads_and_writes(monkeypatch):
    zot = _collections_zot(100, monkeypatch)

    result = write.manage_collections(
        item_keys=[f"K{i:07d}" for i in range(100)], add_to=["NEWCOLL1"],
        remove_from=["OLDCOLL1"], ctx=DummyContext())

    assert zot.requests == 4  # 2 itemKey reads + 2 POSTs; it was 400
    assert all(it["data"]["collections"] == ["NEWCOLL1"] for it in zot._items.values())
    assert result.count("Added ") == 100 and result.count("Removed ") == 100


def test_set_item_collections_reports_every_item_when_one_fails(monkeypatch):
    """A key that could not be read used to abort the call with a bare
    error, hiding that earlier items had already been filed."""
    zot = _collections_zot(2, monkeypatch, refuse={
        "K0000001": {"key": "K0000001", "code": 400, "message": "bad collection"}})
    real_item = zot.item
    zot.item = lambda key: real_item(key) if key in zot._items else (_ for _ in ()).throw(
        RuntimeError("404 Not Found"))

    result = write.manage_collections(
        item_keys=["K0000000", "MISSING1", "K0000001"], add_to=["NEWCOLL1"],
        ctx=DummyContext())

    assert result.splitlines() == [
        "Added K0000000 to NEWCOLL1",
        "Failed: item MISSING1 not found",
        "Failed to add K0000001 to NEWCOLL1: bad collection",
    ]
    assert zot._items["K0000000"]["data"]["collections"] == ["OLDCOLL1", "NEWCOLL1"]


def test_set_item_collections_skips_the_write_when_nothing_changes(monkeypatch):
    zot = _collections_zot(1, monkeypatch)

    result = write.manage_collections(
        item_keys=["K0000000"], add_to=["OLDCOLL1"], remove_from=["NEWCOLL1"],
        ctx=DummyContext())

    assert result.splitlines() == ["K0000000 already in OLDCOLL1", "K0000000 not in NEWCOLL1"]
    assert zot.batches == [] and zot.update_item_calls == []


# --- adders act on what happened to the item, not on how the message begins ---


def _doi_page_setup(monkeypatch):
    """A page declaring a DOI CrossRef knows, in an empty library."""
    from conftest import FakeZotero

    from zotero_mcp.html_metadata import EmbeddedMetadata

    zot = FakeZotero()
    zot.items = lambda **kw: []
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    meta = EmbeddedMetadata()
    meta.title, meta.doi, meta.publication = "A Paper", "10.1234/abc", "J"
    meta.authors = [("A", "B")]
    monkeypatch.setattr(write, "_fetch_embedded_metadata", lambda url, ctx: (meta, ""))
    monkeypatch.setattr(write, "_fetch_one_doi_metadata", lambda doi, ctx: (
        "fetched", {"type": "journal-article", "title": ["A Paper"], "DOI": doi}))
    return zot


def test_url_with_a_doi_and_a_required_pdf_creates_one_item(monkeypatch):
    """add_by_doi answered "Error: item created (key: ...) but attach_mode=
    'required' found no open-access PDF"; add_by_url read the leading
    "Error" as "nothing was created" and added the page a second time."""
    zot = _doi_page_setup(monkeypatch)

    def no_pdf(*args, **kwargs):
        raise _helpers.OaPdfRequiredError("no open-access PDF found")

    monkeypatch.setattr(_helpers, "_try_attach_oa_pdf", no_pdf)

    result = write.add_by_url(url="https://publisher.example.com/a/1",
                              attach_mode="required", ctx=DummyContext())

    assert len(zot.created) == 1
    assert result.startswith("Error: item created (key: KEY0000)")
    assert (result.key, result.status) == ("KEY0000", "failed")


def test_add_results_say_what_became_of_the_item(monkeypatch):
    zot = _doi_page_setup(monkeypatch)
    monkeypatch.setattr(_helpers, "_try_attach_oa_pdf", lambda *a, **k: "skipped")

    created = write.add_by_doi(doi="10.1234/abc", ctx=DummyContext())
    assert (created.key, created.status) == ("KEY0000", "created")
    assert created.startswith("Successfully added: **A Paper**")

    zot.items = lambda **kw: [{"key": "EXIST001", "version": 1, "data": {
        "itemType": "journalArticle", "title": "A Paper", "DOI": "10.1234/ABC"}}]
    reused = write.add_by_doi(doi="10.1234/abc", if_exists="skip", ctx=DummyContext())
    assert (reused.key, reused.status) == ("EXIST001", "reused")

    monkeypatch.setattr(write, "_fetch_one_doi_metadata", lambda doi, ctx: (
        "final", f"DOI not found on CrossRef: {doi}"))
    zot.items = lambda **kw: []
    missing = write.add_by_doi(doi="10.1234/abc", ctx=DummyContext())
    assert getattr(missing, "key", None) is None


# --- add_from_file ---


def _file_setup(monkeypatch, tmp_path, metadata, first_page):
    from test_add_from_file import FakeFitzDocument, FakeZoteroForFile, _patch_fitz

    zot = FakeZoteroForFile()
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    _patch_fitz(monkeypatch, FakeFitzDocument(metadata=metadata, first_page_text=first_page))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    return zot, str(pdf)


def test_add_from_file_does_not_guess_between_two_dois(monkeypatch, tmp_path):
    """A PDF cut from a journal spread names its neighbour's DOI too; the
    first one on the page used to be taken as the file's own."""
    zot, pdf = _file_setup(monkeypatch, tmp_path, {}, (
        "...end of the previous article. https://doi.org/10.1000/previous\n"
        "A New Article\nhttps://doi.org/10.1000/this-one\n"))
    monkeypatch.setattr(write, "add_by_doi", lambda **kw: (_ for _ in ()).throw(
        AssertionError("looked up a guessed DOI")))

    result = write.add_from_file(file_path=pdf, ctx=DummyContext())

    assert zot.created[0]["title"] == "paper.pdf"
    assert "DOI not used (ambiguous: 10.1000/previous, 10.1000/this-one)" in result
    assert "File attached: paper.pdf" in result


def test_add_from_file_does_not_hold_the_api_lock_during_the_doi_lookup(monkeypatch, tmp_path):
    """The lookup asks CrossRef; with the lock held across it every other
    tool waited on a third party."""
    import threading

    from zotero_mcp import client

    zot, pdf = _file_setup(monkeypatch, tmp_path, {"subject": "doi:10.1234/abc"}, "")
    seen = {}

    def add_by_doi(**kwargs):
        def probe():
            seen["free"] = client._zotero_api_lock.acquire(blocking=False)
            if seen["free"]:
                client._zotero_api_lock.release()

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        seen.update(kwargs)
        return write._AddResult("Already in library: **T**", "EXIST001", "reused")

    monkeypatch.setattr(write, "add_by_doi", add_by_doi)

    result = write.add_from_file(file_path=pdf, ctx=DummyContext())

    assert seen["free"] is True
    assert seen["doi"] == "10.1234/abc" and seen["attach_mode"] == "none"
    # The outcome is read off the result, not parsed back out of its text.
    assert zot.attachments[0]["parentid"] == "EXIST001"
    assert result.startswith("Item key: `EXIST001`\nDOI: 10.1234/abc\nFile attached: paper.pdf")


# --- an ISBN or a URL alone does not identify a work ---


def _library(*items):
    from conftest import FakeZotero

    zot = FakeZotero()
    zot._items = [{"key": f"ITEM000{i}", "version": 1, "data": d} for i, d in enumerate(items)]
    zot.searches = []
    stored = zot._items

    def items(**kwargs):
        if "q" in kwargs:
            zot.searches.append((kwargs["q"], kwargs.get("qmode")))
        return stored

    zot.items = items
    return zot


ISBN13 = "9780199735815"


def test_isbn_match_needs_a_book_or_the_same_title():
    """Every chapter carries the book's ISBN: the second chapter was reported
    as already in the library, and a library copy skipped it."""
    chapter = {"itemType": "bookSection", "title": "Chapter One", "ISBN": "978-0-19-973581-5"}
    zot = _library(chapter)
    find = _helpers.find_existing_items

    assert find(zot, isbn=ISBN13, title="Chapter Two", item_type="bookSection") == []
    assert find(zot, isbn=ISBN13, item_type="book") == []  # the book is not there yet
    assert len(find(zot, isbn=ISBN13, title="chapter one!", item_type="bookSection")) == 1

    book = _library({"itemType": "book", "title": "The Book", "ISBN": ISBN13})
    assert len(find(book, isbn=ISBN13)) == 1
    assert len(find(book, isbn=ISBN13, title="Another Title", item_type="book")) == 1
    assert find(book, isbn=ISBN13, title="Chapter Two", item_type="bookSection") == []


def test_url_match_needs_the_same_title_unless_the_stored_item_is_a_bare_page():
    url = "https://press.example.org/book"
    find = _helpers.find_existing_items
    chapter = _library({"itemType": "bookSection", "title": "Chapter One", "url": url})

    assert find(chapter, url=url, title="Chapter Two") == []
    assert len(find(chapter, url=url, title="Chapter One")) == 1
    assert len(find(chapter, url=url)) == 1  # nothing to compare: the URL decides, as before

    bare = _library({"itemType": "webpage", "title": url, "url": url})
    assert len(find(bare, url=url, title="The Page's Real Title")) == 1


def test_copying_a_second_chapter_is_not_skipped_as_existing(monkeypatch):
    target = _library({"itemType": "bookSection", "title": "Chapter One", "ISBN": ISBN13,
                       "url": "https://press.example.org/book"})
    source = {"key": "SRC00002", "data": {
        "itemType": "bookSection", "title": "Chapter Two", "ISBN": ISBN13,
        "url": "https://press.example.org/book", "creators": [], "tags": []}}

    result = write._copy_one(
        "SRC00002", source, [], target, write._CopyLibrary("1", "group", 1), (None, None),
        copy_notes=True, copy_attachments=True, copy_tags=True, if_exists="skip",
        ctx=DummyContext())

    assert result["status"] == "copied" and result["error"] is None
    assert target.created[0]["title"] == "Chapter Two"


def test_the_recheck_before_a_create_does_not_repeat_the_identifier_search(monkeypatch):
    """A new DOI cost three searches: the identifier query, then, under the
    lock, the same identifier query again and the title query."""
    zot = _library()
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    monkeypatch.setattr(write, "_fetch_one_doi_metadata", lambda doi, ctx: (
        "fetched", {"type": "journal-article", "title": ["A Paper"], "DOI": doi}))
    monkeypatch.setattr(_helpers, "_try_attach_oa_pdf", lambda *a, **k: "skipped")

    result = write.add_by_doi(doi="10.1234/abc", page_check="thin_only", ctx=DummyContext())

    assert result.status == "created"
    assert zot.searches == [("10.1234/abc", "everything"), ("A Paper", "titleCreatorYear")]


def test_a_recheck_without_a_title_still_asks_by_identifier():
    zot = _library()
    _helpers.find_existing_items(zot, doi="10.1234/abc", recheck=True)
    assert zot.searches == [("10.1234/abc", "everything")]


def test_add_by_isbn_holds_the_api_lock_for_its_zotero_calls(monkeypatch):
    import threading

    from zotero_mcp import client

    zot = _library()
    held = []

    def probe(result):
        def run():
            free = client._zotero_api_lock.acquire(blocking=False)
            if free:
                client._zotero_api_lock.release()
            held.append(not free)

        t = threading.Thread(target=run)
        t.start()
        t.join()
        return result

    real_items, real_create = zot.items, zot.create_items
    zot.items = lambda **kw: probe(real_items(**kw))
    zot.create_items = lambda items, **kw: probe(real_create(items, **kw))
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    monkeypatch.setattr(write, "_lookup_isbn_openlibrary", lambda isbn, ctx: (
        probe(None) or {"source": "Open Library", "title": "The Book", "creators": []}))

    result = write.add_by_isbn(isbn=ISBN13, tags=["ML", "ml"], ctx=DummyContext())

    assert result.startswith("Successfully added: **The Book**")
    # search, lookup (third party: lock released), search, create
    assert held == [True, False, True, True]
    # Tags are merged the way every other adder merges them.
    assert zot.created[0]["tags"] == [{"tag": "ML"}]
