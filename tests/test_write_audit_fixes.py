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


# --- merge_duplicates fills the keeper's empty fields, as its description says ---


def _merge_zot(monkeypatch, keeper, dup, children=None):
    from test_duplicates import FakeZoteroForDuplicates, _make_item

    zot = FakeZoteroForDuplicates()
    k = _make_item("KEEP", "Same Title", abstract="", version=1)
    d = _make_item("DUP1", "Same Title", abstract="The only abstract.", doi="10.1234/x", version=2)
    k["data"].update(keeper)
    d["data"].update(dup)
    zot._items = [k, d] + [c for kids in (children or {}).values() for c in kids]
    zot._children = {"KEEP": [], "DUP1": [], **(children or {})}
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    return zot


def test_merge_fills_the_keepers_empty_fields_and_keeps_its_own_values(monkeypatch):
    """The description promised it and nothing did it: the duplicate's
    abstract and DOI went to the Trash with it."""
    zot = _merge_zot(
        monkeypatch,
        keeper={"date": "2020", "publicationTitle": "", "creators": [], "citationKey": ""},
        dup={"date": "1999", "publicationTitle": "Nature", "citationKey": "dup2024",
             "numPages": "300", "extra": "  "},
    )

    preview = write.merge_duplicates(keeper_key="KEEP", duplicate_keys=["DUP1"],
                                     ctx=DummyContext())
    assert ("**Empty fields to fill:** ['DOI', 'abstractNote', 'creators', "
            "'publicationTitle']") in preview
    assert zot.update_calls == []  # a dry run writes nothing

    result = write.merge_duplicates(keeper_key="KEEP", duplicate_keys=["DUP1"],
                                    confirm=True, ctx=DummyContext())

    keeper = zot.item("KEEP")["data"]
    assert keeper["abstractNote"] == "The only abstract."
    assert keeper["DOI"] == "10.1234/x"
    assert keeper["publicationTitle"] == "Nature"
    assert keeper["creators"] == [
        {"firstName": "A", "lastName": "Author", "creatorType": "author"}]
    assert keeper["date"] == "2020"           # its own value wins
    assert keeper["citationKey"] == ""        # Better BibTeX's, never copied
    assert "numPages" not in keeper           # not a journalArticle field
    assert len(zot.update_calls) == 1         # with the tags, in one write
    assert "- Empty fields filled: DOI, abstractNote, creators, publicationTitle" in result


def test_merge_carries_a_field_across_item_types_through_its_base_field(monkeypatch):
    keeper = {"itemType": "conferencePaper", "title": "T", "proceedingsTitle": "", "creators": []}
    dup = {"itemType": "journalArticle", "title": "T", "publicationTitle": "Proc. of X",
           "journalAbbreviation": "PX", "creators": [{"creatorType": "author", "name": "A"}]}

    assert write._gap_fill(keeper, [dup]) == {"proceedingsTitle": "Proc. of X"}


def test_merge_with_nothing_to_fill_says_none(monkeypatch):
    _merge_zot(monkeypatch, keeper={"abstractNote": "Mine.", "DOI": "10.1234/x"}, dup={})

    result = write.merge_duplicates(keeper_key="KEEP", duplicate_keys=["DUP1"],
                                    confirm=True, ctx=DummyContext())

    assert "- Empty fields filled: none" in result


def test_merge_does_not_trash_when_the_keeper_write_fails(monkeypatch):
    """A collection that could not be added only logged a warning, and the
    summary still counted it."""
    zot = _merge_zot(monkeypatch, keeper={}, dup={"collections": ["COLL0001"]})

    def refuse(item, **kwargs):
        raise RuntimeError("409 collection not found")

    zot.update_item = refuse

    result = write.merge_duplicates(keeper_key="KEEP", duplicate_keys=["DUP1"],
                                    confirm=True, ctx=DummyContext())

    assert "Nothing was moved or trashed" in result
    assert zot.client.patch_calls == []


def test_auto_merge_reads_each_childrens_listing_once_and_trashes_without_a_re_read(monkeypatch):
    note = {"key": "NOTE0001", "version": 5,
            "data": {"itemType": "note", "parentItem": "DUP1", "note": "n"}}
    zot = _merge_zot(monkeypatch, keeper={"DOI": "10.1234/x", "abstractNote": "Mine."},
                     dup={"abstractNote": ""}, children={"DUP1": [note]})
    calls = {"children": [], "item": []}
    real_children, real_item = zot.children, zot.item
    zot.children = lambda key, **kw: (calls["children"].append(key), real_children(key, **kw))[1]
    zot.item = lambda key: (calls["item"].append(key), real_item(key))[1]

    plan = write.merge_duplicates(auto=True, ctx=DummyContext())
    token = plan.split("plan_token='")[1].split("'")[0]
    calls["children"].clear()
    calls["item"].clear()
    result = write.merge_duplicates(auto=True, confirm=True, plan_token=token,
                                    ctx=DummyContext())

    assert "Groups merged: **1** of 1 planned" in result
    # DUP1 has the note, so it is kept and KEEP is trashed.
    assert sorted(calls["children"]) == ["DUP1", "KEEP"]   # was each twice
    assert calls["item"] == ["DUP1", "KEEP"]               # no re-read before the trash
    assert len(zot.client.patch_calls) == 1


# --- citation import ---


def _template(item_type):
    from conftest import FakeZotero

    return FakeZotero().item_template(item_type)


def test_bibtex_and_others_is_not_an_author():
    from zotero_mcp import citation_import as ci

    entry, = ci.parse_bibtex("@article{k, title={T}, author={Smith, John and others}, year=2020}")
    item = ci.bibtex_entry_to_zotero(entry, _template)

    assert item["creators"] == [
        {"creatorType": "author", "firstName": "John", "lastName": "Smith"}]


CROSSREF_CSL = {
    "type": "journal-article", "title": "T", "DOI": "10.1234/x",
    "id": "https://doi.org/10.1234/x",
    # Crossref's bookkeeping, as its CSL transform returns it.
    "indexed": {"date-parts": [[2024, 1, 1]], "timestamp": 1},
    "reference-count": 55, "is-referenced-by-count": 3, "member": "311",
    "prefix": "10.1234", "score": 1,
    "license": [{"URL": "http://example.org/l", "content-version": "vor"}],
    "content-domain": {"domain": [], "crossmark-restriction": False},
    "reference": [{"key": f"ref{i}", "unstructured": "A cited work"} for i in range(60)],
    # Kept: a CSL variable the type has no field for, and a plain custom value.
    "original-date": {"date-parts": [[1999]]},
    "custom-field": "custom-value",
}


def test_csl_import_keeps_a_providers_bookkeeping_out_of_extra():
    """Every unmapped key was dumped into Extra as JSON: with Crossref's CSL
    that is the whole reference list, 7 KB for a 60-reference article."""
    from zotero_mcp import citation_import as ci

    item = ci.csl_json_to_zotero(CROSSREF_CSL, _template)

    assert item["extra"].splitlines() == [
        'original-date: {"date-parts": [[1999]]}',
        "custom-field: custom-value",
    ]


def test_csl_id_becomes_a_citation_key_only_when_it_looks_like_one():
    from zotero_mcp import citation_import as ci

    def extra(csl_id):
        return ci.csl_json_to_zotero({"type": "book", "title": "T", "id": csl_id}, _template)["extra"]

    assert extra("smith2020deep") == "Citation Key: smith2020deep"
    assert extra("Müller:2020-a") == "Citation Key: Müller:2020-a"
    for record_id in ("https://doi.org/10.1234/x", "http://zotero.org/users/1/items/ABCD2345",
                      "10.1234/x", "ITEM-1", "12345", 7, "two words"):
        assert extra(record_id) == "", record_id


TWO_ENTRIES_ONE_DOI = """
@article{a, title={Same Paper}, doi={10.1234/same}, year={2020}}
@article{b, title={Same Paper}, doi={https://doi.org/10.1234/SAME}, year={2020}}
@article{c, title={Another Paper}, doi={10.1234/other}, year={2021}}
"""


def _import_zot(monkeypatch):
    zot = _library()
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    monkeypatch.setattr(_helpers, "_try_attach_oa_pdf", lambda *a, **k: "skipped")
    return zot


def test_a_doi_listed_twice_in_one_import_is_created_once(monkeypatch):
    """The library lookup runs before anything is created, so the second
    entry could not see the first and both were created."""
    zot = _import_zot(monkeypatch)

    result = write.add_by_bibtex(bibtex=TWO_ENTRIES_ONE_DOI, ctx=DummyContext())

    assert [it["title"] for it in zot.created] == ["Same Paper", "Another Paper"]
    assert "Added 2/3 items. 1 repeated an earlier entry's DOI (added once)." in result
    assert "2. `KEY0000` — Same Paper (DOI: 10.1234/same) [same DOI as entry 1, added once]" in result
    # One lookup pair per distinct DOI: the repeat costs no search.
    assert len(zot.searches) == 4


def test_if_exists_duplicate_still_creates_a_repeated_doi_twice(monkeypatch):
    zot = _import_zot(monkeypatch)

    result = write.add_by_bibtex(bibtex=TWO_ENTRIES_ONE_DOI, if_exists="duplicate",
                                 ctx=DummyContext())

    assert len(zot.created) == 3
    assert "Added 3/3 items." in result


# --- the smaller items ---


def _lock_probe():
    """probe(value) returns value and records whether the API lock was held."""
    import threading

    from zotero_mcp import client

    held = []

    def probe(value=None):
        def run():
            free = client._zotero_api_lock.acquire(blocking=False)
            if free:
                client._zotero_api_lock.release()
            held.append(not free)

        t = threading.Thread(target=run)
        t.start()
        t.join()
        return value

    return probe, held


def test_relation_and_delete_tools_hold_the_api_lock(monkeypatch):
    """They made their Zotero calls beside other tools' calls to the
    single-threaded local API."""
    from test_item_related import FakeZoteroForRelations, _make_item

    zot = FakeZoteroForRelations([_make_item("AAAA1111"), _make_item("BBBB2222")])
    zot.client = type("C", (), {"patch": staticmethod(
        lambda **kw: type("R", (), {"status_code": 204})())})()
    zot.collection = lambda key: {"key": key, "version": 1, "data": {"name": "C"}}
    zot.delete_collection = lambda coll: type("R", (), {"status_code": 204})()
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    probe, held = _lock_probe()
    real_item = zot.item
    zot.item = lambda key: probe(real_item(key))
    ctx = DummyContext()

    assert "Successfully added" in write.add_item_relation("AAAA1111", "BBBB2222", ctx=ctx)
    assert "Successfully removed" in write.remove_item_relation("AAAA1111", "BBBB2222", ctx=ctx)
    assert "Successfully trashed" in write.delete_item("AAAA1111", ctx=ctx)
    real_collection = zot.collection
    zot.collection = lambda key: probe(real_collection(key))
    assert "Deleted collection" in write.delete_collection("COLL0001", ctx=ctx)

    assert held and all(held)


def test_adding_a_relation_reads_each_item_once(monkeypatch):
    """Both items were read a second time before the reverse write, and the
    first of those reads was never used: six requests for two writes."""
    from test_item_related import FakeZoteroForRelations, _make_item

    zot = FakeZoteroForRelations([_make_item("AAAA1111"), _make_item("BBBB2222")])
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    reads = []
    real_item = zot.item
    zot.item = lambda key: (reads.append(key), real_item(key))[1]

    write.add_item_relation("AAAA1111", "BBBB2222", ctx=DummyContext())

    assert reads == ["AAAA1111", "BBBB2222"]
    assert [it["key"] for it in zot.update_calls] == ["AAAA1111", "BBBB2222"]
    assert zot._items["BBBB2222"]["data"]["relations"] == {
        "dc:relation": ["http://zotero.org/users/12345/items/AAAA1111"]}


def test_reverse_relation_is_re_read_only_on_a_version_conflict(monkeypatch):
    from pyzotero.errors import PreConditionFailedError
    from test_item_related import FakeZoteroForRelations, _make_item

    zot = FakeZoteroForRelations([_make_item("AAAA1111"), _make_item("BBBB2222")])
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    reads, refused = [], []
    real_item, real_update = zot.item, zot.update_item
    zot.item = lambda key: (reads.append(key), real_item(key))[1]

    def update(item, **kw):
        if item["key"] == "BBBB2222" and not refused:
            refused.append(True)
            raise PreConditionFailedError("412")
        return real_update(item, **kw)

    zot.update_item = update

    write.add_item_relation("AAAA1111", "BBBB2222", ctx=DummyContext())

    assert reads == ["AAAA1111", "BBBB2222", "BBBB2222"]
    assert "dc:relation" in zot._items["BBBB2222"]["data"]["relations"]


def test_crossref_dates_are_zero_padded():
    cr = {"type": "journal-article", "title": ["T"], "published": {"date-parts": [[2020, 1, 5]]}}
    item, _type, _note = write._crossref_to_item_data(cr, "10.1234/x", _template)
    assert item["date"] == "2020-01-05"

    cr["published"] = {"date-parts": [[2020]]}
    assert write._crossref_to_item_data(cr, "10.1234/x", _template)[0]["date"] == "2020"


def test_batch_extra_errors_are_explained_like_the_other_tools(monkeypatch):
    def not_running(ctx):
        raise ConnectionError("connection refused 127.0.0.1:23119")

    monkeypatch.setattr(_helpers, "_get_write_client", not_running)
    monkeypatch.setattr(_helpers, "format_zotero_error", lambda e, zot=None: "EXPLAINED")

    result = write.batch_update_extra(item_keys=["K0000000"], set_keys={"a": "b"},
                                      ctx=DummyContext())

    assert result == "Error in batch extra update: EXPLAINED"


def test_a_url_batch_counts_outcomes_from_each_results_status(monkeypatch):
    """The header was counted by searching each rendered result for known
    opening phrases; it is read off the result now."""
    from zotero_mcp.html_metadata import EmbeddedMetadata

    zot = _library({"itemType": "webpage", "title": "https://old.example.org/p",
                    "url": "https://old.example.org/p"})
    monkeypatch.setattr(_helpers, "_get_write_client", lambda ctx: (zot, zot))
    article = EmbeddedMetadata()
    article.title, article.publication, article.authors = "An Article", "J", [("A", "B")]

    def page(url, ctx):
        if "journal" in url:
            return article, ""
        if "down" in url:
            raise RuntimeError("boom")
        return None, "the page did not respond in time"

    monkeypatch.setattr(write, "_fetch_embedded_metadata", page)

    result = write.add_by_url(url=[
        "https://journal.example.org/a1",   # created from its meta tags
        "https://blog.example.org/post",    # created as a bare webpage
        "https://old.example.org/p",        # already in the library
        "https://blog.example.org/post",    # repeats entry 2
        "https://down.example.org/x",       # fails
    ], ctx=DummyContext())

    assert result.splitlines()[:2] == [
        "# Added 2 of 5 URLs",
        "1 already in library, 1 repeated in this request, 1 failed.",
    ]
    assert len(zot.created) == 2


def test_set_item_collections_re_applies_the_change_after_a_version_conflict(monkeypatch):
    """A 412 means someone else changed the item: the adds and removes are
    applied to what it holds now, not replayed from the stale read."""
    zot = _collections_zot(1, monkeypatch, refuse={
        "K0000000": {"key": "K0000000", "code": 412, "message": "stale"}})
    real_items = zot.items

    def items_then_someone_files_it(*args, **kwargs):
        rows = real_items(*args, **kwargs)
        zot._items["K0000000"]["data"]["collections"] = ["OLDCOLL1", "ELSEWHERE"]
        return rows

    zot.items = items_then_someone_files_it

    result = write.manage_collections(
        item_keys=["K0000000"], add_to=["NEWCOLL1"], remove_from=["OLDCOLL1"],
        ctx=DummyContext())

    assert result.splitlines() == ["Added K0000000 to NEWCOLL1", "Removed K0000000 from OLDCOLL1"]
    assert zot.update_item_calls[-1]["data"]["collections"] == ["ELSEWHERE", "NEWCOLL1"]
