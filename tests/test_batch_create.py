"""Tests for #A4: batching create_items() calls across add_by_doi,
add_by_bibtex, and add_by_csl_json instead of one POST per entry.

Uses a FakeZotero variant that actually honors ``items(itemKey=...)`` and
records call shapes, so the assertions can pin down call *counts* (one
create_items() POST per <=50-item chunk, one bulk items() read per chunk,
zero per-item item() GETs when atomic collection filing worked) rather than
just end-to-end output text, which the other add_by_* test files already
cover.
"""

from unittest.mock import MagicMock

from conftest import DummyContext, FakeZotero, _FakeResponse, fake_crossref_get

from zotero_mcp.tools import write


class _BatchRecordingZotero(FakeZotero):
    """FakeZotero that records create_items chunk sizes and items()/item()
    read-call shapes, and can simulate atomic collection filing on
    create_items() succeeding or failing per item (mirrors #235)."""

    def __init__(self, atomic_filing_works: bool = True):
        super().__init__()
        self._atomic = atomic_filing_works
        self.create_calls: list[int] = []
        self.bulk_items_calls: list[list[str]] = []
        self.single_item_calls: list[str] = []
        self.item_template_calls: list[str] = []
        self._created_items: dict[str, dict] = {}
        self._next_key = 0

    def item_template(self, item_type):
        self.item_template_calls.append(item_type)
        return super().item_template(item_type)

    def create_items(self, items, **kwargs):
        self.created.extend(items)
        self.create_calls.append(len(items))
        result: dict[str, str] = {}
        for i, item in enumerate(items):
            key = f"KEY{self._next_key:04d}"
            self._next_key += 1
            result[str(i)] = key
            stored_collections = (
                list(item.get("collections") or []) if self._atomic else []
            )
            self._created_items[key] = {
                "key": key,
                "version": 1,
                "data": {**item, "key": key, "collections": stored_collections},
            }
        return {"success": result, "successful": {}, "failed": {}}

    def items(self, **kwargs):
        item_key = kwargs.get("itemKey")
        if item_key is None:
            return super().items(**kwargs)
        keys = item_key.split(",")
        self.bulk_items_calls.append(keys)
        return [self._created_items[k] for k in keys if k in self._created_items]

    def item(self, item_key):
        self.single_item_calls.append(item_key)
        if item_key in self._created_items:
            return self._created_items[item_key]
        return super().item(item_key)

    def addto_collection(self, collection_key, item, **kwargs):
        key = item["key"] if isinstance(item, dict) else item
        stored = self._created_items.get(key)
        if stored:
            cols = stored["data"].setdefault("collections", [])
            if collection_key not in cols:
                cols.append(collection_key)
        return _FakeResponse(204)


def _make_crossref_response(doi="10.1234/test"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status": "ok",
        "message": {
            "type": "journal-article",
            "title": [f"Paper {doi}"],
            "DOI": doi,
            "author": [{"given": "A", "family": "Author"}],
        },
    }
    resp.raise_for_status = MagicMock()
    return resp


def _patch_write_client(monkeypatch, zot, *, pdf_status="skipped (test)"):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (zot, zot)
    )
    monkeypatch.setattr(
        "requests.get",
        fake_crossref_get(lambda doi: _make_crossref_response(doi=doi).json()["message"]),
    )
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._try_attach_oa_pdf",
        lambda *a, **kw: pdf_status,
    )


# ---------------------------------------------------------------------------
# add_by_doi
# ---------------------------------------------------------------------------

class TestAddByDoiBatching:
    def test_one_create_items_call_for_a_small_batch(self, monkeypatch):
        z = _BatchRecordingZotero()
        _patch_write_client(monkeypatch, z)

        dois = [f"10.1234/d{i}" for i in range(10)]
        result = write.add_by_doi(doi=",".join(dois), ctx=DummyContext())

        assert z.create_calls == [10]
        assert len(z.created) == 10
        assert "# Added 10 DOIs" in result
        for i in range(10):
            assert f"Successfully added: **Paper 10.1234/d{i}**" in result

    def test_chunks_at_50(self, monkeypatch):
        z = _BatchRecordingZotero()
        _patch_write_client(monkeypatch, z)

        dois = [f"10.1234/d{i}" for i in range(55)]
        result = write.add_by_doi(doi=",".join(dois), ctx=DummyContext())

        assert z.create_calls == [50, 5]
        assert len(z.created) == 55
        assert "# Added 55 DOIs" in result

    def test_item_template_memoized_across_the_batch(self, monkeypatch):
        """#A3: CROSSREF_TYPE_MAP maps every DOI here onto the same
        Zotero item type, so a 10-DOI batch should fetch the template
        once, not once per DOI."""
        z = _BatchRecordingZotero()
        _patch_write_client(monkeypatch, z)

        dois = [f"10.1234/d{i}" for i in range(10)]
        write.add_by_doi(doi=",".join(dois), ctx=DummyContext())

        assert z.item_template_calls == ["journalArticle"]

    def test_collection_membership_one_bulk_read_no_per_item_gets_when_atomic_works(
        self, monkeypatch
    ):
        z = _BatchRecordingZotero(atomic_filing_works=True)
        z._collections = [
            {"key": "ABCDEFGH", "data": {"name": "Target", "parentCollection": False}},
        ]
        _patch_write_client(monkeypatch, z)

        dois = [f"10.1234/d{i}" for i in range(5)]
        result = write.add_by_doi(
            doi=",".join(dois), collections="ABCDEFGH", ctx=DummyContext()
        )

        assert z.create_calls == [5]
        # One bulk read for the whole chunk, no per-item item() GETs — every
        # item's atomic collection filing already "worked" in this fake.
        assert len(z.bulk_items_calls) == 1
        assert sorted(z.bulk_items_calls[0]) == [f"KEY{i:04d}" for i in range(5)]
        assert z.single_item_calls == []
        assert "Filed in ['ABCDEFGH']" in result

    def test_collection_membership_backstops_only_real_misses(self, monkeypatch):
        z = _BatchRecordingZotero(atomic_filing_works=False)
        z._collections = [
            {"key": "ABCDEFGH", "data": {"name": "Target", "parentCollection": False}},
        ]
        _patch_write_client(monkeypatch, z)

        dois = [f"10.1234/d{i}" for i in range(3)]
        result = write.add_by_doi(
            doi=",".join(dois), collections="ABCDEFGH", ctx=DummyContext()
        )

        assert len(z.bulk_items_calls) == 1
        # Atomic filing failed for everyone, so the backstop (which re-fetches
        # via item()) runs for every created item.
        assert sorted(z.single_item_calls) == [f"KEY{i:04d}" for i in range(3)]
        assert "Filed in ['ABCDEFGH']" in result

    def test_partial_create_failure_reports_per_entry(self, monkeypatch):
        z = _BatchRecordingZotero()

        def flaky_create(items, **kwargs):
            z.created.extend(items)
            success, failed = {}, {}
            for i, _item in enumerate(items):
                if i == 1:
                    failed[str(i)] = "simulated write failure"
                else:
                    success[str(i)] = f"KEY{i:04d}"
            return {"success": success, "successful": {}, "failed": failed}

        z.create_items = flaky_create
        _patch_write_client(monkeypatch, z)

        dois = [f"10.1234/d{i}" for i in range(3)]
        result = write.add_by_doi(doi=",".join(dois), ctx=DummyContext())

        assert "Successfully added: **Paper 10.1234/d0**" in result
        assert "Failed to create item:" in result
        assert "simulated write failure" in result
        assert "Successfully added: **Paper 10.1234/d2**" in result


# ---------------------------------------------------------------------------
# add_by_bibtex / add_by_csl_json
# ---------------------------------------------------------------------------

_BIBTEX_ENTRIES = "\n".join(
    f'@article{{e{i}, title={{Title {i}}}, author={{Author, {i}}}, year={{2020}}}}'
    for i in range(6)
)

_CSL_ENTRIES = [
    {"id": f"e{i}", "type": "article-journal", "title": f"Title {i}"} for i in range(6)
]


class TestAddByBibtexBatching:
    def test_one_create_items_call_for_all_entries(self, monkeypatch):
        z = _BatchRecordingZotero()
        _patch_write_client(monkeypatch, z)

        result = write.add_by_bibtex(bibtex=_BIBTEX_ENTRIES, ctx=DummyContext())

        assert z.create_calls == [6]
        assert len(z.created) == 6
        assert "Added 6/6 items" in result


class TestAddByCslJsonBatching:
    def test_one_create_items_call_for_all_entries(self, monkeypatch):
        z = _BatchRecordingZotero()
        _patch_write_client(monkeypatch, z)

        result = write.add_by_csl_json(csl_json=_CSL_ENTRIES, ctx=DummyContext())

        assert z.create_calls == [6]
        assert len(z.created) == 6
        assert "Added 6/6 items" in result
