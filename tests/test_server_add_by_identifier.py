from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
        return None


class FakeResponse:
    def __init__(
        self,
        payload=None,
        ok=True,
        status_code=200,
        headers=None,
        text="",
        url="https://example.org",
        bytes_data=b"%PDF-1.4 test",
    ):
        self._payload = payload or {}
        self.ok = ok
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.url = url
        self._bytes_data = bytes_data

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise server.requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._bytes_data), chunk_size):
            yield self._bytes_data[i : i + chunk_size]


class FakeZotero:
    def __init__(self):
        self.created = []
        self.imported = []

    def collections(self):
        return [
            {"key": "COLL001", "data": {"name": "PhD Research"}},
            {"key": "COLL002", "data": {"name": "Reading List"}},
        ]

    def collection(self, key):
        if key == "COLL001":
            return {"key": key, "data": {"name": "PhD Research"}}
        return {"key": key, "data": {"name": "Unknown"}}

    def addto_collection(self, _collection_key, _item_keys):
        return True

    def create_items(self, items):
        self.created.extend(items)
        if len(items) == 1 and items[0].get("itemType") == "attachment":
            return {"success": {"0": "ATTACH001"}}
        return {"success": {"0": "ITEM0001"}}

    def attachment_simple(self, files, parentid=None, parent_item=None):
        parent = parentid or parent_item
        file_path = files[0] if isinstance(files, list) else files
        self.imported.append({"file": file_path, "parentItem": parent})
        self.created.append(
            {
                "itemType": "attachment",
                "parentItem": parent,
                "linkMode": "imported_file",
                "title": "PDF",
            }
        )
        return {"successful": {"0": "ATTACHIMP1"}}


class FakeZoteroNoImport(FakeZotero):
    def attachment_simple(self, *_args, **_kwargs):
        raise RuntimeError("upload unavailable")


def _crossref_payload(doi="10.1000/test-doi"):
    return {
        "message": {
            "type": "journal-article",
            "title": ["A DOI Based Paper"],
            "author": [{"given": "Jane", "family": "Doe"}],
            "container-title": ["Journal of Tests"],
            "issued": {"date-parts": [[2024, 5, 1]]},
            "URL": f"https://doi.org/{doi}",
            "link": [{"content-type": "application/pdf", "URL": "https://example.org/paper.pdf"}],
        }
    }

def _crossref_payload_no_pdf_link(doi="10.1000/no-direct-pdf"):
    return {
        "message": {
            "type": "journal-article",
            "title": ["A DOI Based Paper"],
            "author": [{"given": "Jane", "family": "Doe"}],
            "container-title": ["Journal of Tests"],
            "issued": {"date-parts": [[2024, 5, 1]]},
            "URL": f"https://doi.org/{doi}",
        }
    }


def test_add_by_identifier_supports_raw_doi(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload("10.1000/rawdoi")),
    )
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(headers={"Content-Type": "application/pdf"}),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/rawdoi",
        attach_pdfs=True,
        attach_mode="linked_url",
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert "created item key `ITEM0001`" in result
    assert fake_zot.created[0]["DOI"] == "10.1000/rawdoi"
    assert "pdf: linked_url" in result


def test_add_by_identifier_supports_doi_url(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload("10.1000/url-doi")),
    )

    result = server.add_by_identifier(
        identifiers="https://doi.org/10.1000/url-doi",
        attach_pdfs=False,
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert fake_zot.created[0]["DOI"] == "10.1000/url-doi"
    assert "pdf: not requested" in result


def test_add_by_identifier_rejects_non_doi_identifier(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    result = server.add_by_identifier(
        identifiers="PMID:12345678",
        ctx=DummyContext(),
    )

    assert "No items were created." in result
    assert "Supported formats: DOI" in result


def test_add_by_identifier_reports_crossref_network_error(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    def _boom(*_args, **_kwargs):
        raise server.requests.RequestException("Crossref unavailable")

    monkeypatch.setattr(server.requests, "get", _boom)

    result = server.add_by_identifier(
        identifiers="10.1000/network-failure",
        ctx=DummyContext(),
    )

    assert "Network error fetching metadata" in result
    assert "Crossref unavailable" in result


def test_add_by_identifier_resolves_collection_names(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload("10.1000/collection")),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/collection",
        collection_names=["PhD Research"],
        attach_pdfs=False,
        ctx=DummyContext(),
    )

    assert "collections: PhD Research" in result


def test_add_by_identifier_skips_pdf_attachment_if_not_pdf(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload_no_pdf_link("10.1000/no-pdf")),
    )
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(headers={"Content-Type": "text/html"}),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/no-pdf",
        attach_pdfs=True,
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert len(fake_zot.created) == 1
    assert "pdf: failed" in result


def test_add_by_identifier_discovers_pdf_from_landing_page(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    def _mock_get(url, *_args, **_kwargs):
        if "api.crossref.org/works/" in url:
            return FakeResponse(payload=_crossref_payload_no_pdf_link("10.1000/discover-pdf"))
        return FakeResponse(
            text='<html><head><meta name="citation_pdf_url" content="/downloads/paper.pdf"></head></html>',
            url="https://publisher.org/article",
        )

    monkeypatch.setattr(server.requests, "get", _mock_get)
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(headers={"Content-Type": "application/pdf"}),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/discover-pdf",
        attach_pdfs=True,
        attach_mode="linked_url",
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    # metadata item + discovered PDF attachment
    assert len(fake_zot.created) == 2
    assert fake_zot.created[1]["itemType"] == "attachment"
    assert "pdf: linked_url" in result


def test_add_by_identifier_supports_arxiv_id(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>https://arxiv.org/abs/2401.00001</id>
        <published>2024-01-01T00:00:00Z</published>
        <title>ArXiv Import Test</title>
        <summary>Test abstract from arXiv.</summary>
        <author><name>Alice Researcher</name></author>
        <author><name>Bob Scientist</name></author>
        <arxiv:doi>10.1000/arxivdoi</arxiv:doi>
      </entry>
    </feed>"""

    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(text=arxiv_xml),
    )
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(headers={"Content-Type": "application/pdf"}),
    )

    result = server.add_by_identifier(
        identifiers="arXiv:2401.00001",
        attach_pdfs=True,
        attach_mode="linked_url",
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert fake_zot.created[0]["itemType"] == "preprint"
    assert fake_zot.created[0]["title"] == "ArXiv Import Test"
    assert fake_zot.created[0]["DOI"] == "10.1000/arxivdoi"
    assert "publicationTitle" not in fake_zot.created[0]
    assert len(fake_zot.created) == 2
    assert fake_zot.created[1]["itemType"] == "attachment"
    assert "pdf: linked_url" in result


def test_add_by_identifier_supports_arxiv_url(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>https://arxiv.org/abs/2301.12345</id>
        <published>2023-01-10T00:00:00Z</published>
        <title>URL Import Test</title>
        <summary>Another abstract.</summary>
        <author><name>Carol Author</name></author>
      </entry>
    </feed>"""

    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(text=arxiv_xml),
    )

    result = server.add_by_identifier(
        identifiers="https://arxiv.org/abs/2301.12345",
        attach_pdfs=False,
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert fake_zot.created[0]["itemType"] == "preprint"
    assert fake_zot.created[0]["title"] == "URL Import Test"


def test_add_by_identifier_supports_arxiv_namespace_doi(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>https://arxiv.org/abs/2603.02553</id>
        <published>2026-03-04T00:00:00Z</published>
        <title>Namespace DOI Import Test</title>
        <summary>Imported via arXiv DOI namespace.</summary>
        <author><name>Dana Example</name></author>
      </entry>
    </feed>"""

    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(text=arxiv_xml),
    )
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(headers={"Content-Type": "application/pdf"}),
    )

    result = server.add_by_identifier(
        identifiers="10.48550/arXiv.2603.02553",
        attach_pdfs=True,
        attach_mode="linked_url",
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert fake_zot.created[0]["itemType"] == "preprint"
    assert fake_zot.created[0]["DOI"] == "10.48550/arXiv.2603.02553"
    assert "pdf: linked_url" in result


def test_add_by_identifier_import_file_mode_success(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)

    def _mock_get(url, *_args, **_kwargs):
        if "api.crossref.org/works/" in url:
            return FakeResponse(payload=_crossref_payload("10.1000/import-ok"))
        return FakeResponse(headers={"Content-Type": "application/pdf"})

    monkeypatch.setattr(server.requests, "get", _mock_get)

    result = server.add_by_identifier(
        identifiers="10.1000/import-ok",
        attach_pdfs=True,
        attach_mode="import_file",
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert "pdf: imported" in result


def test_add_by_identifier_linked_url_mode(monkeypatch):
    fake_zot = FakeZotero()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload("10.1000/linked")),
    )
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(headers={"Content-Type": "application/pdf"}),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/linked",
        attach_pdfs=True,
        attach_mode="linked_url",
        ctx=DummyContext(),
    )

    assert "pdf: linked_url" in result
    assert fake_zot.created[1]["linkMode"] == "linked_url"


def test_add_by_identifier_import_file_mode_failure_warns(monkeypatch):
    fake_zot = FakeZoteroNoImport()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload("10.1000/import-fail")),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/import-fail",
        attach_pdfs=True,
        attach_mode="import_file",
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert "pdf: failed" in result
    # only metadata item should be created
    assert len(fake_zot.created) == 1


def test_add_by_identifier_auto_fallback_to_linked_url(monkeypatch):
    fake_zot = FakeZoteroNoImport()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload("10.1000/auto-fallback")),
    )
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(headers={"Content-Type": "application/pdf"}),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/auto-fallback",
        attach_pdfs=True,
        attach_mode="auto",
        ctx=DummyContext(),
    )

    assert "pdf: linked_url" in result
    assert len(fake_zot.created) == 2
    assert fake_zot.created[1]["linkMode"] == "linked_url"


def test_add_by_identifier_auto_both_paths_fail(monkeypatch):
    fake_zot = FakeZoteroNoImport()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        server.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(payload=_crossref_payload("10.1000/auto-both-fail")),
    )
    monkeypatch.setattr(
        server.requests,
        "head",
        lambda *_args, **_kwargs: FakeResponse(ok=False, status_code=404, headers={"Content-Type": "text/html"}),
    )

    result = server.add_by_identifier(
        identifiers="10.1000/auto-both-fail",
        attach_pdfs=True,
        attach_mode="auto",
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert "pdf: failed" in result
