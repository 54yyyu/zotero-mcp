"""Add by PubMed ID (#692): PubMed XML -> CSL JSON -> Zotero.

The network is never touched: ``fixtures/pubmed_efetch.xml`` is a recorded
E-utilities efetch response (see ``fixtures/README.md``), served to
``add_by_pmid`` through a stubbed ``requests.get``.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from conftest import FakeZotero

from zotero_mcp.citation_import import (
    _expand_pages,
    _pubmed_date,
    _title_case,
    csl_json_to_zotero,
    pubmed_xml_to_csl,
)
from zotero_mcp.schema import valid_fields
from zotero_mcp.tools import write

FIXTURE = Path(__file__).parent / "fixtures" / "pubmed_efetch.xml"


def _records() -> dict[str, dict]:
    root = ET.parse(FIXTURE).getroot()
    out = {}
    for a in root.findall("PubmedArticle"):
        csl = pubmed_xml_to_csl(a)
        out[csl["PMID"]] = csl
    return out


def schema_template(item_type: str) -> dict:
    """Every field the bundled Zotero schema gives the type, PMID included."""
    return {
        "itemType": item_type,
        **{f: "" for f in valid_fields(item_type)},
        "creators": [],
        "tags": [],
        "collections": [],
        "relations": {},
    }


# ---------------------------------------------------------------------------
# Converter, on the recorded records
# ---------------------------------------------------------------------------

class TestPubmedXmlToCsl:
    def test_fixture_has_the_three_records(self):
        assert set(_records()) == {"31842878", "16446872", "10375727"}

    def test_identifiers_and_journal(self):
        csl = _records()["31842878"]
        assert csl["PMCID"] == "PMC6912996"
        assert csl["DOI"] == "10.1186/s12916-019-1466-7"
        assert csl["container-title"] == "BMC Medicine"
        assert csl["container-title-short"] == "BMC Med"
        assert csl["ISSN"] == "1741-7015"
        assert csl["source"] == "PubMed"
        assert csl["issued"] == {"date-parts": [[2019, 12, 16]]}
        assert csl["Epub Date"] == "2019-12-16"

    def test_title_and_short_title(self):
        csl = _records()["31842878"]
        assert csl["title"] == "Calibration: the Achilles heel of predictive analytics"
        assert csl["title-short"] == "Calibration"

    def test_translated_title_loses_brackets_and_period(self):
        csl = _records()["16446872"]
        assert csl["title"] == "Delirium in Chilean elderly inpatients: an overlooked problem"
        assert csl["title-short"] == "Delirium in Chilean elderly inpatients"
        assert csl["language"] == "spa"

    def test_structured_abstract_keeps_labels(self):
        abstract = _records()["31842878"]["abstract"]
        assert abstract.startswith("Background: ")
        assert "\nMain text: " in abstract  # "MAIN TEXT" in the XML
        assert "\nConclusion: " in abstract

    def test_mesh_and_author_keywords_become_keywords(self):
        kw = _records()["31842878"]["keyword"]
        assert "Humans" in kw  # MeSH descriptor
        assert "Calibration" in kw
        assert len(kw) == len(set(kw))

    def test_medline_pages_expanded(self):
        assert _records()["16446872"]["page"] == "1449-1454"
        assert _records()["10375727"]["page"] == "207-210"

    def test_retraction_notice_and_publication_type(self):
        csl = _records()["10375727"]
        assert "Retracted Publication" in csl["Publication Type"]
        assert csl["Retraction In"].endswith("(PMID 11439911)")
        assert "PMCID" not in csl


class TestHelpers:
    @pytest.mark.parametrize("raw, expanded", [
        ("2409-19", "2409-2419"),
        ("1183-92", "1183-1192"),
        ("207-210", "207-210"),
        ("230", "230"),
        ("109S-115S", "109S-115S"),
    ])
    def test_expand_pages(self, raw, expanded):
        assert _expand_pages(raw) == expanded

    @pytest.mark.parametrize("xml, parts", [
        ("<PubDate><Year>2005</Year><Month>Dec</Month></PubDate>", [2005, 12]),
        ("<PubDate><Year>2009</Year><Season>Nov-Dec</Season></PubDate>", [2009]),
        ("<PubDate><MedlineDate>1998 Dec-1999 Jan</MedlineDate></PubDate>", [1998, 12]),
        ("<PubDate><MedlineDate>2009</MedlineDate></PubDate>", [2009]),
    ])
    def test_pubmed_date(self, xml, parts):
        assert _pubmed_date(ET.fromstring(xml)) == parts

    def test_title_case_keeps_small_words(self):
        assert _title_case("Statistics in medicine") == "Statistics in Medicine"
        assert _title_case("the journal of physiology") == "The Journal of Physiology"


class TestZoteroItem:
    def test_native_fields_on_the_item(self):
        item = csl_json_to_zotero(_records()["31842878"], schema_template)
        assert item["PMID"] == "31842878"
        assert item["PMCID"] == "PMC6912996"
        assert item["journalAbbreviation"] == "BMC Med"
        assert item["libraryCatalog"] == "PubMed"
        assert item["shortTitle"] == "Calibration"
        assert "PMID" not in item["extra"]
        assert "Epub Date: 2019-12-16" in item["extra"]
        assert {"tag": "Humans"} in item["tags"]


# ---------------------------------------------------------------------------
# add_by_pmid / zotero_add_item(source_type="pmid")
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


@pytest.fixture
def efetch(monkeypatch):
    """Serve the recorded efetch response, cut to the requested PMIDs."""
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append({"url": url, **(params or {})})
        root = ET.parse(FIXTURE).getroot()
        wanted = set(params["id"].split(","))
        for a in list(root):
            if a.findtext("MedlineCitation/PMID") not in wanted:
                root.remove(a)
        return _Resp(ET.tostring(root))

    monkeypatch.setattr(write.requests, "get", fake_get)
    return calls


@pytest.fixture
def fake(monkeypatch):
    z = FakeZotero()
    z.item_template = lambda item_type, linkmode=None: schema_template(item_type)
    monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client",
                        lambda ctx: (z, z))
    monkeypatch.setattr("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                        lambda *a, **k: "no open-access PDF found (stubbed)")
    return z


class TestAddByPmid:
    def test_adds_with_native_pmid(self, efetch, fake, dummy_ctx):
        result = write.add_by_pmid("31842878", attach_mode="none", ctx=dummy_ctx)
        assert "Successfully added" in result
        assert len(fake.created) == 1
        item = fake.created[0]
        assert item["PMID"] == "31842878"
        assert item["PMCID"] == "PMC6912996"
        assert efetch[0]["db"] == "pubmed"
        assert efetch[0]["retmode"] == "xml"

    def test_several_pmids_one_request(self, efetch, fake, dummy_ctx):
        write.add_by_pmid("31842878, 16446872", attach_mode="none", ctx=dummy_ctx)
        assert len(efetch) == 1
        assert efetch[0]["id"] == "31842878,16446872"
        assert {i["PMID"] for i in fake.created} == {"31842878", "16446872"}

    def test_via_add_item(self, efetch, fake, dummy_ctx):
        write.add_item(source="10375727", source_type="pmid",
                       attach_mode="none", ctx=dummy_ctx)
        assert fake.created[0]["PMID"] == "10375727"

    def test_rejects_non_numeric_without_a_request(self, efetch, fake, dummy_ctx):
        result = write.add_by_pmid("PMC6912996", ctx=dummy_ctx)
        assert result.startswith("Error")
        assert efetch == []
        assert fake.created == []

    def test_unknown_pmid_reported(self, efetch, fake, dummy_ctx):
        result = write.add_by_pmid("31842878,1", ctx=dummy_ctx)
        assert result == "Error: PubMed returned no journal-article record for PMID 1"
        assert fake.created == []

    def test_network_failure_reported(self, monkeypatch, fake, dummy_ctx):
        def boom(*a, **k):
            raise write.requests.ConnectionError("offline")
        monkeypatch.setattr(write.requests, "get", boom)
        result = write.add_by_pmid("31842878", ctx=dummy_ctx)
        assert result.startswith("Error: PubMed lookup failed")
        assert fake.created == []
