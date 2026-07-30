from __future__ import annotations

import json
import unittest

from zotero_mcp.scholarly import (
    ArxivProvider,
    CrossrefProvider,
    EuropePMCProvider,
    HttpClient,
    OpenAlexProvider,
    Paper,
    PaperAccumulator,
    PaperSearchService,
    Provider,
    SemanticScholarProvider,
    normalize_doi,
    reconstruct_openalex_abstract,
    source_record,
    strip_markup,
)


class NormalizationTests(unittest.TestCase):
    def test_normalize_doi_accepts_common_forms(self) -> None:
        expected = "10.1000/example"
        self.assertEqual(normalize_doi(" DOI:10.1000/Example. "), expected)
        self.assertEqual(normalize_doi("https://doi.org/10.1000/Example"), expected)

    def test_markup_is_removed_from_crossref_abstract(self) -> None:
        value = "<jats:p>We <jats:bold>show</jats:bold> a result.</jats:p>"
        self.assertEqual(strip_markup(value), "We show a result.")

    def test_openalex_inverted_abstract_is_reconstructed(self) -> None:
        inverted = {"retrieval": [3], "Claim": [0], "aware": [2], "is": [1]}
        self.assertEqual(reconstruct_openalex_abstract(inverted), "Claim is aware retrieval")


class ProviderParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = HttpClient(timeout=1, retries=0)

    def test_crossref_record(self) -> None:
        provider = CrossrefProvider(self.client)
        paper = provider.parse_item(
            {
                "DOI": "10.1234/Test",
                "title": ["A Test Paper"],
                "author": [{"given": "Ada", "family": "Lovelace"}],
                "abstract": "<jats:p>An <jats:italic>important</jats:italic> result.</jats:p>",
                "published-online": {"date-parts": [[2024, 3, 1]]},
                "container-title": ["Journal of Tests"],
                "URL": "https://doi.org/10.1234/test",
                "is-referenced-by-count": 12,
            }
        )
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.doi, "10.1234/test")
        self.assertEqual(paper.abstract, "An important result.")
        self.assertEqual(paper.authors, ["Ada Lovelace"])
        self.assertEqual(paper.year, 2024)
        self.assertEqual(paper.abstract_source, "crossref")

    def test_semantic_scholar_record(self) -> None:
        provider = SemanticScholarProvider(self.client)
        paper = provider.parse_item(
            {
                "paperId": "s2-id",
                "title": "A Test Paper",
                "abstract": "A complete abstract.",
                "year": 2024,
                "authors": [{"name": "Ada Lovelace"}],
                "externalIds": {"DOI": "10.1234/test", "ArXiv": "2401.00001"},
                "url": "https://www.semanticscholar.org/paper/s2-id",
                "venue": "Journal of Tests",
                "citationCount": 14,
                "openAccessPdf": {"url": "https://example.org/test.pdf"},
            }
        )
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.source_ids["arxiv"], "2401.00001")
        self.assertEqual(paper.source_ids["semantic_scholar"], "s2-id")
        self.assertEqual(paper.pdf_url, "https://example.org/test.pdf")

    def test_europe_pmc_record(self) -> None:
        provider = EuropePMCProvider(self.client)
        paper = provider.parse_item(
            {
                "id": "12345678",
                "source": "MED",
                "pmid": "12345678",
                "pmcid": "PMC999",
                "doi": "10.1234/test",
                "title": "A Biomedical Test",
                "abstractText": "<p>Biomedical abstract.</p>",
                "pubYear": "2022",
                "authorList": {"author": [{"fullName": "Grace Hopper"}]},
                "journalTitle": "Medical Tests",
                "citedByCount": 3,
                "fullTextUrlList": {
                    "fullTextUrl": [
                        {"documentStyle": "pdf", "url": "https://example.org/biomedical.pdf"}
                    ]
                },
            }
        )
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.source_ids["pmid"], "12345678")
        self.assertEqual(paper.url, "https://europepmc.org/articles/PMC999")
        self.assertEqual(paper.abstract, "Biomedical abstract.")

    def test_openalex_record(self) -> None:
        provider = OpenAlexProvider(self.client)
        paper = provider.parse_item(
            {
                "id": "https://openalex.org/W123",
                "display_name": "An OpenAlex Test",
                "publication_year": 2021,
                "doi": "https://doi.org/10.1234/test",
                "authorships": [{"author": {"display_name": "Edsger Dijkstra"}}],
                "abstract_inverted_index": {"A": [0], "result": [1]},
                "primary_location": {
                    "landing_page_url": "https://example.org/article",
                    "source": {"display_name": "Computing Tests"},
                },
                "best_oa_location": {"pdf_url": "https://example.org/open.pdf"},
                "ids": {"openalex": "https://openalex.org/W123", "pmid": "https://pubmed.ncbi.nlm.nih.gov/42/"},
                "cited_by_count": 9,
                "is_retracted": False,
            }
        )
        self.assertIsNotNone(paper)
        assert paper is not None
        self.assertEqual(paper.abstract, "A result")
        self.assertEqual(paper.source_ids["openalex"], "W123")
        self.assertEqual(paper.source_ids["pmid"], "42")
        self.assertFalse(paper.is_retracted)

    def test_arxiv_feed(self) -> None:
        provider = ArxivProvider(self.client)
        feed = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>https://arxiv.org/abs/2401.00001v1</id>
            <title>An arXiv Test</title>
            <summary>We report a useful result.</summary>
            <published>2024-01-01T00:00:00Z</published>
            <author><name>Barbara Liskov</name></author>
            <arxiv:doi>10.1234/test</arxiv:doi>
            <arxiv:journal_ref>Journal of Tests</arxiv:journal_ref>
            <link rel="alternate" href="https://arxiv.org/abs/2401.00001v1" />
            <link title="pdf" href="https://arxiv.org/pdf/2401.00001v1" type="application/pdf" />
          </entry>
        </feed>"""
        papers = provider.parse_feed(feed)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].source_ids["arxiv"], "2401.00001v1")
        self.assertEqual(papers[0].doi, "10.1234/test")
        self.assertEqual(papers[0].year, 2024)


class MergeAndRankingTests(unittest.TestCase):
    def test_records_merge_and_preserve_provenance(self) -> None:
        crossref = source_record(
            "crossref",
            title="Shared Paper",
            abstract="Short publisher abstract.",
            authors=["Ada Lovelace"],
            year=2024,
            doi="10.1234/shared",
            source_ids={"crossref": "10.1234/shared"},
        )
        europe = source_record(
            "europe_pmc",
            title="Shared Paper",
            abstract="Longer author abstract from a domain-specific record.",
            authors=["Ada Lovelace", "Grace Hopper"],
            year=2024,
            doi="10.1234/shared",
            source_ids={"pmid": "123"},
        )
        crossref.provider_ranks["crossref"] = 2
        europe.provider_ranks["europe_pmc"] = 1
        accumulator = PaperAccumulator()
        accumulator.add(crossref)
        accumulator.add(europe)
        papers = accumulator.ranked()
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].abstract_source, "europe_pmc")
        self.assertEqual(papers[0].authors, ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(papers[0].sources, ["crossref", "europe_pmc"])
        self.assertEqual(papers[0].provenance["abstract"], ["crossref", "europe_pmc"])

    def test_conflicting_dois_are_not_merged_by_title(self) -> None:
        accumulator = PaperAccumulator()
        accumulator.add(source_record("crossref", title="Same Title", doi="10.1/first"))
        accumulator.add(source_record("crossref", title="Same Title", doi="10.1/second"))
        self.assertEqual(len(accumulator.ranked()), 2)

    def test_multi_provider_consensus_ranks_first(self) -> None:
        consensus = Paper(title="Consensus", provider_ranks={"crossref": 2, "semantic_scholar": 3})
        singleton = Paper(title="Singleton", provider_ranks={"crossref": 1})
        accumulator = PaperAccumulator()
        accumulator.add(singleton)
        accumulator.add(consensus)
        self.assertEqual(accumulator.ranked()[0].title, "Consensus")


class _FixtureProvider(Provider):
    def __init__(self, name: str, papers: list[Paper] | None = None, error: Exception | None = None) -> None:
        super().__init__(HttpClient(timeout=1, retries=0))
        self.name = name
        self.papers = papers or []
        self.error = error

    def search(self, query: str, limit: int, year_from=None, year_to=None) -> list[Paper]:
        if self.error:
            raise self.error
        return self.papers[:limit]

    def lookup_doi(self, doi: str) -> Paper | None:
        if self.error:
            raise self.error
        for paper in self.papers:
            if paper.doi == doi:
                return paper
        return None


class ServiceTests(unittest.TestCase):
    def test_provider_failure_is_reported_without_losing_other_results(self) -> None:
        good_paper = source_record("good", title="Recovered Result", doi="10.1/good", abstract="Evidence")
        service = PaperSearchService(
            {
                "good": _FixtureProvider("good", [good_paper]),
                "bad": _FixtureProvider("bad", error=RuntimeError("upstream unavailable")),
            }
        )
        result = service.search("query", ["good", "bad"], limit_per_provider=5)
        self.assertEqual(result["total"], 1)
        self.assertIn("bad", result["errors"])
        self.assertEqual(result["papers"][0]["doi"], "10.1/good")
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
