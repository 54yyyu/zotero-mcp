"""Search and reconcile scholarly-paper metadata from public repositories."""

from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from zotero_mcp._version import __version__

VERSION = __version__
DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 2
DEFAULT_PROVIDERS = ("crossref", "semantic_scholar", "europe_pmc", "arxiv")
ALL_PROVIDERS = DEFAULT_PROVIDERS + ("openalex",)
RRF_K = 60

PROVIDER_ALIASES = {
    "s2": "semantic_scholar",
    "semantic": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "europepmc": "europe_pmc",
    "europe-pmc": "europe_pmc",
    "open_alex": "openalex",
}

# Prefer abstracts closest to the author/publisher record. All chosen text still
# carries explicit source provenance, since provider coverage and quality vary.
ABSTRACT_PRIORITY = {
    "arxiv": 100,
    "europe_pmc": 100,
    "crossref": 90,
    "semantic_scholar": 80,
    "openalex": 70,
}

TRACKED_FIELDS = (
    "title",
    "abstract",
    "authors",
    "year",
    "doi",
    "venue",
    "url",
    "pdf_url",
    "citation_count",
    "is_retracted",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_doi(value: Any) -> str | None:
    doi = clean_space(value)
    if not doi:
        return None
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = urllib.parse.unquote(doi).strip().rstrip(".,;)")
    return doi.lower() or None


def normalize_external_id(value: Any) -> str | None:
    text = clean_space(value)
    if not text:
        return None
    for prefix in (
        "https://openalex.org/",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/",
        "https://arxiv.org/abs/",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :]
    return text.strip("/ ") or None


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_space(value)).casefold()
    text = "".join(character if character.isalnum() else " " for character in text)
    return clean_space(text)


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\b(?:18|19|20|21)\d{2}\b", str(value))
        return int(match.group(0)) if match else None


def first_nonempty(values: Any) -> str | None:
    if isinstance(values, list):
        for value in values:
            cleaned = clean_space(value)
            if cleaned:
                return cleaned
        return None
    cleaned = clean_space(values)
    return cleaned or None


class _MarkupTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)


def strip_markup(value: Any) -> str | None:
    text = clean_space(value)
    if not text:
        return None
    parser = _MarkupTextExtractor()
    try:
        parser.feed(text)
        cleaned = clean_space(" ".join(parser.parts))
    except Exception:
        cleaned = clean_space(re.sub(r"<[^>]+>", " ", text))
    return html.unescape(cleaned) or None


def reconstruct_openalex_abstract(inverted_index: Any) -> str | None:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, indices in inverted_index.items():
        if not isinstance(indices, list):
            continue
        for index in indices:
            if isinstance(index, int) and index >= 0:
                positions.append((index, str(word)))
    positions.sort(key=lambda pair: pair[0])
    return clean_space(" ".join(word for _, word in positions)) or None


def source_record(source: str, **values: Any) -> Paper:
    paper = Paper(sources=[source], **values)
    for field_name in TRACKED_FIELDS:
        value = getattr(paper, field_name)
        if value not in (None, "", [], {}):
            paper.provenance[field_name] = [source]
    if paper.abstract:
        paper.abstract_source = source
    return paper


@dataclass
class Paper:
    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    citation_count: int | None = None
    is_retracted: bool | None = None
    abstract_source: str | None = None
    sources: list[str] = field(default_factory=list)
    source_ids: dict[str, str] = field(default_factory=dict)
    provider_ranks: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, list[str]] = field(default_factory=dict)
    rrf_score: float = 0.0

    def __post_init__(self) -> None:
        self.title = clean_space(self.title)
        self.abstract = clean_space(self.abstract) or None
        self.authors = [clean_space(author) for author in self.authors if clean_space(author)]
        self.doi = normalize_doi(self.doi)
        self.sources = sorted(set(source for source in self.sources if source))
        self.source_ids = {
            str(key): str(value)
            for key, value in self.source_ids.items()
            if key and value not in (None, "")
        }

    @property
    def canonical_id(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        for source_name in ("pmid", "pmcid", "arxiv", "openalex", "semantic_scholar"):
            if source_name in self.source_ids:
                return f"{source_name}:{self.source_ids[source_name]}"
        normalized = normalize_title(self.title)
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        return f"title:{digest}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "title": self.title,
            "abstract": self.abstract,
            "abstract_source": self.abstract_source,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "venue": self.venue,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "citation_count": self.citation_count,
            "is_retracted": self.is_retracted,
            "sources": self.sources,
            "source_ids": dict(sorted(self.source_ids.items())),
            "provider_ranks": dict(sorted(self.provider_ranks.items())),
            "rrf_score": round(self.rrf_score, 8),
            "provenance": {key: sorted(set(value)) for key, value in sorted(self.provenance.items())},
        }


def _add_provenance(target: Paper, incoming: Paper) -> None:
    for field_name, source_names in incoming.provenance.items():
        target.provenance.setdefault(field_name, [])
        target.provenance[field_name] = sorted(set(target.provenance[field_name] + source_names))


def merge_papers(target: Paper, incoming: Paper) -> Paper:
    """Merge complementary metadata while retaining field provenance."""
    _add_provenance(target, incoming)
    target.sources = sorted(set(target.sources + incoming.sources))

    for key, value in incoming.source_ids.items():
        target.source_ids.setdefault(key, value)
    for key, rank in incoming.provider_ranks.items():
        target.provider_ranks[key] = min(rank, target.provider_ranks.get(key, rank))

    if not target.title and incoming.title:
        target.title = incoming.title
    if len(incoming.authors) > len(target.authors):
        target.authors = list(incoming.authors)
    if target.year is None and incoming.year is not None:
        target.year = incoming.year
    if not target.doi and incoming.doi:
        target.doi = incoming.doi
    for field_name in ("venue", "url", "pdf_url"):
        if not getattr(target, field_name) and getattr(incoming, field_name):
            setattr(target, field_name, getattr(incoming, field_name))

    if incoming.abstract:
        current_priority = ABSTRACT_PRIORITY.get(target.abstract_source or "", 0)
        incoming_priority = ABSTRACT_PRIORITY.get(incoming.abstract_source or "", 0)
        replace_abstract = not target.abstract or incoming_priority > current_priority
        if incoming_priority == current_priority and len(incoming.abstract) > len(target.abstract or ""):
            replace_abstract = True
        if replace_abstract:
            target.abstract = incoming.abstract
            target.abstract_source = incoming.abstract_source

    if incoming.citation_count is not None:
        if target.citation_count is None or incoming.citation_count > target.citation_count:
            target.citation_count = incoming.citation_count
    if incoming.is_retracted is True:
        target.is_retracted = True
    elif target.is_retracted is None and incoming.is_retracted is not None:
        target.is_retracted = incoming.is_retracted
    return target


class PaperAccumulator:
    """Deduplicate by identifiers first and exact normalized title second."""

    def __init__(self) -> None:
        self.papers: list[Paper] = []
        self.by_doi: dict[str, Paper] = {}
        self.by_external_id: dict[str, Paper] = {}
        self.by_title: dict[str, Paper] = {}

    def _candidate(self, paper: Paper) -> Paper | None:
        candidates: list[Paper] = []
        if paper.doi and paper.doi in self.by_doi:
            candidates.append(self.by_doi[paper.doi])
        for key, value in paper.source_ids.items():
            external_key = f"{key}:{value}".casefold()
            if external_key in self.by_external_id:
                candidates.append(self.by_external_id[external_key])
        title_key = normalize_title(paper.title)
        if title_key and title_key in self.by_title:
            candidates.append(self.by_title[title_key])

        for candidate in candidates:
            # Do not collapse records carrying two different, explicit DOIs just
            # because their titles match (editions and corrections can do that).
            if candidate.doi and paper.doi and candidate.doi != paper.doi:
                continue
            return candidate
        return None

    def _index(self, paper: Paper) -> None:
        if paper.doi:
            self.by_doi[paper.doi] = paper
        for key, value in paper.source_ids.items():
            self.by_external_id[f"{key}:{value}".casefold()] = paper
        title_key = normalize_title(paper.title)
        if title_key:
            self.by_title[title_key] = paper

    def add(self, paper: Paper) -> None:
        candidate = self._candidate(paper)
        if candidate is None:
            self.papers.append(paper)
            self._index(paper)
            return
        merge_papers(candidate, paper)
        self._index(candidate)

    def ranked(self) -> list[Paper]:
        for paper in self.papers:
            paper.rrf_score = sum(1.0 / (RRF_K + rank) for rank in paper.provider_ranks.values())
        return sorted(
            self.papers,
            key=lambda paper: (
                -paper.rrf_score,
                min(paper.provider_ranks.values(), default=10**9),
                -(paper.citation_count or 0),
                paper.title.casefold(),
            ),
        )


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderConfigurationError(RuntimeError):
    pass


class HttpClient:
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        email: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        contact = f"; mailto:{email}" if email else ""
        self.user_agent = f"zotero-mcp/{VERSION} (+research repository search{contact})"

    def get_text(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        query = urllib.parse.urlencode(
            [(key, value) for key, value in (params or {}).items() if value is not None],
            doseq=True,
        )
        full_url = f"{url}{'&' if '?' in url else '?'}{query}" if query else url
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json, application/xml, text/xml"}
        request_headers.update(headers or {})
        retry_statuses = {429, 500, 502, 503, 504}

        for attempt in range(self.retries + 1):
            request = urllib.request.Request(full_url, headers=request_headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return response.read().decode(charset, errors="replace")
            except urllib.error.HTTPError as exc:
                body = exc.read(1000).decode("utf-8", errors="replace")
                if exc.code in retry_statuses and attempt < self.retries:
                    retry_after = exc.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else 1.0 * (2**attempt)
                    time.sleep(min(delay, 8.0))
                    continue
                detail = clean_space(body)
                suffix = f": {detail[:300]}" if detail else ""
                raise ApiError(f"HTTP {exc.code} for {url}{suffix}", status_code=exc.code) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.retries:
                    time.sleep(min(1.0 * (2**attempt), 8.0))
                    continue
                raise ApiError(f"Request failed for {url}: {exc}") from exc
        raise ApiError(f"Request failed for {url}")

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        text = self.get_text(url, params=params, headers=headers)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ApiError(f"Invalid JSON returned by {url}: {exc}") from exc
        if not isinstance(value, dict):
            raise ApiError(f"Expected a JSON object from {url}")
        return value


class Provider:
    name = "provider"
    requires_key = False

    def __init__(self, client: HttpClient, email: str | None = None) -> None:
        self.client = client
        self.email = email

    @property
    def configured(self) -> bool:
        return True

    def search(
        self,
        query: str,
        limit: int,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Paper]:
        raise NotImplementedError

    def lookup_doi(self, doi: str) -> Paper | None:
        raise NotImplementedError


def _crossref_year(item: Mapping[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        date_parts = item.get(key, {})
        if isinstance(date_parts, dict):
            rows = date_parts.get("date-parts")
            if isinstance(rows, list) and rows and isinstance(rows[0], list) and rows[0]:
                year = safe_int(rows[0][0])
                if year:
                    return year
    return None


class CrossrefProvider(Provider):
    name = "crossref"
    base_url = "https://api.crossref.org/v1/works"

    def _params(self) -> dict[str, Any]:
        return {"mailto": self.email} if self.email else {}

    def parse_item(self, item: Mapping[str, Any]) -> Paper | None:
        title = first_nonempty(item.get("title"))
        if not title:
            return None
        authors: list[str] = []
        for author in item.get("author", []) if isinstance(item.get("author"), list) else []:
            if not isinstance(author, dict):
                continue
            name = clean_space(author.get("name"))
            if not name:
                name = clean_space(f"{author.get('given', '')} {author.get('family', '')}")
            if name:
                authors.append(name)
        doi = normalize_doi(item.get("DOI"))
        pdf_url = None
        for link in item.get("link", []) if isinstance(item.get("link"), list) else []:
            if isinstance(link, dict) and "pdf" in clean_space(link.get("content-type")).lower():
                pdf_url = first_nonempty(link.get("URL"))
                if pdf_url:
                    break
        return source_record(
            self.name,
            title=title,
            abstract=strip_markup(item.get("abstract")),
            authors=authors,
            year=_crossref_year(item),
            doi=doi,
            venue=first_nonempty(item.get("container-title")),
            url=first_nonempty(item.get("URL")) or (f"https://doi.org/{doi}" if doi else None),
            pdf_url=pdf_url,
            citation_count=safe_int(item.get("is-referenced-by-count")),
            source_ids={"crossref": doi} if doi else {},
        )

    def search(self, query: str, limit: int, year_from: int | None = None, year_to: int | None = None) -> list[Paper]:
        params: dict[str, Any] = {"query.bibliographic": query, "rows": min(max(limit, 1), 1000)}
        params.update(self._params())
        filters: list[str] = []
        if year_from:
            filters.append(f"from-pub-date:{year_from}-01-01")
        if year_to:
            filters.append(f"until-pub-date:{year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        payload = self.client.get_json(self.base_url, params=params)
        items = payload.get("message", {}).get("items", [])
        return [paper for item in items if isinstance(item, dict) and (paper := self.parse_item(item))]

    def lookup_doi(self, doi: str) -> Paper | None:
        encoded_doi = urllib.parse.quote(doi, safe="/")
        payload = self.client.get_json(f"{self.base_url}/{encoded_doi}", params=self._params())
        item = payload.get("message", {})
        return self.parse_item(item) if isinstance(item, dict) else None


class SemanticScholarProvider(Provider):
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"
    fields = (
        "title,abstract,authors,year,publicationDate,externalIds,url,venue,"
        "citationCount,openAccessPdf,fieldsOfStudy,publicationTypes"
    )

    def __init__(self, client: HttpClient, email: str | None = None) -> None:
        super().__init__(client, email)
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv(
            "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY"
        )

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def _get_json(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return self.client.get_json(url, params=params, headers=self._headers())
        except ApiError as exc:
            if exc.status_code == 403 and self.api_key:
                return self.client.get_json(url, params=params)
            raise

    def parse_item(self, item: Mapping[str, Any]) -> Paper | None:
        title = first_nonempty(item.get("title"))
        if not title:
            return None
        authors = [
            clean_space(author.get("name"))
            for author in item.get("authors", []) if isinstance(author, dict) and clean_space(author.get("name"))
        ]
        external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        source_ids: dict[str, str] = {}
        mapping = {"ArXiv": "arxiv", "PubMed": "pmid", "PubMedCentral": "pmcid", "CorpusId": "corpus_id"}
        for external_name, output_name in mapping.items():
            value = normalize_external_id(external.get(external_name))
            if value:
                source_ids[output_name] = value
        paper_id = normalize_external_id(item.get("paperId"))
        if paper_id:
            source_ids[self.name] = paper_id
        doi = normalize_doi(external.get("DOI"))
        open_access = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
        return source_record(
            self.name,
            title=title,
            abstract=first_nonempty(item.get("abstract")),
            authors=authors,
            year=safe_int(item.get("year")) or safe_int(item.get("publicationDate")),
            doi=doi,
            venue=first_nonempty(item.get("venue")),
            url=first_nonempty(item.get("url")),
            pdf_url=first_nonempty(open_access.get("url")),
            citation_count=safe_int(item.get("citationCount")),
            source_ids=source_ids,
        )

    def search(self, query: str, limit: int, year_from: int | None = None, year_to: int | None = None) -> list[Paper]:
        # The official API documents that hyphenated terms do not match in this endpoint.
        semantic_query = re.sub(r"(?<=\w)-(?=\w)", " ", query)
        params: dict[str, Any] = {"query": semantic_query, "limit": min(max(limit, 1), 100), "fields": self.fields}
        if year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        payload = self._get_json(f"{self.base_url}/paper/search", params)
        items = payload.get("data", [])
        return [paper for item in items if isinstance(item, dict) and (paper := self.parse_item(item))]

    def lookup_doi(self, doi: str) -> Paper | None:
        paper_id = urllib.parse.quote(f"DOI:{doi}", safe=":")
        payload = self._get_json(f"{self.base_url}/paper/{paper_id}", {"fields": self.fields})
        return self.parse_item(payload)


class EuropePMCProvider(Provider):
    name = "europe_pmc"
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def _params(self, query: str, limit: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": min(max(limit, 1), 1000),
        }
        if self.email:
            params["email"] = self.email
        return params

    def parse_item(self, item: Mapping[str, Any]) -> Paper | None:
        title = first_nonempty(item.get("title"))
        if not title:
            return None
        author_list = item.get("authorList", {}).get("author", []) if isinstance(item.get("authorList"), dict) else []
        authors = [
            clean_space(author.get("fullName") or f"{author.get('firstName', '')} {author.get('lastName', '')}")
            for author in author_list if isinstance(author, dict)
        ]
        authors = [author for author in authors if author]
        if not authors and item.get("authorString"):
            authors = [clean_space(author) for author in str(item["authorString"]).split(",") if clean_space(author)]
        source_ids: dict[str, str] = {}
        for input_name, output_name in (("pmid", "pmid"), ("pmcid", "pmcid"), ("id", "europe_pmc")):
            value = normalize_external_id(item.get(input_name))
            if value:
                source_ids[output_name] = value
        doi = normalize_doi(item.get("doi"))
        full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", []) if isinstance(item.get("fullTextUrlList"), dict) else []
        pdf_url = None
        for link in full_text_urls:
            if not isinstance(link, dict):
                continue
            if clean_space(link.get("documentStyle")).lower() == "pdf":
                pdf_url = first_nonempty(link.get("url"))
                if pdf_url:
                    break
        pmcid = source_ids.get("pmcid")
        source = clean_space(item.get("source")).lower()
        external_id = source_ids.get("europe_pmc")
        url = f"https://europepmc.org/articles/{pmcid}" if pmcid else None
        if not url and external_id:
            url = f"https://europepmc.org/article/{source or 'med'}/{external_id}"
        return source_record(
            self.name,
            title=title,
            abstract=strip_markup(item.get("abstractText")),
            authors=authors,
            year=safe_int(item.get("pubYear") or item.get("firstPublicationDate")),
            doi=doi,
            venue=first_nonempty(item.get("journalTitle")),
            url=url,
            pdf_url=pdf_url,
            citation_count=safe_int(item.get("citedByCount")),
            source_ids=source_ids,
        )

    def search(self, query: str, limit: int, year_from: int | None = None, year_to: int | None = None) -> list[Paper]:
        api_query = query
        if year_from or year_to:
            start = f"{year_from or 1800}-01-01"
            end = f"{year_to or 2200}-12-31"
            api_query = f"({query}) AND FIRST_PDATE:[{start} TO {end}]"
        payload = self.client.get_json(self.base_url, params=self._params(api_query, limit))
        items = payload.get("resultList", {}).get("result", [])
        return [paper for item in items if isinstance(item, dict) and (paper := self.parse_item(item))]

    def lookup_doi(self, doi: str) -> Paper | None:
        payload = self.client.get_json(self.base_url, params=self._params(f'DOI:"{doi}"', 5))
        items = payload.get("resultList", {}).get("result", [])
        for item in items:
            if isinstance(item, dict) and normalize_doi(item.get("doi")) == doi:
                return self.parse_item(item)
        return None


class ArxivProvider(Provider):
    name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"
    atom = "http://www.w3.org/2005/Atom"
    arxiv = "http://arxiv.org/schemas/atom"

    @staticmethod
    def _text(element: ET.Element, path: str, namespaces: Mapping[str, str]) -> str | None:
        child = element.find(path, namespaces)
        return clean_space(child.text) if child is not None and child.text else None

    def parse_feed(self, text: str) -> list[Paper]:
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise ApiError(f"Invalid XML returned by arXiv: {exc}") from exc
        namespaces = {"atom": self.atom, "arxiv": self.arxiv}
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", namespaces):
            title = self._text(entry, "atom:title", namespaces)
            if not title:
                continue
            entry_url = self._text(entry, "atom:id", namespaces)
            arxiv_id = normalize_external_id(entry_url)
            authors = [
                clean_space(author.findtext(f"{{{self.atom}}}name"))
                for author in entry.findall("atom:author", namespaces)
            ]
            authors = [author for author in authors if author]
            pdf_url = None
            alternate_url = entry_url
            for link in entry.findall("atom:link", namespaces):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href")
                if link.attrib.get("rel") == "alternate":
                    alternate_url = link.attrib.get("href") or alternate_url
            published = self._text(entry, "atom:published", namespaces)
            doi = normalize_doi(self._text(entry, "arxiv:doi", namespaces))
            source_ids = {"arxiv": arxiv_id} if arxiv_id else {}
            papers.append(
                source_record(
                    self.name,
                    title=title,
                    abstract=self._text(entry, "atom:summary", namespaces),
                    authors=authors,
                    year=safe_int(published),
                    doi=doi,
                    venue=self._text(entry, "arxiv:journal_ref", namespaces),
                    url=alternate_url,
                    pdf_url=pdf_url,
                    source_ids=source_ids,
                )
            )
        return papers

    def _request(self, search_query: str, limit: int) -> list[Paper]:
        text = self.client.get_text(
            self.base_url,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": min(max(limit, 1), 100),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            headers={"Accept": "application/atom+xml"},
        )
        return self.parse_feed(text)

    def search(self, query: str, limit: int, year_from: int | None = None, year_to: int | None = None) -> list[Paper]:
        safe_query = clean_space(query).replace('"', "")
        search_query = f'all:"{safe_query}"'
        if year_from or year_to:
            start = f"{year_from or 1800}01010000"
            end = f"{year_to or 2200}12312359"
            search_query += f" AND submittedDate:[{start} TO {end}]"
        return self._request(search_query, limit)

    def lookup_doi(self, doi: str) -> Paper | None:
        papers = self._request(f'doi:"{doi}"', 5)
        for paper in papers:
            if paper.doi == doi:
                return paper
        return None


class OpenAlexProvider(Provider):
    name = "openalex"
    base_url = "https://api.openalex.org/works"
    requires_key = True

    def __init__(self, client: HttpClient, email: str | None = None) -> None:
        super().__init__(client, email)
        self.api_key = os.getenv("OPENALEX_API_KEY") or os.getenv("PAPER_SEARCH_MCP_OPENALEX_API_KEY")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _base_params(self) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderConfigurationError(
                "OpenAlex currently requires a free API key; set OPENALEX_API_KEY."
            )
        return {"api_key": self.api_key}

    def parse_item(self, item: Mapping[str, Any]) -> Paper | None:
        title = first_nonempty(item.get("display_name")) or first_nonempty(item.get("title"))
        if not title:
            return None
        authors = [
            clean_space(authorship.get("author", {}).get("display_name"))
            for authorship in item.get("authorships", [])
            if isinstance(authorship, dict) and isinstance(authorship.get("author"), dict)
        ]
        authors = [author for author in authors if author]
        ids = item.get("ids") if isinstance(item.get("ids"), dict) else {}
        source_ids: dict[str, str] = {}
        for input_name, output_name in (("openalex", "openalex"), ("pmid", "pmid"), ("pmcid", "pmcid"), ("mag", "mag")):
            value = normalize_external_id(ids.get(input_name) or (item.get("id") if input_name == "openalex" else None))
            if value:
                source_ids[output_name] = value
        doi = normalize_doi(item.get("doi") or ids.get("doi"))
        primary = item.get("primary_location") if isinstance(item.get("primary_location"), dict) else {}
        best_oa = item.get("best_oa_location") if isinstance(item.get("best_oa_location"), dict) else {}
        source = primary.get("source") if isinstance(primary.get("source"), dict) else {}
        return source_record(
            self.name,
            title=title,
            abstract=reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
            authors=authors,
            year=safe_int(item.get("publication_year") or item.get("publication_date")),
            doi=doi,
            venue=first_nonempty(source.get("display_name")),
            url=first_nonempty(primary.get("landing_page_url")) or first_nonempty(item.get("id")),
            pdf_url=first_nonempty(best_oa.get("pdf_url")) or first_nonempty(primary.get("pdf_url")),
            citation_count=safe_int(item.get("cited_by_count")),
            is_retracted=item.get("is_retracted") if isinstance(item.get("is_retracted"), bool) else None,
            source_ids=source_ids,
        )

    def search(self, query: str, limit: int, year_from: int | None = None, year_to: int | None = None) -> list[Paper]:
        params = self._base_params()
        params.update({"search": query, "per_page": min(max(limit, 1), 100)})
        filters: list[str] = []
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        payload = self.client.get_json(self.base_url, params=params)
        items = payload.get("results", [])
        return [paper for item in items if isinstance(item, dict) and (paper := self.parse_item(item))]

    def lookup_doi(self, doi: str) -> Paper | None:
        params = self._base_params()
        doi_path = urllib.parse.quote(doi, safe="/")
        payload = self.client.get_json(f"{self.base_url}/https://doi.org/{doi_path}", params=params)
        return self.parse_item(payload)


def build_providers(client: HttpClient, email: str | None = None) -> dict[str, Provider]:
    providers: list[Provider] = [
        CrossrefProvider(client, email),
        SemanticScholarProvider(client, email),
        EuropePMCProvider(client, email),
        ArxivProvider(client, email),
        OpenAlexProvider(client, email),
    ]
    return {provider.name: provider for provider in providers}


def parse_provider_names(value: str, providers: Mapping[str, Provider]) -> list[str]:
    lowered = clean_space(value).lower()
    if lowered in ("", "auto"):
        names = list(DEFAULT_PROVIDERS)
        if providers["openalex"].configured:
            names.append("openalex")
        return names
    if lowered == "all":
        return list(ALL_PROVIDERS)
    names: list[str] = []
    for raw_name in value.split(","):
        name = PROVIDER_ALIASES.get(clean_space(raw_name).lower(), clean_space(raw_name).lower())
        if name not in providers:
            raise ValueError(f"Unknown provider '{raw_name}'. Available: {', '.join(ALL_PROVIDERS)}")
        if name not in names:
            names.append(name)
    return names


class PaperSearchService:
    def __init__(self, providers: Mapping[str, Provider]) -> None:
        self.providers = dict(providers)

    @staticmethod
    def _in_year_range(paper: Paper, year_from: int | None, year_to: int | None) -> bool:
        if paper.year is None:
            return True
        if year_from is not None and paper.year < year_from:
            return False
        if year_to is not None and paper.year > year_to:
            return False
        return True

    def search(
        self,
        query: str,
        provider_names: Sequence[str],
        limit_per_provider: int = 5,
        max_results: int = 25,
        year_from: int | None = None,
        year_to: int | None = None,
        require_abstract: bool = False,
    ) -> dict[str, Any]:
        started = time.monotonic()
        futures: dict[str, concurrent.futures.Future[list[Paper]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(provider_names))) as executor:
            for name in provider_names:
                provider = self.providers[name]
                futures[name] = executor.submit(
                    provider.search, query, limit_per_provider, year_from, year_to
                )

            provider_counts: dict[str, int] = {}
            errors: dict[str, str] = {}
            accumulator = PaperAccumulator()
            for name in provider_names:
                try:
                    papers = futures[name].result()
                except Exception as exc:
                    provider_counts[name] = 0
                    errors[name] = str(exc)
                    continue
                filtered = [paper for paper in papers if self._in_year_range(paper, year_from, year_to)]
                provider_counts[name] = len(filtered)
                for rank, paper in enumerate(filtered, start=1):
                    paper.provider_ranks[name] = rank
                    accumulator.add(paper)

        papers = accumulator.ranked()
        if require_abstract:
            papers = [paper for paper in papers if paper.abstract]
        papers = papers[: max(0, max_results)]
        return {
            "operation": "search",
            "query": query,
            "generated_at": utc_now_iso(),
            "providers_requested": list(provider_names),
            "provider_counts": provider_counts,
            "errors": errors,
            "raw_total": sum(provider_counts.values()),
            "total": len(papers),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "papers": [paper.as_dict() for paper in papers],
        }

    def lookup_doi(
        self,
        doi: str,
        provider_names: Sequence[str],
        require_abstract: bool = False,
    ) -> dict[str, Any]:
        started = time.monotonic()
        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            raise ValueError("A valid DOI is required")
        futures: dict[str, concurrent.futures.Future[Paper | None]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(provider_names))) as executor:
            for name in provider_names:
                futures[name] = executor.submit(self.providers[name].lookup_doi, normalized_doi)

            accumulator = PaperAccumulator()
            provider_counts: dict[str, int] = {}
            errors: dict[str, str] = {}
            for name in provider_names:
                try:
                    paper = futures[name].result()
                except ApiError as exc:
                    # A provider not knowing a DOI is expected, not fatal to the others.
                    provider_counts[name] = 0
                    if exc.status_code != 404:
                        errors[name] = str(exc)
                    continue
                except Exception as exc:
                    provider_counts[name] = 0
                    errors[name] = str(exc)
                    continue
                provider_counts[name] = 1 if paper else 0
                if paper:
                    paper.provider_ranks[name] = 1
                    accumulator.add(paper)

        papers = accumulator.ranked()
        if require_abstract:
            papers = [paper for paper in papers if paper.abstract]
        return {
            "operation": "lookup_doi",
            "doi": normalized_doi,
            "generated_at": utc_now_iso(),
            "providers_requested": list(provider_names),
            "provider_counts": provider_counts,
            "errors": errors,
            "total": len(papers),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "papers": [paper.as_dict() for paper in papers],
        }


def provider_status(providers: Mapping[str, Provider]) -> dict[str, Any]:
    descriptions = {
        "crossref": "Broad DOI metadata; abstracts only when deposited by members.",
        "semantic_scholar": "Broad relevance search and abstracts; optional key improves rate limits.",
        "europe_pmc": "Life-sciences metadata, abstracts, identifiers, and OA links.",
        "arxiv": "Preprints with author abstracts and direct PDF links.",
        "openalex": "Broad scholarly graph and reconstructed abstracts; free API key required.",
    }
    return {
        "generated_at": utc_now_iso(),
        "providers": [
            {
                "name": name,
                "configured": providers[name].configured,
                "requires_key": providers[name].requires_key,
                "description": descriptions[name],
            }
            for name in ALL_PROVIDERS
        ],
        "environment": {
            "contact_email_set": bool(os.getenv("SCHOLAR_API_EMAIL")),
            "semantic_scholar_key_set": bool(
                os.getenv("SEMANTIC_SCHOLAR_API_KEY")
                or os.getenv("PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY")
            ),
            "openalex_key_set": bool(
                os.getenv("OPENALEX_API_KEY") or os.getenv("PAPER_SEARCH_MCP_OPENALEX_API_KEY")
            ),
        },
    }
