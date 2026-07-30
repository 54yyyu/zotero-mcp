# Research Repository Providers

Research checked July 2026. API behavior and rate limits can change, so the
provider layer isolates each service behind a small adapter.

## Implemented providers

| Provider | Primary role | Authentication | Abstract behavior | Important limitation |
|---|---|---|---|---|
| Crossref | DOI identity and publisher-deposited metadata | No key; contact email recommended | Present only when a depositor supplies it; may contain JATS markup | Abstract coverage is incomplete |
| Semantic Scholar | Broad relevance search, abstracts, citation counts, OA links | Anonymous access; optional key improves limits | Plaintext abstract when available | Anonymous calls can be rate-limited |
| Europe PMC | Biomedical/life-sciences metadata and enrichment | No key | `resultType=core` includes abstracts and full metadata | Domain-specific rather than universal |
| arXiv | Preprint discovery, author abstracts, direct PDFs | No key | Author-provided summary in Atom feed | Strong in covered preprint fields, not a universal registry |
| OpenAlex | Broad graph, identifiers, topics, retraction flag, OA location | Free API key currently required | Supplied as an inverted word-position index and reconstructed locally | Plaintext is not supplied directly; key required |

Official references:

- [Crossref REST API](https://support.crossref.org/hc/en-us/articles/214320426-REST-API)
- [Crossref access and polite-pool guidance](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/)
- [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/)
- [Europe PMC REST service](https://dev.europepmc.org/RestfulWebService)
- [OpenAlex API overview](https://developers.openalex.org/api-reference/introduction)
- [OpenAlex work schema](https://developers.openalex.org/api-reference/works/get-a-single-work)
- [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html)

## What the linked MCP contributes

[`openags/paper-search-mcp`](https://github.com/openags/paper-search-mcp) has a
wide connector inventory and demonstrates a useful two-layer structure:
source-specific adapters below a unified search/download surface. Its current
scope includes repositories and download fallbacks that are valuable later,
including CORE, OpenAIRE, PubMed Central, Unpaywall, HAL, and Zenodo.

For Zotero enrichment, breadth alone is insufficient. The ingestion layer also
needs to retain where every field came from and merge complementary records. A
first-seen dedupe can discard an abstract from one provider when another provider
supplies the DOI first. The Zotero MCP integration therefore adds:

1. DOI and external-ID normalization.
2. Exact normalized-title fallback matching.
3. Conflict protection when identical titles carry different explicit DOIs.
4. Field-level provenance.
5. Deterministic abstract-source priority.
6. Reciprocal-rank fusion across provider rankings.
7. Partial-success envelopes rather than an all-or-nothing query.

## Why Google Scholar is deferred

The Google Scholar MCPs inspected for this project scrape result pages. The
commonly labeled `Abstract` value comes from the Scholar result-snippet element,
which is not guaranteed to contain the full author abstract. Google also treats
search scrapers and other automated querying as automated traffic and may return
CAPTCHAs or blocks.

Scholar can still serve as an explicitly enabled discovery fallback for titles
or version clues. It should not overwrite a Zotero abstract or become the
canonical source without retrieving and validating metadata from the publisher
or another documented scholarly API.

References:

- [Google's automated-traffic guidance](https://support.google.com/websearch/answer/86640)
- [The linked Google Scholar MCP scraper](https://github.com/JackKuo666/Google-Scholar-MCP-Server/blob/main/google_scholar_web_search.py)

## Good next connectors

The next additions should be driven by a measured coverage test against the
actual Zotero library:

- Unpaywall: DOI-to-open-access-location resolution; requires an email.
- PubMed E-utilities: authoritative biomedical identifiers and indexing.
- CORE/OpenAIRE: repository and lawful full-text discovery.
- ACL Anthology: particularly useful for computational linguistics.
- DataCite: datasets, preprints, reports, and other DOI-registered objects that
  may not be Crossref records.

## Follow-on ingestion workflow

The online-discovery tools provide the metadata needed for a later bulk
enrichment workflow:

1. Export or read Zotero items missing `abstractNote`.
2. Query by DOI across providers.
3. If DOI is missing, search by title plus first author and require a conservative
   match score before accepting an identity.
4. Save the selected abstract with source URL, retrieval time, and content hash.
5. Write to a sidecar database first; add Zotero write-back only after a dry-run
   diff is reviewable.
6. Embed title-plus-abstract under the Zotero key while retaining the original
   text and provenance for inspection.
