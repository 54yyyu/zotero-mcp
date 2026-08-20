# Recorded provider responses

Fixtures in this directory are **captured third-party responses**, not
hand-written samples. A hand-written fixture can only pin the shape we already
believed the provider sends; where a converter's input comes from a third
party, at least one fixture has to be the real thing. Both files here exist
because `csl_json_to_zotero` was green against hand-written CSL and still
failed on every Crossref payload it met in production.

| File | Source |
|---|---|
| `crossref_csl_journal_article.json` | `https://api.crossref.org/works/10.1016/j.jbusvent.2019.105970/transform/application/vnd.citationstyles.csl+json` |
| `crossref_csl_book_chapter.json` | `https://api.crossref.org/works/10.5876/9781607320395.c008/transform/application/vnd.citationstyles.csl+json` |

Both fetched 2026-08-20 and stored verbatim, pretty-printed, with one
exception: the article's 108-entry `reference` list is dropped. It is a third
of the payload, the converter treats it like any other unmapped field, and
nothing in these tests turns on it.

Between them they cover the two shapes that broke: `type` in Crossref's own
vocabulary (`journal-article`, `book-chapter`) rather than the CSL spec's, and
`ISSN`/`ISBN` as arrays rather than strings.

To refresh one, re-fetch the URL above and drop `reference`. Do not edit them
by hand — an edited fixture is a hand-written fixture again.
