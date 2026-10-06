// The open paper ahead of time (zotero/paper.ts): the metadata text, the page-marked file, its cache key and cleanup,
// and the rule that the first message never waits more than WAIT_MS for the file.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { WAIT_MS, aboutText, evictions, fileName, fullTextLine, headLine, markPages, parseHead, within } from "../../src/zotero/paper.ts";

const FACTS = {
  itemKey: "ITEM1234", creators: ["Brantly Callaway", "Pedro H. C. Sant'Anna"], year: "2021", venue: ["Journal of Econometrics"],
  doi: "10.1016/j.jeconom.2020.12.001", abstract: "In this article, we consider  identification,\n estimation and inference.", tags: ["did", "methods"],
  collections: ["Causal inference"], pages: 39,
};

describe("aboutText", () => {
  it("says what `get metadata` would, in a few lines", () => {
    assert.equal(aboutText(FACTS), [
      "About item ITEM1234: Brantly Callaway, Pedro H. C. Sant'Anna · 2021 · Journal of Econometrics · DOI 10.1016/j.jeconom.2020.12.001 · 39 pages",
      "Tags: did, methods",
      "In collections: Causal inference",
      "Abstract: In this article, we consider identification, estimation and inference.",
    ].join("\n"));
  });
  it("caps the abstract at 1,500 characters, 20 tags and 8 authors", () => {
    const t = aboutText({ ...FACTS, abstract: "word ".repeat(1000), tags: Array.from({ length: 25 }, (_, i) => `t${i}`), creators: Array.from({ length: 12 }, (_, i) => `A${i}`) });
    const abs = /Abstract: (.*)/.exec(t)![1]!;
    assert.equal(abs.length, 1500);
    assert.ok(abs.endsWith("…"));
    assert.match(t, /Tags: t0, .*t19 and 5 more$/m);
    assert.match(t, /A7 and 4 more · 2021/);
  });
  it("leaves out what the item does not have", () => {
    assert.equal(aboutText({ itemKey: "K", creators: [], year: "", venue: [], doi: "", abstract: "", tags: [], collections: [] }), "About item K: no metadata");
  });
});

describe("the full-text file", () => {
  it("is named by the attachment key and a short slug", () => {
    assert.equal(fileName("3QW3D95Y", "Callaway", "2021", "Difference-in-Differences"), "3QW3D95Y-callaway-2021.txt");
    assert.equal(fileName("3QW3D95Y", "Sant'Anna Ñúñez", "", "x"), "3QW3D95Y-sant-anna-nunez.txt");
    assert.equal(fileName("3QW3D95Y", "", "", "Structuring Sparsity: Block-Sparse Featurizers Capture"), "3QW3D95Y-structuring-sparsity-block-sparse.txt");
    assert.equal(fileName("3QW3D95Y", "", "", ""), "3QW3D95Y.txt");
  });

  it("starts with a line that says what it is and which PDF it came from (the cache key)", () => {
    const head = headLine({ title: "A  title", itemKey: "ITEM1234", attKey: "ATT12345", pages: 39, source: "123 bytes, modified 1700000000000" });
    assert.equal(head, 'Full text of "A title" (item ITEM1234, PDF ATT12345), 39 pages; a line [p.N] starts page N, as zotero-cli read and citations count pages. Source: 123 bytes, modified 1700000000000.');
    assert.deepEqual(parseHead(`${head}\n\n[p.1]\nbody`), { pages: 39, source: "123 bytes, modified 1700000000000" });
    assert.equal(parseHead("anything else\n"), null);
    assert.equal(parseHead(""), null);
  });

  it("marks each page of Zotero's text (pages end with a form feed) with [p.N], 1-based", () => {
    assert.deepEqual(markPages("Title\nIntro.\fSecond page.\n\f\fThe end.\f"), {
      body: "[p.1]\n\nTitle\nIntro.\n\n[p.2]\n\nSecond page.\n\n[p.3]\n\n(no text on this page: a scan or a picture)\n\n[p.4]\n\nThe end.", pages: 4,
    });
    assert.deepEqual(markPages("one page"), { body: "[p.1]\n\none page", pages: 1 });
    assert.deepEqual(markPages(" \f \f"), { body: "", pages: 0 }, "no text at all: no file");
  });

  it("the agent is told where it is, once, in one line", () => {
    assert.equal(fullTextLine("/x/papers/ATT12345-fel-2026.txt", 39), "Full text is at /x/papers/ATT12345-fel-2026.txt (39 pages, page markers like [p.7]); read it with grep/sed or zotero-cli read for specific pages.");
    assert.match(fullTextLine("/f", 1), /\(1 page,/);
  });
});

describe("the cache's cleanup", () => {
  const DAY = 86_400_000, MB = 1024 * 1024, now = 1_800_000_000_000;
  const f = (name: string, mb: number, daysAgo: number) => ({ path: `/p/${name}`, size: mb * MB, mtime: now - daysAgo * DAY });
  it("drops files unused for 60 days, then the least recently used beyond 200 MB", () => {
    const files = [f("AAAAAAAA-a.txt", 1, 61), f("BBBBBBBB-b.txt", 120, 30), f("CCCCCCCC-c.txt", 60, 10), f("DDDDDDDD-d.txt", 50, 1)];
    assert.deepEqual(evictions(files, now), ["/p/AAAAAAAA-a.txt", "/p/BBBBBBBB-b.txt"]);
    assert.deepEqual(evictions(files.slice(2), now), [], "under both limits nothing goes");
  });
  it("keeps only the newest file of an attachment (a renamed paper gets a new slug)", () => {
    assert.deepEqual(evictions([f("AAAAAAAA-old-title.txt", 1, 5), f("AAAAAAAA-new-title.txt", 1, 1), f("BBBBBBBB.txt", 1, 9)], now), ["/p/AAAAAAAA-old-title.txt"]);
  });
});

describe("within: the first message never waits long for the file", () => {
  it("a ready file is used at once; a slow one is skipped after the wait; a failure is no file", async () => {
    assert.deepEqual(await within(Promise.resolve({ path: "/f", pages: 3 }), 50), { path: "/f", pages: 3 });
    assert.equal(await within(Promise.reject(new Error("cli failed")), 50), null);
    const t0 = Date.now();
    assert.equal(await within(new Promise<null>(() => {}), WAIT_MS), null);
    const waited = Date.now() - t0;
    assert.ok(WAIT_MS === 1500 && waited >= 1490 && waited < 1700, `waited ${waited} ms`);
  });
});
