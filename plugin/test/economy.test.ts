// Context economy: a chip goes to the agent in full once per session and again only when it changes.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { ContextChip } from "../src/types.ts";
import { chatStats, chipHash, contextFill, estimateTokens, imageTokens, planContext, repeatLine, sentLine } from "../src/ui/economy.ts";
import { replay } from "../src/ui/transcript.ts";
import { Chat } from "../src/ui/chat.ts";
import { FakeHost } from "../src/ui/fake-host.ts";
import { env } from "../src/ui/dom.ts";

env.win = globalThis as typeof env.win; // the transcript saver's timers

/** A PNG header of the given size (enough for imageTokens), padded to a realistic payload. */
function png(w: number, h: number, bytes = 60_000): string {
  const b = Buffer.alloc(bytes);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]).copy(b, 0);
  b.write("IHDR", 12, "latin1");
  b.writeUInt32BE(w, 16);
  b.writeUInt32BE(h, 20);
  return b.toString("base64");
}

const INDEX = "In this PDF: 14 annotations (9 highlights, 3 notes, 2 areas); on this page: 2 highlights. Read them with `zotero-cli annotations list --item-key ITEM1` if useful.";
const reader = (pageIndex: number, index = INDEX): ContextChip => ({ id: "reader:ATT1", kind: "reader", label: "Bell 2017", auto: true, pinned: false, ref: { libraryID: 1, itemKey: "ITEM1", attachmentKey: "ATT1", pageIndex, pageLabel: String(pageIndex + 1) }, text: index });
const SELECTION: ContextChip = { id: "selection:ATT1", kind: "selection", label: "Text Selection", auto: true, pinned: false, ref: { libraryID: 1, attachmentKey: "ATT1", pageIndex: 2, pageLabel: "3" }, text: "Assigning a professional destination would increase pay. ".repeat(26).slice(0, 1500) };
const AREA: ContextChip = { id: "area:ANN1", kind: "area", label: "Selected Area · p.3", auto: true, pinned: false, ref: { libraryID: 1, attachmentKey: "ATT1", annotationKey: "ANN1", pageIndex: 2, pageLabel: "3" }, image: { mime: "image/png", data: png(900, 600) } };

describe("planContext", () => {
  it("sends a chip in full once, then marks it repeat until it changes", () => {
    const sent = new Map<string, string>();
    assert.deepEqual(planContext([reader(2), SELECTION], sent).map((c) => !!c.repeat), [false, false]);
    assert.deepEqual(planContext([reader(2), SELECTION], sent).map((c) => !!c.repeat), [true, true]);
    assert.deepEqual(planContext([reader(3), SELECTION], sent).map((c) => !!c.repeat), [false, true], "a page turn refreshes the where-am-I line");
    const edited = { ...SELECTION, text: "another passage" };
    assert.equal(planContext([edited], sent)[0]!.repeat, undefined, "a different selection goes in full");
  });

  it("the fingerprint covers what the agent reads: page, text, image", () => {
    assert.equal(chipHash(reader(1)), chipHash(reader(1)));
    assert.notEqual(chipHash(reader(1)), chipHash(reader(2)));
    assert.notEqual(chipHash(AREA), chipHash({ ...AREA, image: { mime: "image/png", data: png(900, 601) } }));
    assert.notEqual(chipHash(AREA), chipHash({ ...AREA, image: undefined }), "an area whose PNG arrives later is sent again with it");
  });

  it("a selected annotation with the live selection's text is said once, with its key", () => {
    const ann: ContextChip = { id: "annotation:H1", kind: "annotation", label: "highlight · p.3", auto: true, pinned: false, ref: { libraryID: 1, annotationKey: "H1", pageIndex: 2 }, text: ` ${SELECTION.text}\n` };
    assert.deepEqual(planContext([reader(2), SELECTION, ann], new Map()).map((c) => c.kind), ["reader", "annotation"]);
  });

  it("the reminder names each repeated chip briefly, so 'this selection' still resolves", () => {
    const line = repeatLine(planContext([reader(2), SELECTION, AREA], new Map()).map((c) => ({ ...c, repeat: true })));
    assert.match(line, /^Still open, unchanged: reading Bell 2017 p\.3\.\nStill pointing at, unchanged/);
    assert.match(line, /selected text p\.3 "Assigning a professional destination would increase pay\. Assigning…"/);
    assert.match(line, /selected area p\.3 \(annotation ANN1\)/);
    assert.ok(line.length < 280, `short: ${line.length} chars`);
    assert.equal(repeatLine([reader(2)]), "", "nothing repeated, no line");
  });
});

describe("token estimate", () => {
  it("text at 4 characters a token, images by their pixel size", () => {
    assert.equal(imageTokens(png(900, 600)), 720);
    assert.equal(imageTokens(png(3136, 1568)), Math.ceil((1568 * 784) / 750), "fitted to a 1568 px long edge");
    assert.equal(imageTokens("not a png"), 1600, "unknown: the cap");
    assert.equal(estimateTokens({ text: "x".repeat(400), images: [{ mime: "image/png", data: png(900, 600) }] }), 820);
  });
});

describe("a chat sends unchanged context once", () => {
  const deps = (host: FakeHost) => ({ host, onChange() {}, blockReason: () => null, turnEnded() {}, setupFailed() {}, sessionChanged() {} });

  it("10 turns with a persistent area image and selection: the image goes once, the saving is measured", async (t) => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const chat = new Chat(deps(host));
    // The user turns the page at turn 5 and adds a highlight at turn 8 (the PDF's annotation counts change).
    const index8 = INDEX.replace("14 annotations (9 highlights", "15 annotations (10 highlights").replace("2 highlights.", "3 highlights.");
    const chips = (turn: number) => [reader(turn < 5 ? 2 : 3, turn < 8 ? INDEX : index8), SELECTION, AREA];
    let full = 0;
    for (let i = 0; i < 10; i++) {
      host.sim.nextAnswer = "ok"; // a plain answer: the fake agent reads "notes" in the prompt as a request to add one
      await chat.send(`question ${i}`, chips(i));
      const d = host.describeContext(chips(i));
      full += estimateTokens({ text: `${d.text}\n\nquestion ${i}`, images: d.images });
    }
    const prompts = host.sim.prompts;
    assert.equal(prompts.length, 10);
    assert.equal(prompts.filter((p) => p.images?.length).length, 1, "the unchanged image is never sent again");
    assert.equal(prompts.filter((p) => p.text.includes(SELECTION.text!)).length, 1, "the selection text goes once");
    assert.ok(prompts.slice(1).every((p) => /Still pointing at, unchanged/.test(p.text)), "later turns carry the reminder");
    assert.match(prompts[5]!.text, /- reader: Bell 2017/, "the page turn re-sends the reader line");
    assert.deepEqual(prompts.map((p, i) => (p.text.includes("In this PDF: 1") ? i : -1)).filter((i) => i >= 0), [0, 5, 8], "the annotation index goes once, then with the reader line on a page turn or when the counts change");
    assert.ok(prompts[8]!.text.includes("15 annotations"));
    const delta = prompts.reduce((n, p) => n + estimateTokens(p), 0);
    t.diagnostic(`10 turns: ${full} tokens resending everything, ${delta} with delta context (${Math.round((100 * delta) / full)}%)`);
    assert.ok(delta < full * 0.3, `delta ${delta} vs full ${full}`);
  });

  it("a new session, a resumed chat or a compaction sends everything in full again", async () => {
    const host = new FakeHost({ speed: 0, noHistory: true });
    const chat = new Chat(deps(host));
    const used = [reader(2), SELECTION, AREA];
    const say = (q: string) => { host.sim.nextAnswer = "ok"; return chat.send(q, used); };
    await say("one");
    await say("two");
    chat.dispatch({ t: "notice", level: "info", message: "Older parts of this chat were summarised to make room.", compacted: true });
    await say("three");
    await chat.closeSession(); // the bridge died: the next send starts (or resumes) a session
    await say("four");
    assert.deepEqual(host.sim.prompts.map((p) => p.images?.length ?? 0), [1, 0, 1, 1]);
  });
});

describe("the context popover's numbers", () => {
  it("come from the transcript: the last fill (a compaction's too), messages, the last turn, compactions", () => {
    const tr = replay([
      { t: "user", id: "u1", text: "q", chips: [] }, { t: "turn_start", turn: "t1" },
      { t: "turn_end", turn: "t1", stop: "end_turn", usage: { inputTokens: 24000, outputTokens: 47, contextUsed: 150000, contextSize: 200000 } },
      { t: "notice", level: "info", message: "summarised", compacted: true, context: { used: 6000, size: 200000 } },
    ]);
    assert.deepEqual(contextFill(tr), { used: 6000, size: 200000, pct: 3 });
    assert.deepEqual(chatStats(tr), { messages: 2, compactions: 1, last: { inputTokens: 24000, outputTokens: 47, contextUsed: 150000, contextSize: 200000 } });
    assert.equal(contextFill(replay([{ t: "notice", level: "info", message: "x", compacted: true }])), null, "an automatic compaction says no fill of its own");
  });

  it("sentLine names what went in full and what was only named", () => {
    const sent = new Map<string, string>();
    planContext([reader(2), SELECTION], sent);
    assert.equal(sentLine(planContext([reader(3), SELECTION], sent)), "Reader p.4 in full; selection p.3 named only (unchanged)");
    assert.equal(sentLine([]), "");
  });
});
