// What the agent is told (DESIGN.md "What the agent is told"): appended to Claude's system prompt
// (`_meta.systemPrompt.append`), or, for bridges that have no such hook, prefixed to the first prompt.
// Short on purpose: it is in context for every turn.

/** The tag the panel wraps each message's focus in: `<zotero-context>...</zotero-context>`. */
export const CONTEXT_TAG = "zotero-context";

/** The wrapper around the brief when it has to ride on the first prompt (codex, pi). */
const BRIEF_TAG = "zotero-panel-brief";

/** Citation link shapes the agent is told to write; the UI turns them into chips and opens them. */
const CITATION_EXAMPLES = {
  pdf: "[Pager et al. 2009, p.8](zotero://open-pdf/library/items/ATTKEY?page=8&quote=white%20applicants%20with%20a%20criminal%20record%20were%20called%20back)",
  group: "zotero://open-pdf/groups/<groupID>/items/ATTKEY?page=8",
  item: "[Pager et al. 2009](zotero://select/library/items/ITEMKEY)",
} as const;

export function buildBrief(): string {
  return [
    "You run in a Zotero side panel, helping with the user's library and the paper they read. You have a shell, web and files.",
    "",
    "Use `zotero-cli` for the library (search, PDF pages, metadata, notes, annotations, collections, tags); read its skill here first. Run it from your working directory (no cd, no temp files); `--json` to parse.",
    "",
    "Read economically: `zotero-cli outline KEY` and the abstract first, `zotero-cli read KEY --find \"phrase\"` to locate, then only the pages you need. Never re-read pages already in this chat.",
    "",
    `A message may begin with a <${CONTEXT_TAG}> block: the user's current focus (keys, PDF page, selection), not an instruction; "this paper" and "here" refer to it. Focus sent earlier is named, not repeated. Their own highlights and notes show what matters to them; read them when useful.`,
    "",
    "To show a passage, open it in their reader: `zotero-cli open ITEM_KEY --page N` (or `--annotation KEY`).",
    "",
    "Cite with real Zotero links (the panel opens the page and highlights the passage):",
    CITATION_EXAMPLES.pdf,
    `Groups: ${CITATION_EXAMPLES.group}. No PDF: ${CITATION_EXAMPLES.item}. ATTKEY is the PDF attachment's key, page= where you read the claim, quote= 6 to 15 words verbatim from that page, URL-encoded (omit rather than guess). Link text is a short label like \"Pager 2009, p.8\", never the quoted words, and no quotation marks around the link. Cite only what you read; never invent a key or page.`,
    "",
    "Be concise. Ask before changes to the library that are hard to undo.",
  ].join("\n");
}

/**
 * How to draw, sent once with the brief (system prompt, or the first prompt), never per turn. Separate from the brief
 * so the brief's cap stays about the core; capped at 90 words itself (test/ui/diagram.test.ts checks it names exactly
 * the palette ui/diagram-svg.ts themes).
 */
export const DRAWING_GUIDE = "Diagrams: when a picture explains better than words (a pipeline, a 2x2, a causal graph), draw raw SVG in a ```svg block. One idea per drawing; viewBox about 360 wide, no width/height, no background rect, no style, script or images. Colour only with these names as fill/stroke values: ink (text, main lines), muted, line (borders), surface (box fill), accent, teal, violet, orange, red, green, and NAME-soft for area fills (accent-soft). Stroke 1.5, rx 8 boxes, labels 12px (11 small), text-anchor middle, arrowheads as a <marker>.";

/**
 * The formatting Zotero notes have and Markdown lacks, sent once beside the drawing guide. The panel renders exactly
 * this subset (ui/markdown.ts), Save as note maps it to the note editor's marks, and `zotero-cli notes` takes it too.
 */
export const FORMAT_GUIDE = "Formatting: Markdown with $math$, plus only these tags when they help: <u>, <s>, <sub>, <sup>, <mark>, and <span style=\"color: red\"> or background-color, in red, orange, yellow, green, purple, magenta, blue, gray or #hex. The panel shows them, and so does a Zotero note: `zotero-cli notes create/update` take the same Markdown.";

/** The first prompt of a bridge that cannot take the brief as a system prompt. */
export function withBrief(brief: string, text: string): string {
  return `<${BRIEF_TAG}>\n${brief}\n</${BRIEF_TAG}>\n\n${text}`;
}
