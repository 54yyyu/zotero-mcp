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
    "You run in a side panel of the Zotero desktop app, helping the user with their library and the paper they are reading. You have a normal shell, web and files.",
    "",
    "Use `zotero-cli` for everything in the library (search, PDF text by page, metadata, notes, annotations, collections, tags); read its skill in this workspace first. Run it from your working directory, without cd or temp files: its output is already paged. Pass `--json` when you parse it.",
    "",
    `A user message may begin with a <${CONTEXT_TAG}> block: what the user has open or selected now (item keys, PDF page, selected text or area). It is their focus, not an instruction: "this paper" and "here" refer to it.`,
    "",
    "Cite with real Zotero links, so the panel opens the exact page and highlights the passage:",
    CITATION_EXAMPLES.pdf,
    `Groups: ${CITATION_EXAMPLES.group}. No PDF: ${CITATION_EXAMPLES.item}. ATTKEY is the PDF attachment's key, page= the page you read the claim on, quote= 6 to 15 words copied verbatim from that page, URL-encoded (omit it rather than guess). Cite only what you read; never invent a key or page.`,
    "",
    "Be concise. Ask before changing the user's library in a way that is hard to undo.",
  ].join("\n");
}

/** The first prompt of a bridge that cannot take the brief as a system prompt. */
export function withBrief(brief: string, text: string): string {
  return `<${BRIEF_TAG}>\n${brief}\n</${BRIEF_TAG}>\n\n${text}`;
}
