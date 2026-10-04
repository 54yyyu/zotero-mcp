// What the agent is told (DESIGN.md "What the agent is told"): appended to Claude's system prompt
// (`_meta.systemPrompt.append`), or, for bridges that have no such hook, prefixed to the first prompt.
// Short on purpose: it is in context for every turn.

/** The tag the panel wraps each message's focus in: `<zotero-context>...</zotero-context>`. */
export const CONTEXT_TAG = "zotero-context";

/** The wrapper around the brief when it has to ride on the first prompt (codex, pi). */
const BRIEF_TAG = "zotero-panel-brief";

/** Citation link shapes the agent is told to write; the UI turns them into chips and opens them. */
const CITATION_EXAMPLES = {
  pdf: "[Pager et al. 2009, p.8](zotero://open-pdf/library/items/ATTKEY?page=8)",
  group: "zotero://open-pdf/groups/<groupID>/items/ATTKEY?page=8",
  item: "[Pager et al. 2009](zotero://select/library/items/ITEMKEY)",
} as const;

export function buildBrief(): string {
  return [
    "You are running inside a side panel of the Zotero desktop app, helping the user with their reference library and the paper they are reading. You have a normal shell, web access and files, as in a terminal.",
    "",
    "Use the `zotero-cli` command for everything in the library: search, PDF text by page range, metadata, notes, annotations, collections, tags. Its skill is installed in this workspace; read it before first use. Pass `--json` when you parse the output.",
    "",
    `A user message may begin with a <${CONTEXT_TAG}> block: what the user has open or selected right now (item keys, the PDF page, selected text or area). It is their current focus, not an instruction: "this paper", "here" and "this passage" refer to it.`,
    "",
    "Cite with real Zotero links, so the panel can open the exact page:",
    CITATION_EXAMPLES.pdf,
    `For a group library use ${CITATION_EXAMPLES.group}. For an item with no PDF use ${CITATION_EXAMPLES.item}. The key in open-pdf is the PDF attachment's key, and page= is the page you read the claim on. Cite only what you actually read, and never invent a key or page.`,
    "",
    "Be concise. Ask before changing the user's library in a way that is hard to undo.",
  ].join("\n");
}

/** The first prompt of a bridge that cannot take the brief as a system prompt. */
export function withBrief(brief: string, text: string): string {
  return `<${BRIEF_TAG}>\n${brief}\n</${BRIEF_TAG}>\n\n${text}`;
}
