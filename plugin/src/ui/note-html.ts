// Markdown to Zotero note HTML, for "Save as note". PURE: no DOM, no Zotero, so `node --test` checks it.
// It walks markdown.ts's tree, which already holds the link and image policy (http(s) and zotero: links only), and
// escapes every text and attribute it writes: the note gets the tags below and nothing from the agent as markup.
// Math becomes the note editor's own math nodes, which it renders with KaTeX: <span class="math">$…$</span> and
// <pre class="math">$$…$$</pre> (schema version 9; Zotero writes 8 when a note has no math). Citation links stay
// ordinary zotero:// links, which the note keeps clickable. A diagram becomes an embedded image when the host
// imported its PNG (`image(i)`), else its SVG source as code.
import { mdToTree } from "./markdown.ts";
import type { MdEl, MdNode } from "./markdown.ts";

export interface NoteHtmlOpts {
  /** The note's first line (Zotero titles a note by it). */
  title?: string;
  /** The embedded image for the i-th closed ```svg block, or null to keep it as code. */
  image?(i: number): { key: string; width: number; height: number } | null;
}

/** Tags written as they are (their attributes are not). */
const PLAIN = new Set(["p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "li", "blockquote", "em", "strong", "del", "code", "table", "thead", "tbody", "tr", "th", "td"]);

// eslint-disable-next-line no-control-regex
const esc = (s: string): string => s.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const textOf = (n: MdNode): string => (typeof n === "string" ? n : (n.kids ?? []).map(textOf).join(""));
const px = (n: number): number => Math.max(1, Math.min(4000, Math.round(Number(n) || 1)));
const isBlank = (n: MdNode) => typeof n === "string" && !n.trim();

/** The ```svg blocks that count as drawings (closed fences), in order: the UI renders a PNG for each of these. */
export function noteDiagrams(markdown: string): string[] {
  const out: string[] = [];
  const visit = (nodes: MdNode[]) => {
    for (const n of nodes) {
      if (typeof n === "string") continue;
      if (n.tag === "diagram") { if (n.attrs?.open !== "1") out.push(textOf(n)); }
      else if (n.kids) visit(n.kids);
    }
  };
  visit(mdToTree(markdown));
  return out;
}

export function noteHtml(markdown: string, o: NoteHtmlOpts = {}): string {
  let diagrams = 0;
  let math = false;

  const mathHtml = (tex: string, block: boolean): string => {
    math = true; // the editor strips only the outer delimiters, so the TeX goes in as it is
    return block ? `<pre class="math">$$${esc(tex)}$$</pre>` : `<span class="math">$${esc(tex)}$</span>`;
  };

  const kids = (n: MdEl, block = false): string => (n.kids ?? []).map((k) => node(k, block)).join("");

  // `block`: the node sits where a block may go (the top level, a blockquote).
  function node(n: MdNode, block = false): string {
    if (typeof n === "string") return esc(n);
    const a = n.attrs ?? {};
    switch (n.tag) {
      case "math": {
        const tex = textOf(n);
        if (a.display !== "1") return mathHtml(tex, false);
        return block ? mathHtml(tex, true) : mathHtml(`\\displaystyle ${tex}`, false);
      }
      case "p": {
        // A paragraph that is only a display formula is a math block, not a formula in a paragraph.
        const real = (n.kids ?? []).filter((k) => !isBlank(k));
        const only = real[0];
        if (real.length === 1 && typeof only !== "string" && only?.tag === "math" && only.attrs?.display === "1") return mathHtml(textOf(only), true);
        return `<p>${kids(n)}</p>`;
      }
      case "codeblock":
      case "pre":
        return `<pre>${esc(textOf(n))}</pre>`;
      case "diagram": {
        const src = textOf(n);
        if (a.open === "1") return `<pre>${esc(src)}</pre>`;
        const img = o.image?.(diagrams++);
        return img ? `<p><img data-attachment-key="${esc(img.key)}" width="${px(img.width)}" height="${px(img.height)}"></p>` : `<pre>${esc(src)}</pre>`;
      }
      case "a":
      case "cite": {
        const href = a.href ?? ""; // markdown.ts let only http(s) and zotero: links through
        const inner = n.tag === "cite" ? esc(textOf(n)) : kids(n);
        return /^(?:https?|zotero):\/\//i.test(href) ? `<a href="${esc(href)}">${inner}</a>` : inner;
      }
      case "ol": return `<ol${a.start && /^\d{1,6}$/.test(a.start) ? ` start="${a.start}"` : ""}>${kids(n)}</ol>`;
      case "br": return "<br>";
      case "hr": return "<hr>";
      case "img": return esc(a.alt ?? ""); // a data: image inside the answer: its words, not a second attachment path
      case "blockquote": return `<blockquote>${kids(n, true)}</blockquote>`;
    }
    if (PLAIN.has(n.tag)) return `<${n.tag}>${kids(n)}</${n.tag}>`;
    return kids(n, block); // tablewrap, our spans: their content only
  }

  const body = mdToTree(markdown.replace(/\r\n?/g, "\n")).map((n) => node(n, true)).join("");
  const title = o.title?.trim() ? `<h1>${esc(o.title.replace(/\s+/g, " ").trim())}</h1>` : "";
  return `<div data-schema-version="${math ? 9 : 8}">${title}${body}</div>`;
}
