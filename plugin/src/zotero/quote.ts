// Finding a quoted passage in a PDF page's text, and the rects that flash over it in the reader.
// PURE (runs under node --test): open.ts feeds it the reader's own per-character page data.

/** One character of the reader's page data (`view._pdfPages[i].chars`): the fields used here. */
export interface PdfChar {
  c: string;
  rect: ArrayLike<number>;
  inlineRect?: ArrayLike<number>;
  rotation?: number;
  lineBreakAfter?: boolean;
  ignorable?: boolean;
}

/** Letters and digits only, lowercased, ligatures and accents folded: spacing, hyphenation, punctuation
 *  and quote styles never decide a match, so text copied from another extractor still finds its place. */
export const fold = (s: string): string => s.normalize("NFKD").toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");

/** Under this many folded characters (about three words) a match is not specific enough to show. */
const MIN = 16;
/** The fallbacks when the whole quote is not on the page: its first, then its last, this many words. */
const PART = 8;

/** The quote's [first, last] char index on the page (inclusive), or null. */
export function findQuote(chars: ArrayLike<PdfChar>, quote: string): [number, number] | null {
  let text = "";
  const at: number[] = []; // folded code unit -> char index
  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i]!;
    if (ch.ignorable) continue;
    const f = fold(ch.c ?? "");
    text += f;
    for (let k = 0; k < f.length; k++) at.push(i);
  }
  const words = quote.split(/\s+/).filter((w) => fold(w));
  const tried = new Set<string>();
  for (const part of [words, words.slice(0, PART), words.slice(-PART)]) {
    const needle = fold(part.join(""));
    if (needle.length < MIN || tried.has(needle)) continue;
    tried.add(needle);
    const pos = text.indexOf(needle);
    if (pos >= 0) return [at[pos]!, at[pos + needle.length - 1]!];
  }
  return null;
}

const norm = (r: ArrayLike<number>): number[] =>
  [Math.min(r[0]!, r[2]!), Math.min(r[1]!, r[3]!), Math.max(r[0]!, r[2]!), Math.max(r[1]!, r[3]!)];

/** One rect per line from char `start` to `end`, built the way the reader builds a highlight annotation's
 *  rects from a selection (its getRangeRects): the line's box across, the run's first to last char along. */
export function rangeRects(chars: ArrayLike<PdfChar>, start: number, end: number): number[][] {
  const rects: number[][] = [];
  let from = start;
  for (let i = start; i <= end; i++) {
    const ch = chars[i]!;
    if (!ch.lineBreakAfter && i !== end) continue;
    const first = chars[from]!;
    const a = norm(first.rect), b = norm(ch.rect), line = norm(first.inlineRect ?? first.rect);
    const vertical = first.rotation === 90 || first.rotation === 270;
    rects.push(vertical ? [line[0]!, a[1]!, line[2]!, b[3]!] : [a[0]!, line[1]!, b[2]!, line[3]!]);
    from = i + 1;
  }
  return rects;
}
