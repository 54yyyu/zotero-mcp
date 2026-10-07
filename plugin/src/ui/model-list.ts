// The model dropdown's list as data: the order the rows show in, what folds under More models, provider groups, and the
// search. Pure (no DOM), so the rules are unit-tested and a keystroke over 400+ models costs one scan of short strings.
//
// A short catalog (Claude's 11, Codex's 4) keeps the bridge's order: the first four, then More models; the current model
// and the agent's default are listed without expanding. A long one (pi lists every OpenRouter model) gets a search field
// instead of the fold: the current model and the default first, then each provider's models (the id before the first
// "/"), smaller providers first, since a provider the user configured lists a handful and a cloud aggregator hundreds.
import type { ModelOption } from "../types.ts";

/** Rows listed before "More models" in a short catalog. */
export const TOP = 4;
/** A catalog longer than this has a search field instead of the fold. */
export const SHORT = 12;
/** Rows drawn at once in a long list; the rest wait behind "Show all" (a search narrows them first). */
export const LIMIT = 80;

export interface Row {
  m: ModelOption;
  /** The name as shown: without "provider/" when a provider heading (or the note) already says it. */
  label: string;
  /** The quiet line under the label: the description, and the provider of a row pinned above the groups. */
  note?: string;
  provider: string;
  /** The current model or the default, listed first in a long list. */
  pinned: boolean;
  /** Shown before More models (short catalogs). */
  top: boolean;
  /** What the search matches: name, id and description, lower case. */
  hay: string;
}

export interface ModelList {
  rows: Row[];
  /** Long: a search field and one scrolling list, no fold. */
  long: boolean;
  /** Rows sit under provider headings (a long list with more than one provider). */
  grouped: boolean;
}

/** "openrouter/moonshotai/kimi-k2.6" -> "openrouter"; "" when the id has no provider part. */
export const providerOf = (id: string): string => (id.includes("/") ? id.slice(0, id.indexOf("/")) : "");

/** The name without its provider prefix: pi names a model "openrouter/MoonshotAI: Kimi K2.6". */
export function shortName(m: ModelOption): string {
  const p = providerOf(m.id);
  return p && m.name.startsWith(`${p}/`) ? m.name.slice(p.length + 1) : m.name;
}

/** The rows in display order. `current`: the model in use; `def`: the one the agent itself starts on. */
export function arrange(models: ModelOption[], current?: string, def?: string): ModelList {
  const long = models.length > SHORT;
  const size = new Map<string, number>();
  for (const m of models) size.set(providerOf(m.id), (size.get(providerOf(m.id)) ?? 0) + 1);
  const grouped = long && size.size > 1;
  const pin = long ? [current, def].filter((id, i, a): id is string => !!id && a.indexOf(id) === i && models.some((m) => m.id === id)) : [];
  const first = new Map<string, number>();
  models.forEach((m, i) => { if (!first.has(providerOf(m.id))) first.set(providerOf(m.id), i); });
  const rest = models.filter((m) => !pin.includes(m.id));
  if (grouped) {
    const rank = (p: string) => size.get(p)! * models.length + first.get(p)!;
    rest.sort((a, b) => rank(providerOf(a.id)) - rank(providerOf(b.id)));
  }
  const row = (m: ModelOption, i: number, pinned: boolean): Row => {
    const provider = providerOf(m.id);
    const note = pinned && grouped ? [provider, m.description].filter(Boolean).join(" · ") : m.description;
    return {
      m, provider, pinned, label: grouped ? shortName(m) : m.name, ...(note ? { note } : {}),
      top: !long && (i < TOP || m.id === current || m.id === def),
      hay: `${m.name} ${m.id} ${m.description ?? ""}`.toLowerCase(),
    };
  };
  const pinned = pin.map((id) => row(models.find((m) => m.id === id)!, 0, true));
  // (a short list pins nothing and is never sorted, so `i` is the catalog index there)
  return { long, grouped, rows: [...pinned, ...rest.map((m, i) => row(m, i, false))] };
}

/** The rows every word of `query` is found in (name, id or description, any case), in display order; all for a blank query. */
export function search(list: ModelList, query: string): Row[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return words.length ? list.rows.filter((r) => words.every((w) => r.hay.includes(w))) : list.rows;
}

/** The rows with a provider heading wherever a group starts (pinned rows carry their provider in the note instead). */
export function withHeadings(rows: Row[], grouped: boolean): (Row | { heading: string })[] {
  if (!grouped) return rows;
  const out: (Row | { heading: string })[] = [];
  let prev: Row | undefined;
  for (const r of rows) {
    if (!r.pinned && (!prev || prev.pinned || prev.provider !== r.provider)) out.push({ heading: r.provider || "Other" });
    out.push(r);
    prev = r;
  }
  return out;
}

export const countLabel = (n: number): string => (n === 1 ? "1 model" : `${n} models`);
