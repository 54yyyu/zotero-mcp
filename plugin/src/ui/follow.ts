/** Whether the conversation follows its bottom as an answer streams in.
 *
 *  Following lets go at the first sign the reader moved up — a wheel or
 *  key or touch that means "up", or any scroll that moved up without the
 *  content shrinking under it (a scrollbar drag) — and takes hold again only
 *  when the reader comes back *down* to the bottom, or presses the jump pill.
 *
 *  The old rule ("within 48 px of the bottom = following") fought the reader:
 *  a trackpad scrolls a few pixels per event, so the first dozen events of
 *  every upward scroll landed inside the 48 px, the scroll handler took hold
 *  again, and the next streamed token pinned the feed back down — the feed
 *  shook in place and only a flick that cleared 48 px in one event got away.
 *  Pure, so the rule is tested without a browser (`test/ui/follow.test.ts`). */

export interface Follow {
  /** Pinned to the bottom: new content scrolls the feed. */
  on: boolean;
  /** Where the last scroll (ours or the reader's) left the feed. */
  top: number;
  /** The scrollable height then: a smaller one means content shrank. */
  height: number;
}

export interface Metrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

/** How close to the bottom counts as "back at the bottom" (sub-pixel scroll
 *  positions under zoom never land on exactly 0). Only a downward scroll
 *  reaching it takes hold again, so it cannot trap an upward one. */
export const AT_BOTTOM_PX = 4;

export const START: Follow = { on: true, top: 0, height: 0 };

const distance = (m: Metrics) => m.scrollHeight - m.scrollTop - m.clientHeight;

/** The reader asked to go up (wheel up, PageUp/Home/ArrowUp, a finger
 *  dragging down). It comes before the scroll it causes, which is the point:
 *  a token rendered in between must not pin the feed back down. Nothing to
 *  let go of when the feed is already at its top. */
export function intentUp(f: Follow, m: Metrics): Follow {
  return f.on && m.scrollTop > 0 ? { ...f, on: false } : f;
}

/** A scroll event, whoever caused it. */
export function scrolled(f: Follow, m: Metrics): Follow {
  const up = m.scrollTop < f.top - 0.5;
  const down = m.scrollTop > f.top + 0.5;
  // Content shrinking clamps scrollTop upward; that is not the reader.
  const shrank = m.scrollHeight < f.height - 0.5;
  let on = f.on;
  if (up && !shrank) on = false;
  else if (down && distance(m) <= AT_BOTTOM_PX) on = true;
  return { on, top: m.scrollTop, height: m.scrollHeight };
}

/** We pinned the feed to the bottom (or the jump pill asked us to). */
export function pinned(m: Metrics): Follow {
  return { on: true, top: m.scrollTop, height: m.scrollHeight };
}

/** Keys that scroll a focused scroller up. */
export function isUpKey(e: { key: string; shiftKey?: boolean }): boolean {
  return e.key === "PageUp" || e.key === "Home" || e.key === "ArrowUp" || (e.key === " " && !!e.shiftKey);
}
