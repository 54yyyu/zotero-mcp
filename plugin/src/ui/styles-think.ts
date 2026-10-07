// The turn's one indicator (parts.ts dots/glint; on messages.ts's working line, or in the thought row while a thought
// streams; states from think.ts): a 3x3 dot matrix in the accent, one pattern per state, and a label with a soft glint
// running across it. Every keyframe moves only opacity or transform, and the indicator exists only while a turn runs.
// test/ui/think.test.ts fails if a keyframe here animates anything else.
export const THINK_STYLES = `
.zmc .working { display: flex; align-items: center; gap: var(--s2); height: var(--h-sm); color: var(--ink-muted); }
.zmc .parts:not(:empty) + .working { margin-top: var(--s2); }
.zmc .working--pending { margin-top: calc(-1 * var(--s2)); }
.zmc .working__t { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }

/* ---------- the dots: each one's place in every pattern (w diagonal wave, r round the ring, c column, o rings out
   from the centre, p spiral); the state picks one, a time per step (--g) and a cycle (--t) ---------- */
.zmc .dm { display: grid; grid-template-columns: repeat(3, 4px); grid-auto-rows: 4px; gap: 2.5px; flex: none; }
.zmc .dm i {
  width: 4px; height: 4px; border-radius: 50%; background: var(--agent); opacity: 0.18; transform: scale(0.7);
  --d: var(--w); --g: 110ms; --t: 1.6s;
  /* started one cycle back, so every dot is mid-pattern from the first frame */
  animation: zmc-dot var(--t) ease-in-out calc(var(--d) * var(--g) - var(--t)) infinite;
}
.zmc .dm i:nth-child(1) { --w: 0; --r: 0; --c: 0; --o: 1; --p: 0; }
.zmc .dm i:nth-child(2) { --w: 1; --r: 1; --c: 1; --o: 1; --p: 1; }
.zmc .dm i:nth-child(3) { --w: 2; --r: 2; --c: 2; --o: 1; --p: 2; }
.zmc .dm i:nth-child(4) { --w: 1; --r: 7; --c: 0; --o: 1; --p: 7; }
.zmc .dm i:nth-child(5) { --w: 2; --r: 0; --c: 1; --o: 0; --p: 8; }
.zmc .dm i:nth-child(6) { --w: 3; --r: 3; --c: 2; --o: 1; --p: 3; }
.zmc .dm i:nth-child(7) { --w: 2; --r: 6; --c: 0; --o: 1; --p: 6; }
.zmc .dm i:nth-child(8) { --w: 3; --r: 5; --c: 1; --o: 1; --p: 5; }
.zmc .dm i:nth-child(9) { --w: 4; --r: 4; --c: 2; --o: 1; --p: 4; }
.zmc .working[data-s="searching"] .dm i { --d: var(--r); --g: 125ms; --t: 1s; }
.zmc .working[data-s="searching"] .dm i:nth-child(5) { animation: none; opacity: 0.3; }
.zmc .working[data-s="reading"] .dm i { --d: var(--c); --g: 170ms; --t: 1.2s; }
.zmc .working[data-s="writing"] .dm i { --d: var(--o); --g: 150ms; --t: 1.3s; }
.zmc .working[data-s="working"] .dm i { --d: var(--p); --g: 111ms; --t: 1s; }
.zmc .working[data-s="waiting"] .dm i { opacity: 0.9; transform: none; animation: zmc-pulse 1.6s ease-in-out infinite; }

/* ---------- the label: a glint runs across it letter by letter (each letter's opacity, staggered by --st). Measured in
   Zotero: a background-position shimmer, or a sliding window with a mask, cost 130-300 ms of main-thread time a second;
   this costs what any animation costs there (about 15 ms a second). Waiting is said plainly. ---------- */
.zmc :is(.working, .thought--active) .glint > span { opacity: 0.62; color: var(--ink); }
.zmc .working:not([data-s="waiting"]) .glint > span, .zmc .thought--active .glint > span { animation: zmc-glint 2.1s ease-in-out calc(var(--i) * var(--st) - 2.1s) infinite; }
.zmc .working[data-s="waiting"] .glint > span { opacity: 1; }
.zmc .thought__mark:empty { display: none; }

@keyframes zmc-dot { 0%, 100% { opacity: 0.18; transform: scale(0.7); } 35% { opacity: 1; transform: scale(1.12); } }
@keyframes zmc-glint { 0%, 24%, 100% { opacity: 0.62; } 12% { opacity: 1; } }
@media (prefers-reduced-motion: reduce) {
  .zmc .dm i { animation: none !important; opacity: 0.6 !important; transform: none; }
  .zmc :is(.working, .thought--active) .glint > span { animation: none !important; opacity: 1 !important; color: var(--ink-muted); }
}
`;
