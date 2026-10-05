// Context economy in the composer: the quiet fill meter and the long-chat suggestion. Same tokens as styles.ts.
export const CONTEXT_STYLES = `
.zmc .cmeter { flex: none; display: inline-flex; align-items: center; gap: 3px; height: var(--h-sm); padding: 0 var(--s1); color: var(--ink-faint); font-size: var(--fs-1); font-variant-numeric: tabular-nums; cursor: default; }
.zmc .cmeter[hidden], .zmc .cnote[hidden] { display: none; }
.zmc .cmeter svg { width: 14px; height: 14px; transform: rotate(-90deg); }
.zmc .cmeter circle { fill: none; stroke-width: 2.5; }
.zmc .cmeter__track { stroke: var(--rule); }
.zmc .cmeter__arc { stroke: currentColor; stroke-linecap: round; }
.zmc .cmeter--full { color: var(--warn); }
.zmc .cnote { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s2); margin-bottom: var(--s2); padding: var(--s1) var(--s1) var(--s1) var(--s3); border-radius: var(--r1); background: var(--tint-hover); color: var(--ink-muted); font-size: var(--fs-2); }
.zmc .cnote__t { flex: 1 1 12em; min-width: 0; }
.zmc .cnote .lnk { color: var(--ink); }
@container zmc (max-width: 340px) { .zmc .cmeter__pct { display: none; } }
@media (prefers-reduced-motion: no-preference) { .zmc .cmeter__arc { transition: stroke-dasharray 300ms ease; } }
`;
