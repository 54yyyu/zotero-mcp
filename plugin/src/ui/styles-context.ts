// Context economy in the composer: the quiet fill meter and the long-chat suggestion. Same tokens as styles.ts.
export const CONTEXT_STYLES = `
.zmc .cmeter { flex: none; display: grid; place-items: center; width: var(--h-sm); height: var(--h-sm); color: var(--ink-muted); cursor: default; }
.zmc .cmeter[hidden], .zmc .cnote[hidden] { display: none; }
.zmc .cmeter svg { width: 16px; height: 16px; transform: rotate(-90deg); }
.zmc .cmeter circle { fill: none; stroke-width: 2; }
.zmc .cmeter__track { stroke: var(--rule); }
.zmc .cmeter__arc { stroke: currentColor; stroke-linecap: round; }
.zmc .cmeter[data-level="warm"] { color: var(--warn); }
.zmc .cmeter[data-level="full"] { color: var(--danger); }
.zmc .cnote { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s2); margin-bottom: var(--s2); padding: var(--s1) var(--s1) var(--s1) var(--s3); border-radius: var(--r1); background: var(--tint-hover); color: var(--ink-muted); font-size: var(--fs-2); }
.zmc .cnote__t { flex: 1 1 12em; min-width: 0; }
.zmc .cnote .lnk { color: var(--ink); }
@media (prefers-reduced-motion: no-preference) { .zmc .cmeter__arc { transition: stroke-dasharray 400ms ease, color 200ms ease; } }
`;
