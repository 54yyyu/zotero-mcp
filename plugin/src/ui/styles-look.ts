// The look (appearance.ts sets the data attributes and variables read here): the accent, the glass, the backdrop, text
// size and density, and the settings screen's cards. Glass is faux glass inside our own root (Gecko gives a plugin no
// OS vibrancy): translucent cards with a bright edge and a soft shadow over the backdrop (.zmc::before: plain paper by
// default, or the glow, a gradient or the picture; ::after the picture's veil), and a real backdrop blur on the three
// floating surfaces only (composer, menus, the jump pill), so a long chat scrolls as cheaply as without it.
export const LOOK_STYLES = `
/* ---------- the accent: mono ink by default; --agent and --focus follow it, links only once the user chose a colour
   (until then they stay the calm link blue of styles.ts); the logo's Z keeps --brand red ---------- */
.zmc {
  --accent: var(--accent-l, #16181d); --accent-ink: var(--accent-ink-l, #16181d); --on-accent: var(--on-accent-l, #ffffff);
  --brand: #cc2936; --agent: var(--accent); --focus: var(--accent-ink);
}
.zmc[data-theme="dark"] {
  --accent: var(--accent-d, #e6e8eb); --accent-ink: var(--accent-ink-d, #e6e8eb); --on-accent: var(--on-accent-d, #16181d);
  --brand: #ff7b86; --agent: var(--accent); --focus: var(--accent-ink);
}
.zmc[data-accent="custom"] { --link: var(--accent-ink); }
.zmc[data-glass] .send:not(:disabled):not(.send--stop) { background: var(--accent); color: var(--on-accent); transition: filter var(--ease), transform var(--ease); }
.zmc[data-glass] .send:hover:not(:disabled):not(.send--stop) { filter: brightness(1.08); }
.zmc[data-glass] .ubub { background: color-mix(in srgb, var(--accent) 7%, var(--paper-raised)); border-color: color-mix(in srgb, var(--accent) 22%, var(--rule)); }
.zmc[data-glass] :is(.radio--on .radio__dot, .wcard--on .wcard__dot) { border-color: var(--accent); background: var(--accent); }
.zmc[data-glass] :is(.wcard--on, .wcard--on:hover) { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.zmc[data-glass] .switch input:checked + .switch__track { background: var(--accent); border-color: var(--accent); }
.zmc[data-glass] .switch input:checked + .switch__track .switch__thumb { background: var(--on-accent); }
.zmc[data-glass] .eff__fill { background: var(--accent); }
.zmc[data-glass] .eff__stop--on::before { border-color: var(--accent); background: var(--accent); }
.zmc[data-glass] .menu__check { color: var(--accent-ink); }
.zmc[data-glass] .input:focus { border-color: color-mix(in srgb, var(--accent) 55%, var(--rule-strong)); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 16%, transparent); }

/* ---------- text size and density ---------- */
.zmc[data-size="small"] { --fs-1: 10.5px; --fs-2: 11.5px; --fs-3: 12px; --fs-4: 13px; --fs-5: 15px; --fs-6: 18px; }
.zmc[data-size="large"] { --fs-1: 12px; --fs-2: 13px; --fs-3: 14px; --fs-4: 15.5px; --fs-5: 17px; --fs-6: 22px; }
.zmc[data-density="compact"] { --s3: 10px; --s4: 12px; --s5: 18px; --h-row: 32px; }

/* ---------- the backdrop ---------- */
.zmc { isolation: isolate; }
.zmc::before, .zmc::after { content: ""; position: absolute; inset: 0; z-index: -1; display: none; pointer-events: none; }
.zmc:not([data-bg="none"])::before { display: block; }
.zmc[data-bg="image"]::before { inset: calc(var(--bg-blur, 0px) * -3); background: var(--bg-img, none) center / cover no-repeat; filter: blur(var(--bg-blur, 0px)); }
.zmc[data-bg="mist"]::before { background: radial-gradient(90% 55% at 90% 0%, #dbe6f3, transparent 70%), linear-gradient(170deg, #eef2f6, #e4e9f0 60%, #eceaf2); }
.zmc[data-bg="dawn"]::before { background: radial-gradient(90% 50% at 100% 0%, #fbe0d0, transparent 70%), radial-gradient(80% 50% at 0% 100%, #e5e0f7, transparent 70%), linear-gradient(170deg, #fbf1ea, #f5e9ee 55%, #ecebf7); }
.zmc[data-bg="sage"]::before { background: radial-gradient(90% 55% at 0% 0%, #dcebdf, transparent 70%), linear-gradient(170deg, #eff4ef, #e5ede7 60%, #edf1ec); }
.zmc[data-bg="dusk"]::before { background: radial-gradient(90% 55% at 100% 0%, #dcdcf6, transparent 70%), radial-gradient(80% 50% at 0% 100%, #f0dceb, transparent 70%), linear-gradient(170deg, #eceef8, #e5e5f3 55%, #efe7f2); }
.zmc[data-bg="sand"]::before { background: radial-gradient(90% 55% at 100% 0%, #f6e5cc, transparent 70%), linear-gradient(170deg, #f8f3ea, #f1e9dc 60%, #f4eee5); }
.zmc[data-theme="dark"][data-bg="mist"]::before { background: radial-gradient(90% 55% at 90% 0%, #1d2a3a, transparent 70%), linear-gradient(170deg, #151a21, #131820 60%, #17161f); }
.zmc[data-theme="dark"][data-bg="dawn"]::before { background: radial-gradient(90% 50% at 100% 0%, #3a2420, transparent 70%), radial-gradient(80% 50% at 0% 100%, #241f3a, transparent 70%), linear-gradient(170deg, #1b1617, #1b161c 55%, #17161f); }
.zmc[data-theme="dark"][data-bg="sage"]::before { background: radial-gradient(90% 55% at 0% 0%, #1b2c22, transparent 70%), linear-gradient(170deg, #131815, #141a17 60%, #151917); }
.zmc[data-theme="dark"][data-bg="dusk"]::before { background: radial-gradient(90% 55% at 100% 0%, #25244a, transparent 70%), radial-gradient(80% 50% at 0% 100%, #33203a, transparent 70%), linear-gradient(170deg, #141526, #16152a 55%, #1b1626); }
.zmc[data-theme="dark"][data-bg="sand"]::before { background: radial-gradient(90% 55% at 100% 0%, #362b1c, transparent 70%), linear-gradient(170deg, #1a1814, #1c1915 60%, #191714); }
/* Plain ("none", the default) is the paper alone. Glow is three faint washes of colour; the veil fades a picture into the panel */
.zmc { --glow: 14%; --glow-a: #7c9cc9; } /* mono: a cool blue-grey wash; a chosen accent glows in its own colour */
.zmc[data-accent="custom"] { --glow-a: var(--accent); }
.zmc[data-theme="dark"] { --glow: 20%; }
.zmc {
  --glows: radial-gradient(70% 40% at 0% 0%, color-mix(in srgb, var(--glow-a) var(--glow), transparent), transparent 72%),
    radial-gradient(60% 36% at 100% 45%, color-mix(in srgb, #3bb4c9 var(--glow), transparent), transparent 72%),
    radial-gradient(75% 38% at 15% 100%, color-mix(in srgb, #8b7cf6 var(--glow), transparent), transparent 72%);
}
.zmc[data-bg="glow"]::before { background: var(--glows); }
.zmc[data-bg="image"]::after { display: block; background: color-mix(in srgb, var(--paper) var(--bg-veil, 65%), transparent); }
/* over a picture, text drawn straight on it gets a soft halo of the paper colour so it keeps its contrast */
.zmc[data-bg="image"] :is(.md, .empty, .vempty, .hist__group, .wel__hero, .working, .foot) { text-shadow: 0 0 4px var(--paper); }
.zmc:not([data-bg="none"]) .hist__search, .zmc[data-glass="on"] .hist__search { background: transparent; }

/* ---------- glass ---------- */
.zmc[data-glass="on"] {
  --glass: color-mix(in srgb, #ffffff 60%, transparent);
  --glass-strong: color-mix(in srgb, #ffffff 84%, transparent);
  --glass-edge: color-mix(in srgb, #ffffff 85%, transparent);
  --glass-line: rgb(30 36 56 / 0.1);
  --glass-sheen: inset 0 1px 0 rgb(255 255 255 / 0.9);
  --glass-lift: 0 4px 14px -8px rgb(24 32 64 / 0.22);
  --glass-shadow: 0 12px 30px -14px rgb(24 32 64 / 0.32), 0 1px 3px rgb(24 32 64 / 0.07);
}
.zmc[data-glass="on"][data-theme="dark"] {
  --glass: color-mix(in srgb, #3a3f49 46%, transparent);
  --glass-strong: color-mix(in srgb, #23262d 88%, transparent);
  --glass-edge: rgb(255 255 255 / 0.13);
  --glass-line: rgb(255 255 255 / 0.09);
  --glass-sheen: inset 0 1px 0 rgb(255 255 255 / 0.07);
  --glass-lift: 0 4px 14px -8px rgb(0 0 0 / 0.5);
  --glass-shadow: 0 14px 32px -14px rgb(0 0 0 / 0.7), 0 1px 3px rgb(0 0 0 / 0.35);
}
.zmc[data-glass="on"] .hd { border-bottom-color: var(--glass-line); }
/* cards: translucent, a bright top edge, a soft shadow */
.zmc[data-glass="on"] :is(.sec, .plan, .wcard, .pe, .check--ok, .prompts__list, .code, .md-table, .area, .stat, .chip:not(.chip--sm), kbd) { background: var(--glass); border-color: var(--glass-line); }
.zmc[data-glass="on"] :is(.sec, .plan, .wcard, .pe, .check--ok, .prompts__list) { box-shadow: var(--glass-sheen), var(--glass-lift); }
.zmc[data-glass="on"] :is(.code, .md-table) { box-shadow: var(--glass-sheen); }
.zmc[data-glass="on"] .md-table th { background: color-mix(in srgb, var(--ink) 4%, transparent); }
.zmc[data-glass="on"] .code__head { background: color-mix(in srgb, var(--ink) 3%, transparent); border-bottom-color: var(--glass-line); }
/* the cards that carry a coloured edge keep it */
.zmc[data-glass="on"] :is(.setup, .check--bad, .notice, .perm:not(.perm--done), .inlineerr, .wrow--bad) { background: var(--glass); border-color: var(--glass-line); }
.zmc[data-glass="on"] .ubub { background: color-mix(in srgb, var(--accent) 8%, var(--glass)); border-color: color-mix(in srgb, var(--accent) 24%, var(--glass-line)); box-shadow: var(--glass-sheen); }
/* mono: a quiet grey bubble, no ink-coloured edge */
.zmc[data-glass]:not([data-accent]) .ubub { background: color-mix(in srgb, var(--ink) 4%, var(--paper-raised)); border-color: var(--rule); }
.zmc[data-glass="on"]:not([data-accent]) .ubub { background: color-mix(in srgb, var(--ink) 4%, var(--glass)); border-color: var(--glass-line); }
.zmc[data-glass="on"] :is(.input, .folder) { background: var(--glass-strong); border-color: var(--glass-line); }
.zmc[data-glass="on"] .input:hover:not(:disabled):not(:focus) { border-color: var(--rule-strong); }
.zmc[data-glass="on"] .seg { background: color-mix(in srgb, var(--ink) 5%, transparent); border-color: var(--glass-line); }
.zmc[data-glass="on"] .seg__opt--on { background: var(--glass-strong); box-shadow: var(--glass-sheen), 0 1px 3px rgb(0 0 0 / 0.12); }
.zmc[data-glass="on"][data-theme="dark"] .seg__opt--on { background: rgb(255 255 255 / 0.12); }
.zmc[data-glass="on"] .mdd__agents { background: color-mix(in srgb, var(--ink) 6%, transparent); }
.zmc[data-glass="on"] .mdd__agent--on { background: var(--paper-raised); box-shadow: var(--glass-sheen), 0 1px 3px rgb(0 0 0 / 0.12); }
.zmc[data-glass="on"][data-theme="dark"] .mdd__agent--on { background: rgb(255 255 255 / 0.13); }
.zmc[data-glass="on"] .mdd__eff { border-top-color: var(--glass-line); }
.zmc[data-glass="on"] .steps { background: color-mix(in srgb, var(--ink) 4%, transparent); }
/* floating surfaces: frosted (the only backdrop blur in the panel) */
.zmc[data-glass="on"] :is(.composer, .menu, .pop, .jump, .ctip) { border-color: var(--glass-edge); box-shadow: var(--glass-sheen), var(--glass-shadow); }
/* the composer's frost is a layer under it, not the composer itself: an element with a backdrop filter is the backdrop
   root of everything inside it, so the menus and the @ popup (its children) would only blur the composer */
.zmc[data-glass="on"] .composer { background: transparent; isolation: isolate; transition: border-color var(--ease), box-shadow var(--ease); }
.zmc[data-glass="on"] .composer::before { content: ""; position: absolute; inset: 0; z-index: -1; border-radius: inherit; background: var(--glass); backdrop-filter: blur(18px) saturate(170%); pointer-events: none; }
.zmc[data-glass="on"] .composer:focus-within { border-color: color-mix(in srgb, var(--ink) 22%, var(--glass-edge)); }
.zmc[data-glass="on"][data-theme="light"] .composer { border-color: color-mix(in srgb, #ffffff 70%, var(--glass-line)); }
.zmc[data-glass="on"] :is(.menu, .pop, .jump, .ctip) { background: var(--glass-strong); backdrop-filter: blur(27px) saturate(180%); }
.zmc[data-glass="on"] :is(.menu__item:hover, .menu__item:focus-visible, .pop__i--on) { background: color-mix(in srgb, var(--accent) 11%, transparent); }
@supports not (backdrop-filter: blur(1px)) {
  .zmc[data-glass="on"] .composer::before { background: var(--glass-strong); }
  .zmc[data-glass="on"] :is(.menu, .pop, .jump, .ctip) { background: var(--paper-raised); }
}
/* the send button: a small gem of the accent */
.zmc[data-glass="on"] .send:not(:disabled):not(.send--stop) {
  background: linear-gradient(145deg, color-mix(in srgb, var(--accent) 80%, #ffffff), var(--accent) 52%, color-mix(in srgb, var(--accent) 82%, var(--gem-tint, #5b3df5)));
  box-shadow: inset 0 1px 0 rgb(255 255 255 / 0.35), 0 4px 12px -3px color-mix(in srgb, var(--accent) 60%, transparent);
}
.zmc:not([data-accent]) { --gem-tint: var(--accent); } /* mono: a plain ink gem, no violet */
.zmc[data-glass="on"] .send:hover:not(:disabled):not(.send--stop) { transform: translateY(-1px); }
.zmc[data-glass="on"] .send:disabled { background: color-mix(in srgb, var(--ink) 7%, transparent); }
/* pills that lift a pixel */
.zmc[data-glass="on"] .prompts__list { padding: var(--s1); border: 1px solid var(--glass-line); border-radius: var(--r2); }
.zmc[data-glass="on"] .prompt { transition: background var(--ease), transform var(--ease); }
.zmc[data-glass="on"] .prompt:hover:not(:disabled) { background: color-mix(in srgb, var(--accent) 8%, transparent); }
.zmc[data-glass="on"] .prompt__k { background: color-mix(in srgb, var(--ink) 6%, transparent); }
.zmc[data-glass="on"] .wcard:hover:not(.wcard--on) { transform: translateY(-1px); box-shadow: var(--glass-sheen), var(--glass-shadow); }
@media (prefers-reduced-motion: reduce) { .zmc[data-glass="on"] :is(.wcard, .send):hover { transform: none !important; } }

/* ---------- the settings screen: stacked cards, each a title, a line on what it is for, then its rows ---------- */
.zmc .set { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s3); padding-top: var(--s2); }
.zmc .sec { min-width: 0; border: 1px solid var(--rule); border-radius: 12px; background: var(--paper-raised); }
.zmc .sec__head { padding: var(--s3) var(--s4) 0; }
.zmc .sec__t { font-size: var(--fs-4); line-height: 1.35; }
.zmc .sec__d { margin: 2px 0 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.45; }
.zmc .sec__body { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s3); min-width: 0; padding: var(--s3) var(--s4) var(--s4); }
.zmc .sec__body > .swrow, .zmc .sec__body > .row { margin: calc(-1 * var(--s3) / 2) 0; }
.zmc .sec__body > :is(.swrow, .row) + :is(.swrow, .row) { border-top: 1px solid var(--rule); }
.zmc[data-glass="on"] .sec__body > :is(.swrow, .row) + :is(.swrow, .row) { border-top-color: var(--glass-line); }
.zmc .sec__body > .fields { margin: 0; padding-top: var(--s3); border-top: 1px solid var(--rule); }
.zmc[data-glass="on"] .sec__body > .fields { border-top-color: var(--glass-line); }
.zmc .sec__body > .about { margin: 0; }
.zmc .sec__hint { margin: 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.5; overflow-wrap: anywhere; }
.zmc .sec__sub { display: grid; gap: var(--s2); padding-top: var(--s3); border-top: 1px solid var(--rule); }
.zmc[data-glass="on"] .sec__sub { border-top-color: var(--glass-line); }
.zmc .sec__subt { font-weight: 500; }
.zmc .row { display: grid; gap: var(--s1); padding: var(--s2) 0; min-width: 0; }
.zmc .row__main { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--s2) var(--s3); min-width: 0; }
.zmc .row__l { font-weight: 500; }
.zmc .row__v { margin-left: auto; font-size: var(--fs-2); color: var(--ink-muted); font-variant-numeric: tabular-nums; }
.zmc .row__ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); min-width: 0; }
.zmc .row__hint { margin: 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.45; }
.zmc .row input[type="range"] { width: 100%; margin: var(--s1) 0 0; accent-color: var(--accent); cursor: pointer; }
/* swatches: round for colours, small tiles for backgrounds */
.zmc .sws { display: flex; flex-wrap: wrap; gap: var(--s2); }
.zmc .sw { --c: var(--cl); position: relative; flex: none; width: 24px; height: 24px; padding: 0; border: 0; border-radius: 50%; background: var(--c); box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.12); cursor: pointer; transition: transform var(--ease), box-shadow var(--ease); }
.zmc[data-theme="dark"] .sw { --c: var(--cd); }
.zmc .sw:hover { transform: scale(1.08); }
.zmc :is(.sw[aria-checked="true"], .sw--on) { box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.12), 0 0 0 2px var(--paper-raised), 0 0 0 4px var(--c); }
.zmc .sw.sw--mono { background: linear-gradient(135deg, #16181d 50%, #e6e8eb 50%); }
.zmc .sw--custom { background: conic-gradient(from 180deg, #e64980, #fab005, #40c057, #15aabf, #4c6ef5, #be4bdb, #e64980); }
.zmc .sw--custom::after { content: ""; position: absolute; inset: 5px; border-radius: 50%; background: var(--c, var(--paper-raised)); box-shadow: 0 0 0 1px rgb(0 0 0 / 0.1); }
.zmc .sw--custom.sw--on { --c: var(--accent); }
.zmc .sw--custom input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; padding: 0; border: 0; opacity: 0; cursor: pointer; }
.zmc .sw--custom:focus-within { outline: 2px solid var(--focus); outline-offset: 2px; }
.zmc .hex { width: 6.5rem; font-family: var(--mono); }
.zmc .bgsw { position: relative; flex: none; width: 40px; height: 30px; padding: 0; border: 1px solid var(--rule-strong); border-radius: var(--r1); background: var(--paper); cursor: pointer; overflow: hidden; transition: transform var(--ease), box-shadow var(--ease); }
.zmc .bgsw:hover { transform: translateY(-1px); }
.zmc .bgsw[aria-checked="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.zmc .bgsw--none { background: var(--paper); }
.zmc .bgsw--glow { background: radial-gradient(90% 80% at 0% 0%, rgb(124 156 201 / 0.55), transparent 70%), radial-gradient(80% 70% at 100% 55%, rgb(59 180 201 / 0.45), transparent 70%), radial-gradient(90% 80% at 15% 100%, rgb(139 124 246 / 0.45), transparent 70%), var(--paper); }
.zmc .bgsw--image { background: var(--bg-img, var(--paper-sunk)) center / cover; }
.zmc .bgsw--mist { background: linear-gradient(170deg, #eef2f6, #dbe6f3); }
.zmc .bgsw--dawn { background: linear-gradient(150deg, #fbe0d0, #f5e9ee 50%, #e5e0f7); }
.zmc .bgsw--sage { background: linear-gradient(170deg, #eff4ef, #dcebdf); }
.zmc .bgsw--dusk { background: linear-gradient(150deg, #dcdcf6, #e5e5f3 50%, #f0dceb); }
.zmc .bgsw--sand { background: linear-gradient(170deg, #f8f3ea, #f6e5cc); }
.zmc[data-theme="dark"] .bgsw--mist { background: linear-gradient(170deg, #151a21, #1d2a3a); }
.zmc[data-theme="dark"] .bgsw--dawn { background: linear-gradient(150deg, #3a2420, #1b161c 50%, #241f3a); }
.zmc[data-theme="dark"] .bgsw--sage { background: linear-gradient(170deg, #131815, #1b2c22); }
.zmc[data-theme="dark"] .bgsw--dusk { background: linear-gradient(150deg, #25244a, #16152a 50%, #33203a); }
.zmc[data-theme="dark"] .bgsw--sand { background: linear-gradient(170deg, #1a1814, #362b1c); }
@media (prefers-reduced-motion: reduce) { .zmc :is(.sw, .bgsw):hover { transform: none; } }
`;
