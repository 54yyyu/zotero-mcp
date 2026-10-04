// The panel's stylesheet, injected into the shadow root (chat content is in styles-chat.ts). Parley's tokens
// and shapes, as in meeting-buddy's ui/style.css, on the system font stack so it sits beside Zotero's own
// chrome. One scale for each of: space (4 8 12 16 24), type (11 12 13 14 16 20), radius (4 6 10 16 pill),
// control heights (24 28 32 36), icons (14 16). px, not rem: the host window's root font size must not
// scale it. Container queries, not media queries: the panel's width is not the window's.
import { CHAT_STYLES } from "./styles-chat.ts";
import { WELCOME_STYLES } from "./styles-welcome.ts";

const BASE = `
:host { display: block; height: 100%; min-width: 0; }
.zmc, .zmc *, .zmc *::before, .zmc *::after { box-sizing: border-box; }
.zmc {
  --paper: #f5f6f7; --paper-raised: #ffffff; --paper-sunk: #eceef0;
  --ink: #16181d; --ink-muted: #5f636b; --ink-faint: #80868e;
  --rule: #dfe2e6; --rule-strong: #c9ced4; --focus: #16181d;
  --danger: #c0341c; --warn: #b35c00; --ok: #2f9e44; --agent: #cc2936; --link: #2563c9; --info: #3b5bdb;
  --tint-hover: color-mix(in srgb, var(--ink) 5%, transparent);
  --tint-on: color-mix(in srgb, var(--ink) 9%, transparent);
  --shadow: 0 1px 2px rgb(0 0 0 / 0.06), 0 6px 16px rgb(0 0 0 / 0.1);
  --font: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --fs-1: 11px; --fs-2: 12px; --fs-3: 13px; --fs-4: 14px; --fs-5: 16px; --fs-6: 20px;
  --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px;
  --r0: 4px; --r1: 6px; --r2: 10px; --r3: 16px; --pill: 999px;
  --h-chip: 24px; --h-sm: 28px; --h-md: 32px; --h-row: 36px;
  --ease: 120ms ease;

  position: relative; display: flex; flex-direction: column; height: 100%; width: 100%; min-width: 0; overflow: hidden;
  container-type: inline-size; container-name: zmc;
  background: var(--paper); color: var(--ink); color-scheme: light;
  font: 400 var(--fs-3)/1.5 var(--font); -webkit-font-smoothing: antialiased; text-align: left;
}
.zmc[data-theme="dark"] {
  --paper: #131519; --paper-raised: #1a1d22; --paper-sunk: #0e1013;
  --ink: #e6e8eb; --ink-muted: #a0a5ad; --ink-faint: #7d828a;
  --rule: #2a2e35; --rule-strong: #3b4048; --focus: #e6e8eb;
  --danger: #ff8a7a; --warn: #ffb454; --ok: #51cf66; --agent: #ff7b86; --link: #79a8ff; --info: #748ffc;
  --tint-hover: color-mix(in srgb, var(--ink) 6%, transparent);
  --tint-on: color-mix(in srgb, var(--ink) 11%, transparent);
  --shadow: 0 1px 2px rgb(0 0 0 / 0.5), 0 6px 16px rgb(0 0 0 / 0.4);
  color-scheme: dark;
}
.zmc button, .zmc input, .zmc textarea, .zmc select { font: inherit; color: inherit; }
.zmc button { cursor: pointer; }
.zmc button:disabled, .zmc select:disabled { cursor: default; }
.zmc button:not(:disabled):active { opacity: 0.75; }
.zmc :focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: var(--r0); }
.zmc :is(.steps, .menu, .pop, .code) :focus-visible { outline-offset: -2px; }
.zmc ::placeholder { color: var(--ink-faint); opacity: 1; }
.zmc ::selection { background: color-mix(in srgb, var(--link) 28%, transparent); color: inherit; }
.zmc [hidden] { display: none !important; }
.zmc svg { display: block; flex: none; }
.zmc :is(.hrow__f svg, .stopnote svg) { width: 12px; height: 12px; }
.zmc :is(.btn svg, .lnk svg, .chip__i svg, .chip__btn svg, .area__head svg, .pick svg, .pop__search svg, .pop__ic svg, .menu__check svg, .keyrow__ok svg, .folder svg, .step__icon svg, .step__st svg, .plan__head svg, .perm--done svg) { width: 14px; height: 14px; }
.zmc :is(.iconbtn svg, .send svg, .dropveil svg, .jump svg) { width: 16px; height: 16px; }
.zmc :is(.stat__t, .prompt__t, .chip__t, .hrow__f span, .hrow__t, .folder__p, .source__t, .step__title, .perm--done .perm__what) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.zmc h1, .zmc h2, .zmc h3 { margin: 0; font-weight: 600; }
.zmc .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
.zmc :is(.feed, .vw__body, .pop__list, .menu, pre, .md-table, .cin) { scrollbar-width: thin; scrollbar-color: var(--rule-strong) transparent; }
@keyframes zmc-pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
@keyframes zmc-rise { from { opacity: 0; transform: translateY(4px); } }
@keyframes zmc-fade { from { opacity: 0; } }
@keyframes zmc-caret { 0%, 55% { opacity: 0.9; } 56%, 100% { opacity: 0; } }
@media (prefers-reduced-motion: reduce) {
  .zmc *, .zmc *::before, .zmc *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
  .zmc .pulse, .zmc .thought--active .thought__dot { opacity: 1 !important; box-shadow: 0 0 0 2px color-mix(in srgb, var(--agent) 30%, transparent); }
  .zmc .md--streaming > p:last-child::after { opacity: 0.9 !important; }
}

/* ---------- controls ---------- */
.zmc .btn {
  display: inline-flex; align-items: center; justify-content: center; gap: var(--s2); height: var(--h-md); padding: 0 var(--s4);
  background: transparent; color: var(--ink); border: 1px solid var(--rule-strong); border-radius: var(--pill);
  font-size: var(--fs-3); white-space: nowrap; transition: background var(--ease), border-color var(--ease), color var(--ease);
}

.zmc .btn:hover:not(:disabled) { background: var(--tint-hover); border-color: var(--ink-muted); }
.zmc .btn:disabled { opacity: 0.5; }
.zmc .btn--sm { height: var(--h-sm); padding: 0 var(--s3); font-size: var(--fs-2); }
.zmc .btn--solid { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.zmc .btn--solid:hover:not(:disabled) { background: color-mix(in srgb, var(--ink) 82%, var(--paper)); border-color: transparent; }
.zmc .btn--quiet, .zmc .btn--quiet-danger { border-color: transparent; color: var(--ink-muted); padding-inline: var(--s3); }
.zmc .btn--quiet:hover:not(:disabled) { color: var(--ink); background: var(--tint-hover); border-color: transparent; }
.zmc .btn--quiet-danger:hover:not(:disabled) { color: var(--danger); background: color-mix(in srgb, var(--danger) 9%, transparent); border-color: transparent; }
.zmc .btn--danger { background: var(--danger); border-color: var(--danger); color: var(--paper-raised); }
.zmc .btn--danger:hover:not(:disabled) { border-color: var(--danger); filter: brightness(1.08); }
.zmc[data-theme="dark"] .btn--danger { color: var(--paper-raised); }
.zmc .iconbtn {
  flex: none; width: var(--h-md); height: var(--h-md); display: inline-flex; align-items: center; justify-content: center; padding: 0;
  border: 1px solid transparent; border-radius: var(--r1); background: transparent; color: var(--ink-muted);
  transition: color var(--ease), background var(--ease), border-color var(--ease);
}

.zmc .iconbtn:hover:not(:disabled) { color: var(--ink); background: var(--tint-hover); }
.zmc .iconbtn:disabled { opacity: 0.4; }
.zmc .iconbtn[aria-pressed="true"] { color: var(--ink); background: var(--tint-on); }
.zmc .iconbtn--sm { width: var(--h-sm); height: var(--h-sm); }
.zmc .lnk { display: inline-flex; align-items: center; gap: var(--s1); height: var(--h-sm); padding: 0 var(--s2); border: 0; border-radius: var(--pill); background: none; color: var(--ink-muted); font-size: var(--fs-2); transition: background var(--ease), color var(--ease); }
.zmc .lnk:hover { background: var(--tint-hover); color: var(--ink); }

.zmc .eyebrow, .zmc .sec__t, .zmc .hist__group { font-size: var(--fs-1); font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-muted); }
.zmc kbd { display: inline-block; padding: 0 var(--s2); border: 1px solid var(--rule); border-radius: var(--r0); background: var(--paper-raised); font: var(--fs-1)/18px var(--mono); color: var(--ink-muted); white-space: nowrap; }
.zmc .input { width: 100%; height: var(--h-md); padding: 0 var(--s3); background: var(--paper-raised); border: 1px solid var(--rule-strong); border-radius: var(--r1); font-size: var(--fs-3); transition: border-color var(--ease), box-shadow var(--ease); }
.zmc .input--sm { height: var(--h-sm); padding: 0 var(--s2); font-size: var(--fs-2); }
.zmc .input:hover:not(:disabled) { border-color: var(--ink-faint); }
.zmc .input:focus { border-color: var(--ink-muted); outline: none; box-shadow: 0 0 0 3px color-mix(in srgb, var(--ink) 8%, transparent); }
.zmc textarea.input { height: auto; padding: var(--s2) var(--s3); resize: vertical; line-height: 1.5; }
.zmc .pulse { display: block; width: 6px; height: 6px; border-radius: 50%; background: var(--agent); animation: zmc-pulse 900ms ease-in-out infinite; }
.zmc .sk { height: 44px; margin: var(--s2) 0; border-radius: var(--r1); background: var(--paper-sunk); }
.zmc .seg { display: inline-flex; flex-wrap: wrap; padding: 2px; background: var(--paper-sunk); border: 1px solid var(--rule); border-radius: var(--pill); max-width: 100%; }
.zmc .seg__opt { height: var(--h-sm); padding: 0 var(--s3); background: transparent; border: 0; border-radius: var(--pill); font-size: var(--fs-2); color: var(--ink-muted); transition: background var(--ease), color var(--ease); }
.zmc .seg__opt:hover:not(.seg__opt--on):not(:disabled) { color: var(--ink); background: var(--tint-hover); }
.zmc .seg__opt--on { background: var(--paper-raised); color: var(--ink); box-shadow: 0 1px 2px rgb(0 0 0 / 0.1); }
.zmc[data-theme="dark"] .seg__opt--on { background: var(--rule); }
.zmc .seg__opt:disabled { opacity: 0.5; }

/* ---------- header and frame ---------- */
.zmc .hd { flex: none; display: flex; align-items: center; gap: var(--s1); padding: var(--s2); border-bottom: 1px solid var(--rule); }
.zmc .hd__fill { flex: 1; min-width: var(--s1); }
.zmc .stat { display: inline-flex; align-items: center; gap: var(--s2); min-width: 0; max-width: 62%; height: var(--h-sm); padding: 0 var(--s3) 0 10px; border: 1px solid var(--rule); border-radius: var(--pill); background: var(--paper-raised); color: var(--ink-muted); font-size: var(--fs-2); transition: color var(--ease), border-color var(--ease); }
.zmc .stat:hover, .zmc .stat[aria-pressed="true"] { color: var(--ink); border-color: var(--rule-strong); }
.zmc .stat__t { min-width: 0; }
.zmc .stat__dot { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--ink-faint); }
.zmc .stat[data-state="ok"] .stat__dot { background: var(--ok); }
.zmc .stat[data-state="warn"] .stat__dot { background: var(--warn); }
.zmc .stat[data-state="bad"] .stat__dot { background: var(--danger); }
.zmc .stat[data-state="checking"] .stat__dot { animation: zmc-pulse 900ms ease-in-out infinite; }
.zmc .body, .zmc .chat, .zmc .viewhost { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.zmc .body { position: relative; }
.zmc .dock { flex: none; width: 100%; max-width: 46rem; margin: 0 auto; padding: 0 var(--s4) var(--s3); }

/* ---------- setup card and fix log ---------- */
.zmc .setup { padding: var(--s3) var(--s4); border: 1px solid var(--rule-strong); border-radius: var(--r2); background: var(--paper-raised); box-shadow: inset 2px 0 0 var(--warn); }
.zmc .setupslot:empty { display: none; }
.zmc .setupslot { padding-bottom: var(--s2); }
.zmc .setup__head { display: flex; gap: var(--s3); align-items: flex-start; }
.zmc .setup__i { color: var(--warn); } .zmc .setup__i svg { width: 16px; height: 16px; margin-top: 1px; }
.zmc .setup__tx { min-width: 0; }
.zmc .setup__t { font-weight: 600; line-height: 1.4; }
.zmc .setup__d { font-size: var(--fs-2); color: var(--ink-muted); overflow-wrap: anywhere; }
.zmc .setup__help { margin: var(--s2) 0 0; color: var(--ink-muted); font-size: var(--fs-2); line-height: 1.5; }
.zmc .setup__acts { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s3); }
.zmc .fixlog { margin: var(--s2) 0 0; padding: var(--s2) var(--s3); max-height: 144px; overflow: auto; border-radius: var(--r1); background: var(--paper-sunk); font: var(--fs-1)/1.5 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; }

/* ---------- empty state ---------- */
.zmc .emptywrap { display: flex; flex-direction: column; gap: var(--s4); width: 100%; }
.zmc .empty { text-align: center; color: var(--ink-muted); padding: var(--s2) 0; }
.zmc .mark { width: 40px; height: 40px; margin: 0 auto var(--s3); } .zmc .mark svg { width: 100%; height: 100%; overflow: visible; }
.zmc .empty h2 { font-size: var(--fs-6); line-height: 1.25; letter-spacing: -0.01em; color: var(--ink); margin-bottom: var(--s2); }
.zmc .empty__lead { margin: 0 auto; max-width: 24rem; line-height: 1.5; }
.zmc .prompts { margin: var(--s5) auto 0; max-width: 28rem; text-align: left; }
.zmc .prompts__head { display: flex; align-items: center; justify-content: space-between; padding-left: var(--s3); }
.zmc .prompts__list { list-style: none; margin: var(--s1) 0 0; padding: 0; display: grid; gap: 2px; }
.zmc .prompts__none { margin: var(--s2) var(--s3); font-size: var(--fs-2); color: var(--ink-muted); }
.zmc .prompt { display: flex; align-items: center; justify-content: space-between; gap: var(--s3); width: 100%; height: var(--h-row); padding: 0 var(--s3); border: 0; border-radius: var(--r1); background: none; color: var(--ink); text-align: left; transition: background var(--ease); }
.zmc .prompt:hover:not(:disabled) { background: var(--tint-hover); }
.zmc .prompt:disabled { color: var(--ink-faint); }
.zmc .prompt__t { min-width: 0; }
.zmc .prompt__k { flex: none; border-color: transparent; background: var(--paper-sunk); }
@container zmc (max-width: 340px) { .zmc .prompt__k { display: none; } }

/* ---------- composer ---------- */
.zmc .composer { position: relative; display: flex; flex-direction: column; gap: var(--s2); padding: var(--s3) var(--s2) var(--s2) var(--s3); background: var(--paper-raised); border: 1px solid var(--rule); border-radius: var(--r3); box-shadow: 0 1px 2px rgb(0 0 0 / 0.05); transition: border-color var(--ease); }
.zmc .composer:focus-within { border-color: var(--rule-strong); }
.zmc .cchips { display: flex; flex-wrap: wrap; gap: var(--s1); padding-right: var(--s1); }
.zmc .cchips .chip { height: var(--h-sm); padding-right: 2px; }
.zmc .chip { display: inline-flex; align-items: center; min-width: 0; max-width: 100%; height: var(--h-chip); border: 1px solid var(--rule); border-radius: var(--pill); background: var(--paper-raised); color: var(--ink-muted); font-size: var(--fs-2); }
.zmc .chip--pinned { border-color: var(--rule-strong); }
.zmc .chip__main { display: inline-flex; align-items: center; gap: var(--s1); min-width: 0; height: 100%; padding: 0 var(--s2); border: 0; border-radius: var(--pill); background: none; color: inherit; font-size: inherit; }
.zmc .chip__main:hover { color: var(--ink); }
.zmc .chip__i { flex: none; display: inline-grid; color: var(--ink-faint); }
.zmc .chip__t { min-width: 0; }
.zmc .chip__btn { flex: none; display: grid; place-items: center; width: 24px; height: 24px; padding: 0; border: 0; border-radius: 50%; background: none; color: var(--ink-faint); transition: color var(--ease), background var(--ease); }

.zmc .chip__btn:hover { color: var(--ink); background: var(--tint-on); }
.zmc .chip__pin[aria-pressed="true"] { color: var(--ink); } .zmc .chip__pin[aria-pressed="true"] svg { fill: currentColor; }
.zmc .chip__x { display: none; } .zmc .chip:hover .chip__x, .zmc .chip:focus-within .chip__x, .zmc .chip:not(.chip--auto) .chip__x { display: grid; }
@media (hover: none) { .zmc .chip__x { display: grid; } }
.zmc .cerr { flex: 1 0 100%; font-size: var(--fs-2); color: var(--danger); }
.zmc .area { flex: 1 0 100%; padding: var(--s2) var(--s3); border: 1px solid var(--rule); border-radius: var(--r2); background: var(--paper); }
.zmc .area__head { display: flex; align-items: center; gap: var(--s2); margin-bottom: var(--s2); font-size: var(--fs-2); font-weight: 500; color: var(--ink-muted); }
.zmc .area__img { display: block; max-width: 100%; max-height: 120px; margin: 0 auto; border: 1px solid var(--rule); border-radius: var(--r1); background: #fff; object-fit: contain; }
.zmc .area__none { padding: var(--s4); text-align: center; font-size: var(--fs-2); color: var(--ink-faint); }
.zmc .area__acts { display: flex; justify-content: space-between; margin: var(--s2) calc(-1 * var(--s2)) calc(-1 * var(--s1)); }
.zmc .cin { display: block; width: 100%; min-height: 24px; max-height: 168px; padding: 0; margin: 0; border: 0; background: none; resize: none; outline: none; font-size: var(--fs-4); line-height: 24px; overflow-y: auto; }
.zmc .cin:focus-visible { outline: none; }
.zmc .ctools { display: flex; align-items: center; gap: var(--s1); margin-left: calc(-1 * (var(--h-sm) - 16px) / 2); }
.zmc .ctools__fill { flex: 1; min-width: var(--s1); }
.zmc .pick { display: inline-flex; align-items: center; gap: var(--s1); min-width: 0; height: var(--h-sm); padding: 0 var(--s2); border: 1px solid transparent; border-radius: var(--pill); background: none; color: var(--ink-muted); font-size: var(--fs-2); white-space: nowrap; transition: background var(--ease), color var(--ease), border-color var(--ease); }
.zmc .pick:hover, .zmc .pick[aria-expanded="true"] { background: var(--tint-hover); color: var(--ink); }
.zmc .pick__t { min-width: 0; overflow: hidden; text-overflow: ellipsis; }

.zmc .pick .pick__chev { width: 12px; height: 12px; color: var(--ink-faint); }
.zmc :is(.pick--mode, .pick--effort) { flex: none; }
.zmc .pick--effort { margin-left: calc(-1 * var(--s1)); }
.zmc .pick--mode { border-color: var(--rule); }
.zmc .pick--m-plan, .zmc .pick--m-acceptEdits { color: var(--info); border-color: color-mix(in srgb, var(--info) 40%, transparent); }
.zmc .pick--m-bypassPermissions { color: var(--danger); border-color: color-mix(in srgb, var(--danger) 40%, transparent); }
@container zmc (max-width: 400px) { .zmc .pick--mode .pick__t { display: none; } }
@container zmc (max-width: 340px) { .zmc .pick--model .pick__chev { display: none; } }
.zmc .send { flex: none; width: var(--h-md); height: var(--h-md); display: grid; place-items: center; padding: 0; border: 0; border-radius: 50%; background: var(--ink); color: var(--paper); transition: background var(--ease), color var(--ease); }

.zmc .send:hover:not(:disabled) { background: color-mix(in srgb, var(--ink) 82%, var(--paper)); }
.zmc .send:disabled { background: var(--paper-sunk); color: var(--ink-faint); }
.zmc .send--stop, .zmc .send--stop:hover:not(:disabled) { background: var(--danger); color: var(--paper-raised); }

.zmc .dropveil { position: absolute; inset: 0; z-index: 20; display: none; align-items: center; justify-content: center; gap: var(--s2); border-radius: inherit; background: color-mix(in srgb, var(--paper-raised) 88%, transparent); color: var(--ink); font-weight: 500; pointer-events: none; }

.zmc .composer--drop { border-color: var(--ink); box-shadow: 0 0 0 2px color-mix(in srgb, var(--ink) 25%, transparent); }
.zmc .composer--drop .dropveil { display: flex; }

/* the @ / + search popup and the pickers' menus sit above the composer */
.zmc :is(.pop, .menu) { position: absolute; left: 0; right: 0; bottom: calc(100% + var(--s2)); z-index: 30; background: var(--paper-raised); border: 1px solid var(--rule); border-radius: var(--r2); box-shadow: var(--shadow); animation: zmc-fade 120ms ease; }
.zmc .pop { display: flex; flex-direction: column; max-height: min(320px, 60cqh); overflow: hidden; }
.zmc .pop__search { display: flex; align-items: center; gap: var(--s2); padding: 0 var(--s3); height: var(--h-row); border-bottom: 1px solid var(--rule); color: var(--ink-faint); }
.zmc .pop__q { flex: 1; min-width: 0; border: 0; background: none; outline: none; font-size: var(--fs-3); color: var(--ink); }
.zmc .pop__list { list-style: none; margin: 0; padding: var(--s1); overflow-y: auto; min-height: 0; } .zmc .pop__list:empty { display: none; }
.zmc .pop__i { display: flex; align-items: center; gap: var(--s3); min-height: 40px; padding: var(--s1) var(--s2); border-radius: var(--r1); cursor: pointer; }
.zmc .pop__i--on { background: var(--tint-on); }
.zmc .pop__ic { flex: none; display: grid; color: var(--ink-muted); }
.zmc .pop__tx { flex: 1; min-width: 0; display: flex; flex-direction: column; line-height: 1.35; }
.zmc :is(.pop__t, .pop__s) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.zmc .pop__s { font-size: var(--fs-1); color: var(--ink-muted); }
.zmc .pop__status { padding: var(--s3); font-size: var(--fs-2); color: var(--ink-muted); } .zmc .pop__status:empty { display: none; }
.zmc .pop__status--bad { color: var(--danger); }
.zmc .menu { left: auto; right: 0; width: min(288px, 100%); padding: var(--s1); max-height: 304px; overflow-y: auto; }
.zmc :is(.menu--model, .menu--effort) { left: 0; right: auto; }
.zmc .menu__item { display: flex; align-items: flex-start; gap: var(--s2); width: 100%; min-height: var(--h-row); padding: var(--s2); border: 0; border-radius: var(--r1); background: none; text-align: left; color: var(--ink); transition: background var(--ease); }
.zmc .menu__item:hover, .zmc .menu__item:focus-visible { background: var(--tint-hover); outline: none; }
.zmc .menu__check { flex: none; width: 14px; margin-top: 2px; color: var(--ink-muted); }
.zmc .menu__tx { display: flex; flex-direction: column; min-width: 0; }
.zmc .menu__t { overflow-wrap: anywhere; }
.zmc .menu__d { font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.4; }
.zmc .menu__note { padding: var(--s2) var(--s3); font-size: var(--fs-2); color: var(--ink-muted); }
.zmc .menu__note--bad { color: var(--danger); }

/* ---------- other screens ---------- */
.zmc .vw { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.zmc .vw__head { flex: none; display: flex; align-items: center; gap: var(--s1); padding: var(--s2) var(--s4) 0 var(--s2); }
.zmc .vw__t { font-size: var(--fs-5); }
.zmc .vw__body { flex: 1; min-height: 0; overflow-y: auto; padding: var(--s1) var(--s4) var(--s5); overscroll-behavior: contain; }
.zmc .hist__search { position: sticky; top: 0; z-index: 2; padding: var(--s2) 0; background: var(--paper); }
.zmc .hist__group { padding: var(--s4) var(--s1) var(--s1); }
.zmc .hrow { display: flex; align-items: center; gap: 2px; padding-right: var(--s2); border-radius: var(--r1); }
.zmc .hrow:hover, .zmc .hrow:focus-within { background: var(--tint-hover); }
.zmc .hrow--on { background: var(--tint-on); }
.zmc .hrow__main { flex: 1; min-width: 0; display: flex; flex-direction: column; padding: var(--s2); border: 0; background: none; text-align: left; border-radius: var(--r1); }
.zmc .hrow__f { display: flex; align-items: center; gap: var(--s1); min-width: 0; margin-top: 2px; font: var(--fs-1) var(--mono); color: var(--ink-faint); }
 .zmc .hrow__f span { min-width: 0; }
.zmc .hrow__cmd { opacity: 0; } .zmc :is(.hrow:hover, .hrow:focus-within) .hrow__cmd, .zmc .hrow__cmd:focus-visible { opacity: 1; }
@media (hover: none) { .zmc .hrow__cmd { opacity: 0.7; } }
.zmc .hrow__copied { flex: none; color: var(--ok); }
.zmc .hist__note { display: flex; gap: var(--s2); margin: var(--s1) 0 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.5; } .zmc .hist__note svg { width: 14px; height: 14px; margin-top: 2px; }
.zmc .hrow__t { color: var(--ink); }
.zmc .hrow__m { font-size: var(--fs-1); color: var(--ink-muted); }
.zmc .hrow__del { opacity: 0; } .zmc :is(.hrow:hover, .hrow:focus-within) .hrow__del, .zmc .hrow__del:focus-visible { opacity: 1; }
@media (hover: none) { .zmc .hrow__del { opacity: 0.7; } }
.zmc .hrow--confirm { padding: var(--s2); gap: var(--s2); background: color-mix(in srgb, var(--danger) 8%, transparent); }
.zmc .hrow__q { flex: 1; }
.zmc .vempty { display: flex; flex-direction: column; align-items: center; gap: var(--s1); padding: 48px var(--s4); text-align: center; color: var(--ink-muted); }
.zmc .vempty > svg { width: 24px; height: 24px; margin-bottom: var(--s2); color: var(--ink-faint); }
.zmc .vempty__t { color: var(--ink); font-weight: 500; } .zmc .vempty__d { max-width: 22rem; margin-bottom: var(--s2); font-size: var(--fs-2); line-height: 1.5; }
.zmc .vwerr { padding: var(--s3); color: var(--danger); }
.zmc .status__top { display: flex; align-items: center; justify-content: space-between; gap: var(--s2); padding: var(--s1) 0 var(--s3); }
.zmc .status__sum { display: flex; align-items: center; gap: var(--s2); font-weight: 500; }
.zmc .status__dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); } .zmc .status__dot[data-ok="true"] { background: var(--ok); }
.zmc .checks { display: grid; gap: var(--s2); }
.zmc .check { display: flex; gap: var(--s3); padding: var(--s3); border: 1px solid var(--rule); border-radius: var(--r2); background: var(--paper-raised); }
.zmc .check--bad { border-color: var(--rule-strong); box-shadow: inset 2px 0 0 var(--warn); }
.zmc .check__i { color: var(--ok); } .zmc .check--bad .check__i { color: var(--warn); } .zmc .check__i svg { width: 16px; height: 16px; margin-top: 1px; }
.zmc .check__tx { flex: 1; min-width: 0; }
.zmc .check__t { font-weight: 500; line-height: 1.4; }
.zmc .check__d { font-size: var(--fs-2); color: var(--ink-muted); overflow-wrap: anywhere; }
.zmc .check__help { margin: var(--s2) 0 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.5; }
.zmc .check__acts { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s2); }
.zmc .set { display: grid; grid-template-columns: minmax(0, 1fr); gap: var(--s5); padding-top: var(--s2); }
.zmc .sec__t { margin-bottom: var(--s2); }
.zmc .sec__hint { margin: var(--s2) 0 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.5; overflow-wrap: anywhere; }
.zmc .keyrow { display: grid; gap: var(--s1); }
.zmc .keyrow__f { display: flex; gap: var(--s2); } .zmc .keyrow__f .input { flex: 1; min-width: 0; }
.zmc .keyrow__s { display: flex; align-items: center; justify-content: space-between; font-size: var(--fs-2); }
.zmc .keyrow__ok { display: inline-flex; align-items: center; gap: var(--s1); color: var(--ok); }
.zmc .radios { display: grid; gap: 2px; }
.zmc .radio { display: flex; align-items: flex-start; gap: var(--s3); width: 100%; padding: var(--s2); border: 0; border-radius: var(--r1); background: none; text-align: left; transition: background var(--ease); }
.zmc .radio:hover { background: var(--tint-hover); }
.zmc .radio__dot { flex: none; width: 16px; height: 16px; margin-top: 2px; border: 1.5px solid var(--rule-strong); border-radius: 50%; }
.zmc .radio--on .radio__dot { border-color: var(--ink); background: var(--ink); box-shadow: inset 0 0 0 3px var(--paper); }
.zmc .radio__tx { display: flex; flex-direction: column; min-width: 0; } .zmc .radio__d { font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.4; }
.zmc .pe { display: grid; gap: var(--s2); padding: var(--s3); margin-bottom: var(--s2); border: 1px solid var(--rule); border-radius: var(--r2); background: var(--paper-raised); }
.zmc .pe__row { display: flex; gap: var(--s2); align-items: center; } .zmc .pe__row .input:first-child { flex: 1; min-width: 0; }
.zmc .pe__slot { flex: none; width: auto; max-width: 9.5rem; }
.zmc .fields { display: grid; gap: var(--s4); margin-top: var(--s3); }
.zmc .field { display: grid; gap: var(--s2); }
.zmc .field__l { font-weight: 500; }
.zmc .select { position: relative; }
.zmc .select select { appearance: none; padding-right: var(--s5); cursor: pointer; }
.zmc .select::after { content: ""; position: absolute; right: var(--s3); top: 50%; width: 6px; height: 6px; margin-top: -5px; border-right: 1.5px solid var(--ink-muted); border-bottom: 1.5px solid var(--ink-muted); transform: rotate(45deg); pointer-events: none; }
.zmc .swrow { display: flex; align-items: center; justify-content: space-between; gap: var(--s4); min-height: var(--h-row); padding: var(--s2) 0; cursor: pointer; }
.zmc .swrow + .swrow { border-top: 1px solid var(--rule); }
.zmc .swrow__tx { display: flex; flex-direction: column; min-width: 0; }
.zmc .swrow__d { font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.4; }
.zmc .switch { position: relative; flex: none; width: 32px; height: 20px; }
.zmc .switch input { position: absolute; inset: 0; z-index: 1; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: pointer; }
.zmc .switch__track { position: absolute; inset: 0; border-radius: var(--pill); background: var(--paper-sunk); border: 1px solid var(--rule-strong); transition: background var(--ease), border-color var(--ease); }
.zmc .switch__thumb { position: absolute; top: 2px; left: 2px; width: 14px; height: 14px; border-radius: 50%; background: var(--ink-muted); transition: transform var(--ease), background var(--ease); }
.zmc .swrow:hover .switch__track { border-color: var(--ink-muted); }
.zmc .switch input:checked + .switch__track { background: var(--ink); border-color: var(--ink); }
.zmc .switch input:checked + .switch__track .switch__thumb { transform: translateX(12px); background: var(--paper); }
.zmc .switch input:focus-visible + .switch__track { outline: 2px solid var(--focus); outline-offset: 2px; }
.zmc .sk-group { display: grid; gap: var(--s3); margin-top: var(--s3); }
.zmc .sk-group .sk { margin: 0; } .zmc .sk--field { height: var(--h-md); } .zmc .sk--block { height: 96px; }
.zmc .inlineerr { display: flex; align-items: flex-start; gap: var(--s3); margin-top: var(--s3); padding: var(--s3); border: 1px solid var(--rule-strong); border-radius: var(--r2); background: var(--paper-raised); box-shadow: inset 2px 0 0 var(--danger); }
.zmc .inlineerr > svg { width: 16px; height: 16px; margin-top: 2px; color: var(--danger); }
.zmc .inlineerr__tx { flex: 1; min-width: 0; } .zmc .inlineerr__t { font-weight: 500; } .zmc .inlineerr__d { font-size: var(--fs-2); color: var(--ink-muted); overflow-wrap: anywhere; }
.zmc .datarow { display: flex; flex-direction: column; align-items: flex-start; gap: var(--s3); }
.zmc .confirm { display: flex; align-items: center; flex-wrap: wrap; gap: var(--s2); min-height: var(--h-sm); }
.zmc .confirm__q { font-size: var(--fs-2); } .zmc .confirm__msg { font-size: var(--fs-2); color: var(--ink-muted); }
.zmc .folder { display: flex; align-items: center; gap: var(--s2); min-height: var(--h-md); padding: 0 var(--s3); border: 1px solid var(--rule); border-radius: var(--r1); background: var(--paper-raised); color: var(--ink-muted); }

.zmc .folder__p { min-width: 0;  font: var(--fs-2) var(--mono); color: var(--ink); }
.zmc .folder__acts { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s2); }
.zmc .about { display: grid; gap: var(--s1); margin: var(--s3) 0 0; font-size: var(--fs-2); color: var(--ink-muted); }
.zmc .keys { display: grid; grid-template-columns: 1fr auto; gap: var(--s2) var(--s3); align-items: center; margin: 0; font-size: var(--fs-2); }
.zmc .keys dt { color: var(--ink-muted); } .zmc .keys dd { margin: 0; text-align: right; }
.zmc .set__saved { position: sticky; bottom: var(--s3); height: 0; display: flex; justify-content: center; overflow: visible; pointer-events: none; }
.zmc .set__saved span { align-self: flex-end; padding: var(--s1) var(--s3); border-radius: var(--pill); background: var(--ink); color: var(--paper); font-size: var(--fs-2); box-shadow: var(--shadow); }
.zmc .set__saved span:empty { display: none; }
`;

export const STYLES = BASE + CHAT_STYLES + WELCOME_STYLES;
