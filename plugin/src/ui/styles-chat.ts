// The conversation's styles: feed, messages, tool steps, plan, permission cards, Markdown. Same tokens as styles.ts.
export const CHAT_STYLES = `
.zmc .feedwrap { position: relative; flex: 1; min-height: 0; display: flex; flex-direction: column; }
.zmc .feed { flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable; }
.zmc .feed:focus-visible { outline-offset: -2px; }
.zmc .feed__inner { max-width: 46rem; margin: 0 auto; padding: var(--s4) var(--s4) var(--s5); display: flex; flex-direction: column; gap: var(--s5); min-height: 100%; }
.zmc .feed__inner--empty { justify-content: center; }
.zmc .jump { position: absolute; left: 50%; bottom: var(--s3); transform: translateX(-50%); z-index: 5; display: grid; place-items: center; width: var(--h-md); height: var(--h-md); padding: 0; border: 1px solid var(--rule); border-radius: 50%; background: var(--paper-raised); color: var(--ink); box-shadow: var(--shadow); animation: zmc-fade 120ms ease; transition: border-color var(--ease); }
.zmc .jump:hover { border-color: var(--rule-strong); }


/* ---------- messages ---------- */
.zmc .msg { min-width: 0; animation: zmc-rise 120ms ease; }
.zmc .msg--assistant:not([data-state="running"]) { content-visibility: auto; contain-intrinsic-size: auto 120px; }
.zmc .msg--user { display: flex; justify-content: flex-end; }
.zmc .ubub { max-width: min(88%, 34rem); padding: var(--s2) var(--s3); background: var(--paper-raised); border: 1px solid var(--rule); border-radius: var(--r3) var(--r3) var(--r0) var(--r3); }
.zmc[data-theme="dark"] .ubub { background: var(--paper-sunk); }
.zmc .ubub__text { margin: 0; font-size: var(--fs-4); line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; }
.zmc .ubub__chips { display: flex; flex-wrap: wrap; gap: var(--s1); margin-bottom: var(--s2); }
.zmc .chip--sm { gap: var(--s1); padding: 0 var(--s2); background: var(--paper); transition: color var(--ease), border-color var(--ease); }
.zmc .chip--sm:hover { color: var(--ink); border-color: var(--rule-strong); }
.zmc .parts { display: flex; flex-direction: column; gap: var(--s3); } .zmc .parts:empty { display: none; }
.zmc .working { display: flex; align-items: center; gap: var(--s2); height: var(--h-sm); color: var(--ink-muted); }
.zmc .parts:not(:empty) + .working { margin-top: var(--s2); }
.zmc .working--pending { margin-top: calc(-1 * var(--s2)); }
.zmc .foot { margin-top: var(--s1); }
.zmc .foot__row { display: flex; align-items: center; gap: var(--s2); min-height: var(--h-sm); }
.zmc .foot__fill { flex: 1; }
.zmc .foot__acts { display: inline-flex; margin-right: calc(-1 * var(--s1)); }
.zmc .foot__usage { font-size: var(--fs-1); color: var(--ink-muted); font-variant-numeric: tabular-nums; opacity: 0; transition: opacity var(--ease); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.zmc :is(.msg:hover, .msg:focus-within) .foot__usage { opacity: 1; }
@media (hover: none) { .zmc .foot__usage { opacity: 1; } }
.zmc .foot .iconbtn { color: var(--ink-faint); } .zmc .foot .iconbtn:hover { color: var(--ink); }
.zmc .foot__src { display: inline-flex; align-items: center; gap: var(--s1); height: var(--h-sm); padding: 0 var(--s2) 0 0; border: 0; border-radius: var(--pill); background: none; color: var(--ink-muted); font-size: var(--fs-2); transition: color var(--ease); }
.zmc .foot__src:hover { color: var(--ink); }
.zmc .foot__chev { width: 12px; height: 12px; transition: transform 160ms ease; } .zmc .foot--open .foot__chev { transform: rotate(90deg); }
.zmc .sources { list-style: none; margin: 0 0 var(--s1); padding: 0; display: grid; gap: 2px; }
.zmc .source { display: flex; align-items: center; gap: var(--s2); width: 100%; height: var(--h-sm); padding: 0 var(--s2); border: 0; border-radius: var(--r1); background: none; text-align: left; font-size: var(--fs-2); color: var(--ink); transition: background var(--ease); }
.zmc .source:hover { background: var(--tint-hover); }
.zmc .source svg { width: 14px; height: 14px; color: var(--ink-faint); }
.zmc .source__t { flex: 1; min-width: 0; }
.zmc .source__p { flex: none; color: var(--ink-muted); font-size: var(--fs-1); font-variant-numeric: tabular-nums; }
.zmc .stopnote { display: flex; align-items: center; gap: var(--s2); font-size: var(--fs-2); color: var(--ink-muted); }

.zmc .stopnote--refusal { color: var(--danger); } .zmc .stopnote--max_tokens { color: var(--warn); }
.zmc .notice { display: flex; gap: var(--s3); padding: var(--s3); border: 1px solid var(--rule-strong); border-radius: var(--r2); background: var(--paper-raised); line-height: 1.5; box-shadow: inset 2px 0 0 var(--info); }
.zmc .notice--warn { box-shadow: inset 2px 0 0 var(--warn); } .zmc .notice--error { box-shadow: inset 2px 0 0 var(--danger); }
.zmc .notice__i { color: var(--info); } .zmc .notice--warn .notice__i { color: var(--warn); } .zmc .notice--error .notice__i { color: var(--danger); }
.zmc .notice__i svg { width: 16px; height: 16px; margin-top: 2px; }
.zmc .notice__body { min-width: 0; flex: 1; }
.zmc .notice__msg { font-weight: 500; overflow-wrap: anywhere; }
.zmc .notice__hint { font-size: var(--fs-2); color: var(--ink-muted); }
.zmc .notice__acts { margin-top: var(--s2); }

/* ---------- thinking, steps, plan, permission ---------- */
.zmc .thought__head { display: inline-flex; align-items: center; gap: var(--s2); height: var(--h-sm); padding: 0 var(--s2) 0 0; border: 0; border-radius: var(--pill); background: none; color: var(--ink-muted); transition: color var(--ease); }
.zmc .thought__head:hover { color: var(--ink); }
.zmc .thought__dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ink-faint); }
.zmc .thought--active .thought__dot { background: var(--agent); animation: zmc-pulse 900ms ease-in-out infinite; }
.zmc .thought__chev { width: 12px; height: 12px; transition: transform 160ms ease; } .zmc .thought--open .thought__chev { transform: rotate(180deg); }
.zmc .thought__body { margin: var(--s1) 0 var(--s1) 2px; padding-left: var(--s3); border-left: 2px solid var(--rule); color: var(--ink-muted); font-style: italic; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 256px; overflow: auto; }
.zmc .steps { border-radius: var(--r2); background: color-mix(in srgb, var(--ink) 4.5%, transparent); overflow: hidden; }
.zmc .steps__head { display: flex; align-items: center; gap: var(--s2); width: 100%; height: var(--h-row); padding: 0 var(--s3); border: 0; background: none; color: var(--ink); text-align: left; transition: background var(--ease); }
.zmc .steps__head:hover { background: var(--tint-hover); }
.zmc .steps__head svg { width: 14px; height: 14px; color: var(--ink-muted); }
.zmc .steps__sum { flex: 1; min-width: 0; }
.zmc .steps__bad { color: var(--danger); }
.zmc .steps__chev { transition: transform 160ms ease; } .zmc .steps__head[aria-expanded="true"] .steps__chev { transform: rotate(180deg); }
.zmc .step + .step, .zmc .steps__head:not([hidden]) + .steps__list > .step:first-child { border-top: 1px solid color-mix(in srgb, var(--ink) 8%, transparent); }
.zmc .step__row { display: flex; align-items: center; gap: var(--s2); width: 100%; min-height: var(--h-row); padding: 0 var(--s3); border: 0; background: none; color: var(--ink); text-align: left; transition: background var(--ease); }
.zmc .step__row:hover:not(:disabled) { background: var(--tint-hover); }
.zmc .step__row:disabled { cursor: default; opacity: 1; }
.zmc .step__icon { flex: none; display: grid; color: var(--ink-muted); }
.zmc .step__name { flex: none; padding: 0 var(--s2); border: 1px solid var(--rule-strong); border-radius: var(--r0); font: var(--fs-1)/16px var(--mono); color: var(--ink-muted); }
.zmc .step__title { flex: 1; min-width: 0; }
.zmc .step--open .step__title { white-space: normal; overflow-wrap: anywhere; padding: var(--s2) 0; }
.zmc .step__st { flex: none; display: grid; place-items: center; width: 16px; height: 16px; color: var(--ink-faint); }
.zmc .step__st--failed { color: var(--danger); }
.zmc .step__chev { width: 12px; height: 12px; color: var(--ink-faint); transition: transform 160ms ease; } .zmc .step--open .step__chev { transform: rotate(180deg); }
.zmc .step__detail { display: grid; gap: var(--s2); padding: 0 var(--s3) var(--s3) 34px; }
.zmc .sd__head { display: flex; gap: var(--s2); margin-bottom: var(--s1); } .zmc .sd__trim { font-size: var(--fs-1); color: var(--ink-faint); }
.zmc .sd__pre { margin: 0; padding: var(--s2) var(--s3); max-height: 224px; overflow: auto; border-radius: var(--r1); background: var(--paper-sunk); font: var(--fs-1)/1.5 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; }
.zmc .plan { padding: var(--s3); border: 1px solid var(--rule); border-radius: var(--r2); background: var(--paper-raised); }
.zmc .plan__head { display: flex; align-items: center; gap: var(--s2); margin-bottom: var(--s2); font-size: var(--fs-2); font-weight: 600; color: var(--ink-muted); }
.zmc .plan__list { list-style: none; margin: 0; padding: 0; display: grid; gap: var(--s1); }
.zmc .plan__i { display: flex; align-items: flex-start; gap: var(--s2); color: var(--ink-muted); }
.zmc .plan__box { flex: none; display: grid; place-items: center; width: 16px; height: 16px; margin-top: 2px; border: 1.5px solid var(--ink-faint); border-radius: var(--r0); }
.zmc .plan__box svg { width: 12px; height: 12px; color: var(--paper-raised); stroke-width: 2; }
.zmc .plan__i--in_progress { color: var(--ink); } .zmc .plan__i--in_progress .plan__box { border-color: var(--agent); }
.zmc .plan__i--completed .plan__t { text-decoration: line-through; text-decoration-color: var(--rule-strong); }
.zmc .plan__i--completed .plan__box { background: var(--ink-muted); border-color: var(--ink-muted); }
.zmc .perm { padding: var(--s3) var(--s4); border: 1px solid var(--rule-strong); border-radius: var(--r2); background: var(--paper-raised); box-shadow: inset 2px 0 0 var(--warn); animation: zmc-rise 120ms ease; }
.zmc .perm__q { display: flex; align-items: center; gap: var(--s2); font-size: var(--fs-2); color: var(--ink-muted); } .zmc .perm__q svg { width: 14px; height: 14px; color: var(--warn); }
.zmc .perm__what { font-weight: 500; overflow-wrap: anywhere; }
.zmc .perm__more { height: var(--h-sm); margin-left: calc(-1 * var(--s2)); padding: 0 var(--s2); border: 0; border-radius: var(--pill); background: none; color: var(--ink-muted); font-size: var(--fs-2); text-decoration: underline; text-decoration-color: var(--rule-strong); text-underline-offset: 3px; transition: color var(--ease); }
.zmc .perm__more:hover { color: var(--ink); }
.zmc .perm__opts { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s3); }
.zmc .perm--done { display: flex; align-items: center; gap: var(--s2); padding: var(--s2) var(--s3); box-shadow: none; background: transparent; border-color: var(--rule); color: var(--ink-muted); }

.zmc .perm--done .perm__what { flex: 1; min-width: 0; font-weight: 400; }
.zmc .perm__res { flex: none; font-size: var(--fs-2); color: var(--ink); }

/* ---------- markdown ---------- */
.zmc .md { font-size: var(--fs-4); line-height: 1.6; overflow-wrap: anywhere; min-width: 0; }
.zmc .md > * { margin: 0 0 var(--s3); } .zmc .md > :last-child { margin-bottom: 0; }
.zmc .md strong { font-weight: 600; }
.zmc .md a { color: var(--link); text-decoration: underline; text-decoration-color: color-mix(in srgb, var(--link) 35%, transparent); text-underline-offset: 3px; text-decoration-thickness: 1px; }
.zmc .md a:hover { text-decoration-color: var(--link); }
.zmc .md :is(h1, h2, h3, h4, h5, h6) { line-height: 1.35; margin: var(--s4) 0 var(--s2); font-size: var(--fs-4); }
.zmc .md > :is(h1, h2, h3, h4):first-child { margin-top: 0; }
.zmc .md h1 { font-size: var(--fs-5); } .zmc .md :is(h4, h5, h6) { font-size: var(--fs-3); color: var(--ink-muted); }
.zmc .md :is(ul, ol) { padding-left: var(--s5); }
.zmc .md li { margin: var(--s1) 0; } .zmc .md li > :is(ul, ol, p) { margin: var(--s1) 0; } .zmc .md li::marker { color: var(--ink-faint); }
.zmc .md-task { margin-right: var(--s2); color: var(--ink-muted); } .zmc .md li:has(> .md-task) { list-style: none; margin-left: calc(-1 * var(--s4)); }
.zmc .md code { font: 0.9em var(--mono); padding: 1px var(--s1); border-radius: var(--r0); background: color-mix(in srgb, var(--ink) 7%, transparent); }
.zmc .md pre code { padding: 0; background: none; font-size: inherit; }
.zmc .md blockquote { margin-left: 0; padding-left: var(--s3); border-left: 2px solid var(--rule-strong); color: var(--ink-muted); }
.zmc .md blockquote > * { margin: 0 0 var(--s2); } .zmc .md blockquote > :last-child { margin-bottom: 0; }
.zmc .md hr { border: 0; border-top: 1px solid var(--rule); }
.zmc .md img { max-width: 100%; height: auto; border-radius: var(--r1); }
.zmc :is(.md-raw, .md-tail) { white-space: pre-wrap; font: var(--fs-2)/1.5 var(--mono); color: var(--ink-muted); }
.zmc .md--streaming > p:last-child::after, .zmc .md--streaming > :is(ul, ol):last-child > li:last-child::after { content: ""; display: inline-block; width: 2px; height: 1em; margin-left: 2px; margin-right: -4px; vertical-align: -0.15em; background: var(--agent); animation: zmc-caret 1.1s steps(1, end) infinite; }
.zmc .md-table { overflow-x: auto; border: 1px solid var(--rule); border-radius: var(--r1); }
.zmc .md-table table { border-collapse: collapse; width: 100%; min-width: max-content; font-size: var(--fs-3); }
.zmc .md-table :is(th, td) { padding: var(--s2) var(--s3); border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; max-width: 22rem; overflow-wrap: normal; }
.zmc .md-table th { font-weight: 600; font-size: var(--fs-2); color: var(--ink-muted); background: var(--paper-sunk); white-space: nowrap; }
.zmc .md-table tr:last-child td { border-bottom: 0; }
.zmc .md-table [align="center"] { text-align: center; } .zmc .md-table [align="right"] { text-align: right; }
.zmc .code { border: 1px solid var(--rule); border-radius: var(--r1); background: var(--paper-raised); overflow: hidden; }
.zmc[data-theme="dark"] .code { background: var(--paper-sunk); }
.zmc .code__head { display: flex; align-items: center; justify-content: space-between; height: var(--h-sm); padding: 0 var(--s1) 0 var(--s3); border-bottom: 1px solid var(--rule); background: color-mix(in srgb, var(--ink) 3%, transparent); }
.zmc .code__lang { font: var(--fs-1) var(--mono); color: var(--ink-muted); }
.zmc .code pre { margin: 0; padding: var(--s2) var(--s3); overflow: auto; max-height: 360px; font: var(--fs-2)/1.5 var(--mono); white-space: pre; tab-size: 2; }
.zmc .math--raw { font: 0.9em var(--mono); color: var(--ink-muted); white-space: pre-wrap; }
.zmc .math--display { display: block; margin: var(--s2) 0; padding: var(--s1) 0; overflow-x: auto; overflow-y: hidden; text-align: center; }
.zmc .math math { font-size: 1.1em; font-family: "STIX Two Math", "Latin Modern Math", "Cambria Math", "Times New Roman", serif; }
.zmc .math--error { font: 0.9em var(--mono); padding: 1px var(--s1); border-radius: var(--r0); background: color-mix(in srgb, var(--danger) 9%, var(--paper-sunk)); cursor: help; white-space: pre-wrap; }
.zmc div.math--error { padding: var(--s2) var(--s3); text-align: left; }
.zmc .md-nobr { white-space: nowrap; }
.zmc .md-nobr .cite { white-space: normal; }
.zmc .cite { display: inline; margin: 0 1px; padding: 0; border: 0; background: none; font-size: var(--fs-2); line-height: 1; max-width: 100%; }
.zmc .cite__t { display: inline-flex; align-items: center; gap: var(--s1); padding: 2px var(--s2) 2px var(--s1); border-radius: var(--pill); background: color-mix(in srgb, var(--ink) 7%, transparent); color: var(--ink-muted); vertical-align: 0.1em; transition: background var(--ease), color var(--ease); }
.zmc .cite__t svg { width: 12px; height: 12px; color: var(--ink-faint); }
.zmc .cite:hover .cite__t { background: color-mix(in srgb, var(--link) 14%, transparent); color: var(--link); }
.zmc .cite:focus-visible { outline-offset: 1px; border-radius: var(--pill); }
`;
