// The first-run welcome (welcome.ts): the animated logo, the agent cards, the setup checklist. Motion is entrance and
// state change only (nothing loops except the typing dots while the check runs); reduced motion turns it all off in styles.ts.
export const WELCOME_STYLES = `
@keyframes zmc-lift { from { opacity: 0; transform: translateY(10px); } }
@keyframes zmc-pop { from { opacity: 0; transform: scale(0.6); } 60% { transform: scale(1.06); } to { opacity: 1; transform: scale(1); } }
@keyframes zmc-spark { from { opacity: 0; transform: scale(0) rotate(-120deg); } 70% { transform: scale(1.15) rotate(8deg); } to { opacity: 1; transform: none; } }
@keyframes zmc-wave { 0%, 55%, 100% { transform: translateY(0); } 25% { transform: translateY(-5px); } }
@keyframes zmc-twinkle { 0%, 100% { transform: scale(1) rotate(0); } 50% { transform: scale(1.14) rotate(10deg); } }
@keyframes zmc-burst { 0% { transform: scale(1) rotate(0); } 40% { transform: scale(1.5) rotate(25deg); } 100% { transform: scale(1) rotate(0); } }
@keyframes zmc-spray { from { opacity: 1; transform: translate(0, 0) scale(1); } to { opacity: 0; transform: translate(var(--dx), var(--dy)) scale(0.3); } }
@keyframes zmc-spin { to { transform: rotate(360deg); } }
@keyframes zmc-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
@keyframes zmc-ripple { from { box-shadow: 0 0 0 0 color-mix(in srgb, var(--ok) 45%, transparent); } to { box-shadow: 0 0 0 8px transparent; } }

/* ---------- the logo ---------- */
.zmc .mark :is(.mk-b, .mk-d, .mk-w, .mk-t, .mk-s, .mk-p) { transform-box: fill-box; transform-origin: center; }
.zmc .mk-p { opacity: 0; }
.zmc .mark--hero { width: 64px; height: 64px; margin-bottom: var(--s3); }
.zmc .mark--hero .mk-b { animation: zmc-pop 560ms cubic-bezier(0.2, 0.9, 0.3, 1.2) both; }
.zmc .mark--hero .mk-d { animation: zmc-pop 360ms ease-out calc(420ms + var(--i) * 90ms) both; }
.zmc .mark--hero .mk-s { animation: zmc-spark 700ms cubic-bezier(0.2, 0.9, 0.3, 1.1) 560ms both; }
.zmc .mark--hero .mk-t { animation: zmc-twinkle 3.6s ease-in-out 1.6s infinite; }
.zmc .mark--busy .mk-w { animation: zmc-wave 1.3s ease-in-out calc(900ms + var(--i) * 140ms) infinite; }
.zmc .mark--ready .mk-t { animation: zmc-burst 760ms ease-out; }
.zmc .mark--ready .mk-p { animation: zmc-spray 820ms ease-out both; }
.zmc .mark--hello .mk-t { animation: zmc-burst 760ms ease-out; }

/* ---------- the screen ---------- */
.zmc .wel { flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: var(--s4) var(--s4) 0; scrollbar-width: thin; scrollbar-color: var(--rule-strong) transparent; }
.zmc .wel__in { display: grid; gap: var(--s4); max-width: 26rem; margin: 0 auto; }
.zmc .wel__hero { text-align: center; }
.zmc .wel__hero .mark { margin-inline: auto; }
.zmc .wel__hero h2 { font-size: var(--fs-6); line-height: 1.25; letter-spacing: -0.01em; margin-bottom: var(--s2); animation: zmc-lift 420ms ease-out 200ms both; }
.zmc .wel__lead { margin: 0 auto; max-width: 22rem; color: var(--ink-muted); animation: zmc-lift 420ms ease-out 300ms both; }
.zmc .wel__sec { display: grid; gap: var(--s3); align-content: start; animation: zmc-lift 460ms ease-out both; }
.zmc .wel__sec:nth-of-type(1) { animation-delay: 480ms; }
.zmc .wel__sec:nth-of-type(2) { animation-delay: 620ms; }
.zmc .wel__head { display: flex; align-items: center; gap: var(--s2); }
.zmc .wel__n { flex: none; display: grid; place-items: center; width: 20px; height: 20px; border-radius: 50%; background: var(--ink); color: var(--paper); font-size: var(--fs-1); font-weight: 600; }
.zmc .wel__h { font-size: var(--fs-4); flex: 1; }
.zmc .wel__count { font-size: var(--fs-2); color: var(--ink-muted); font-variant-numeric: tabular-nums; }
.zmc .wel__hint { margin: 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.5; } .zmc .wel__hint:empty { display: none; }
.zmc .sk--card { height: 60px; margin: 0; }

/* ---------- agent cards ---------- */
.zmc .wcards { display: grid; gap: var(--s2); }
.zmc .wcard { display: flex; align-items: center; gap: var(--s3); width: 100%; padding: var(--s3); border: 1px solid var(--rule); border-radius: var(--r2); background: var(--paper-raised); text-align: left; transition: border-color var(--ease), box-shadow var(--ease), transform var(--ease); animation: zmc-fade 240ms ease both; }
.zmc .wcard:hover { border-color: var(--rule-strong); }
.zmc .wcard--on, .zmc .wcard--on:hover { border-color: var(--ink); box-shadow: 0 0 0 1px var(--ink); }
.zmc .wcard__dot { flex: none; width: 16px; height: 16px; border: 1.5px solid var(--rule-strong); border-radius: 50%; transition: border-color var(--ease), background var(--ease), box-shadow var(--ease); }
.zmc .wcard--on .wcard__dot { border-color: var(--ink); background: var(--ink); box-shadow: inset 0 0 0 3px var(--paper-raised); }
.zmc .wcard__tx { flex: 1; min-width: 0; display: flex; flex-direction: column; line-height: 1.4; }
.zmc .wcard__t { font-weight: 600; }
.zmc .wcard__d { font-size: var(--fs-2); color: var(--ink-muted); }
.zmc .wcard__st { flex: none; display: inline-flex; align-items: center; gap: var(--s1); max-width: 40%; font-size: var(--fs-1); color: var(--ink-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.zmc .wcard__led { flex: none; width: 7px; height: 7px; border-radius: 50%; background: var(--warn); }
.zmc .wcard__st--ok .wcard__led { background: var(--ok); }
@container zmc (max-width: 340px) { .zmc .wcard__st { display: none; } }

/* ---------- the checklist ---------- */
.zmc .wbar { height: 3px; border-radius: var(--pill); background: var(--paper-sunk); overflow: hidden; }
.zmc .wbar__fill { width: 0; height: 100%; border-radius: inherit; background: var(--ok); transition: width 420ms cubic-bezier(0.3, 0.7, 0.3, 1); }
.zmc .wrows { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.zmc .wrows .sk { margin: 0; height: 40px; }
.zmc .wrow { display: flex; gap: var(--s3); padding: var(--s2) var(--s1); border-radius: var(--r1); animation: zmc-fade 200ms ease; }
.zmc .wrow--bad { padding: var(--s3); background: var(--paper-raised); border: 1px solid var(--rule-strong); box-shadow: inset 2px 0 0 var(--warn); margin: var(--s1) 0; border-radius: var(--r2); }
.zmc .wrow__i { flex: none; display: grid; place-items: center; width: 20px; height: 20px; margin-top: 1px; border-radius: 50%; color: var(--warn); }
.zmc .wrow__i svg { width: 14px; height: 14px; }
.zmc .wrow--ok .wrow__i { background: var(--ok); color: var(--paper-raised); animation: zmc-ripple 700ms ease-out; }
.zmc .tick path { stroke-dasharray: 1; animation: zmc-draw 360ms ease-out 80ms both; }
.zmc .spin { width: 14px; height: 14px; border: 2px solid var(--rule-strong); border-top-color: var(--ink); border-radius: 50%; animation: zmc-spin 700ms linear infinite; }
.zmc .wrow__tx { flex: 1; min-width: 0; }
.zmc .wrow__t { font-weight: 500; line-height: 1.4; }
.zmc .wrow--pending .wrow__t { color: var(--ink-muted); }
.zmc .wrow__d { font-size: var(--fs-2); color: var(--ink-muted); overflow-wrap: anywhere; }
.zmc .wrow__help { margin: var(--s2) 0 0; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.5; }
.zmc .wrow__acts { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s2); }

/* ---------- the way out ---------- */
.zmc .wfoot { position: sticky; bottom: 0; z-index: 2; display: grid; gap: var(--s2); justify-items: center; text-align: center; margin: 0 calc(-1 * var(--s4)); padding: var(--s5) var(--s4) var(--s3); background: linear-gradient(to bottom, transparent, var(--paper) 42%); animation: zmc-lift 460ms ease-out 760ms both; }
.zmc .wready { display: inline-flex; align-items: center; gap: var(--s2); margin: 0; font-weight: 500; animation: zmc-lift 320ms ease-out both; }
.zmc .wready__i { display: grid; place-items: center; width: 20px; height: 20px; border-radius: 50%; background: var(--ok); color: var(--paper-raised); animation: zmc-ripple 800ms ease-out; }
.zmc .wready__i svg { width: 14px; height: 14px; }
.zmc .wfoot__acts { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: var(--s2); }
.zmc .wfoot__acts .btn--solid { min-width: 9rem; height: var(--h-row); font-weight: 500; }
.zmc .wfoot__skip { color: var(--ink-faint); }
`;
