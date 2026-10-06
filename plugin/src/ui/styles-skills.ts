// Skills and prompts: the pinned list in the empty state, the `/` menu's groups (the menu itself is the `.pop` surface),
// and the settings card with its add-skill preview. The glass versions are in styles-look.ts.
export const SKILL_STYLES = `
/* the empty state's "Start with": the pinned skills and prompts, one row each, in a quiet card */
.zmc .pins { margin: var(--s5) auto 0; max-width: 28rem; text-align: left; }
.zmc .pins__head { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--s1); padding-left: calc(var(--s1) + var(--s3)); }
.zmc .pins__list { list-style: none; margin: 0; padding: var(--s1); display: grid; grid-template-columns: minmax(0, 1fr); gap: 2px; border: 1px solid var(--rule); border-radius: var(--r2); background: var(--paper-raised); }
.zmc .pins__none { margin: 0; padding: 0 var(--s3) 0 calc(var(--s1) + var(--s3)); font-size: var(--fs-2); line-height: 1.5; color: var(--ink-muted); }
.zmc .pin { display: flex; align-items: center; gap: var(--s2); width: 100%; height: var(--h-row); padding: 0 var(--s2) 0 var(--s3); border: 0; border-radius: var(--r1); background: none; color: var(--ink); text-align: left; transition: background var(--ease); }
.zmc .pin:hover:not(:disabled) { background: var(--tint-hover); }
.zmc .pin:disabled { color: var(--ink-faint); }
.zmc .pin__t { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.zmc .pin__skill { flex: none; display: inline-flex; color: var(--ink-faint); } .zmc .pin__skill svg { width: 12px; height: 12px; }
.zmc .pin__k { flex: none; margin-left: auto; border-color: transparent; background: color-mix(in srgb, var(--ink) 6%, transparent); }

/* the / menu: quiet group headings; a skill reads as the command it is */
.zmc .pop__h { padding: var(--s2) var(--s2) 2px calc(var(--s2) + 14px + var(--s3)); font-size: var(--fs-1); color: var(--ink-faint); line-height: 1.3; }
.zmc .pop__h:first-child { padding-top: var(--s1); }
.zmc .pop--slash .pop__i { min-height: 36px; }
.zmc .pop--slash .pop__ic svg { width: 14px; height: 14px; }
.zmc .pop__t--cmd { font: 500 var(--fs-2)/1.45 var(--mono); }

/* the settings card */
.zmc .sp__list { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2px; margin: 0 calc(-1 * var(--s2)); }
.zmc .sp { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s2); min-height: 44px; padding: var(--s1) var(--s2); border-radius: var(--r1); transition: background var(--ease); }
.zmc .sp:hover, .zmc .sp:focus-within, .zmc .sp--open { background: var(--tint-hover); }
.zmc .sp__tx { flex: 1 1 100px; min-width: 0; }
.zmc .sp__t { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
.zmc .sp__t--cmd { font: 500 var(--fs-2)/1.5 var(--mono); }
.zmc .sp__tag { color: var(--ink-faint); }
.zmc .sp__d { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.4; }
.zmc .sp--off .sp__tx { opacity: 0.5; }
.zmc .sp__acts { flex: none; display: flex; align-items: center; gap: 2px; margin-left: auto; }
.zmc .sp__pin { display: inline-grid; place-items: center; min-width: 26px; height: 24px; padding: 0 6px; border: 1px solid transparent; border-radius: var(--pill); background: none; color: var(--ink-faint); transition: background var(--ease), color var(--ease); }
.zmc .sp__pin svg { width: 14px; height: 14px; }
.zmc .sp__pin:hover:not(:disabled) { color: var(--ink); background: var(--tint-on); }
.zmc .sp__pin:disabled { opacity: 0.35; }
.zmc .sp__pin--on { color: var(--ink); border-color: var(--rule); background: var(--paper-sunk); }
.zmc .sp__slot { font: var(--fs-1)/1 var(--mono); white-space: nowrap; }
.zmc .switch--sm { width: 28px; height: 16px; margin: 0 var(--s1); }
.zmc .switch--sm .switch__thumb { width: 10px; height: 10px; }
.zmc .sp__acts .iconbtn--sm { color: var(--ink-faint); } .zmc .sp__acts .iconbtn--sm:hover { color: var(--ink); }
.zmc .sp--confirm { gap: var(--s2); background: color-mix(in srgb, var(--danger) 8%, transparent); }
.zmc .sp--confirm:hover { background: color-mix(in srgb, var(--danger) 8%, transparent); }
.zmc .sp__q { flex: 1 1 140px; font-size: var(--fs-2); }
.zmc .sp__list > .pe { margin: 2px var(--s2) var(--s2); }
.zmc .pe__acts { display: flex; justify-content: flex-end; gap: var(--s2); }
.zmc .pe__del { margin-right: auto; padding-left: var(--s2); }
.zmc .sp__path { display: flex; align-items: center; gap: 6px; min-width: 0; font: var(--fs-1) var(--mono); color: var(--ink-muted); }
.zmc .sp__path svg { width: 12px; height: 12px; } .zmc .sp__path span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.zmc .sp__src { min-height: 240px; resize: vertical; font: var(--fs-2)/1.55 var(--mono); tab-size: 2; }
.zmc .sp__dir { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font: var(--fs-2) var(--mono); color: var(--ink); }
.zmc .sp__btns { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s3); }
.zmc .sp__msg { margin-top: var(--s2); color: var(--ink); }

/* Add skill: the safety line, name and description, the whole text, then what is copied and what is left out */
.zmc .imp { display: grid; gap: var(--s3); margin-top: var(--s3); padding: var(--s3); border: 1px solid var(--rule-strong); border-radius: var(--r2); background: color-mix(in srgb, var(--ink) 2.5%, transparent); }
.zmc .imp__warn { display: flex; gap: var(--s2); padding: var(--s2) var(--s3); border-radius: var(--r1); background: color-mix(in srgb, var(--warn) 11%, transparent); font-size: var(--fs-2); line-height: 1.45; }
.zmc .imp__warn svg { width: 14px; height: 14px; margin-top: 2px; color: var(--warn); }
.zmc .imp .field { gap: var(--s1); } .zmc .imp .field__l { font-size: var(--fs-2); }
.zmc .imp__h { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s2); font-size: var(--fs-2); font-weight: 600; color: var(--ink-muted); }
.zmc .imp__n { font-size: var(--fs-1); font-weight: 400; color: var(--ink-faint); font-variant-numeric: tabular-nums; }
.zmc .imp__pre { margin: calc(-1 * var(--s2)) 0 0; max-height: 260px; overflow: auto; padding: var(--s2) var(--s3); border: 1px solid var(--rule); border-radius: var(--r1); background: color-mix(in srgb, var(--ink) 3.5%, transparent); font: var(--fs-1)/1.6 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; color: var(--ink); }
.zmc .imp__files { display: grid; gap: var(--s1); }
.zmc .imp__list { list-style: none; margin: 0 0 var(--s2); padding: 0; display: grid; gap: 3px; font-size: var(--fs-2); }
.zmc .imp__list li { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px var(--s2); min-width: 0; }
.zmc .imp__list svg { width: 12px; height: 12px; align-self: center; color: var(--ok); }
.zmc .imp__list--skip svg { color: var(--ink-faint); }
.zmc .imp__f { min-width: 0; font: var(--fs-1)/1.5 var(--mono); overflow-wrap: anywhere; }
.zmc .imp__list--skip .imp__f { color: var(--ink-muted); }
.zmc .imp__why { margin-left: auto; color: var(--ink-faint); font-size: var(--fs-1); }
.zmc .imp__keep { display: flex; align-items: flex-start; gap: var(--s2); font-size: var(--fs-2); color: var(--ink-muted); line-height: 1.45; cursor: pointer; }
.zmc .imp__keep input { flex: none; margin: 3px 0 0; accent-color: var(--ink); }
.zmc .imp__err { margin: 0; font-size: var(--fs-2); color: var(--danger); }
.zmc .imp__acts { display: flex; gap: var(--s2); }
`;
