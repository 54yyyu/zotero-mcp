// The "Skills and prompts" card: one list (a quiet skill / prompt mark on each), pins (four, the shortcut shown), on/off,
// edit, delete; Add skill… (the full text and every file it copies or leaves out, shown before anything is copied), New
// prompt, Create with the agent; and where the skills folder is. The rules are pure functions in skills-model.ts.
import type { PanelSettings, SettingsHost, SkillEntry, SkillImport } from "../types.ts";
import { env, errMessage, h, icon, isMac, nextId, slotLabel } from "./dom.ts";
import type { Kid } from "./dom.ts";
import { addPrompt, editPrompt, removePrompt, shortPath } from "./settings-model.ts";
import { hint, section } from "./settings-parts.ts";
import { forgetSkill, isOn, nameProblem, setOn, slotOf, togglePin } from "./skills-model.ts";
import type { ItemRef } from "./skills-model.ts";

const SAFETY = "Skills tell your agent what to do, and it acts with your permissions. Only add skills you trust or wrote.";

export function skillsCard(host: SettingsHost, o: { save(patch: Partial<PanelSettings>): Promise<void>; rerender(): void; createSkill?: () => void }): () => HTMLElement {
  let skills: SkillEntry[] | null = null;
  let loadErr = "";
  let editing: string | null = null; // "p:<id>" or "s:<name>"
  let source: string | null = null; // the SKILL.md being edited
  let confirming: string | null = null;
  let adding: (SkillImport & { keep: boolean; err: string }) | null = null;
  let msg = "";
  let timer: number | undefined;
  let card: HTMLElement | null = null;

  const say = (text: string) => { msg = text; env.win.clearTimeout(timer); timer = env.win.setTimeout(() => { msg = ""; o.rerender(); }, 5000); o.rerender(); };
  const load = () => host.skills.list().then((l) => { skills = l; loadErr = ""; }, (e) => { loadErr = errMessage(e); }).finally(o.rerender);
  const keyOf = (r: ItemRef) => (r.kind === "prompt" ? `p:${r.id}` : `s:${r.name}`);

  async function editSkill(name: string) {
    try { source = await host.skills.read(name); editing = `s:${name}`; } catch (e) { say(`Couldn't open it: ${errMessage(e)}`); return; }
    o.rerender();
  }

  async function del(r: ItemRef) {
    const s = host.getSettings();
    confirming = null;
    try {
      if (r.kind === "prompt") await o.save({ prompts: removePrompt(s.prompts, s.prompts.findIndex((p) => p.id === r.id)) });
      else { await host.skills.remove(r.name); await o.save(forgetSkill(s, r.name)); void load(); }
    } catch (e) { say(`Couldn't delete it: ${errMessage(e)}`); }
  }

  async function pick() {
    try {
      const plan = await host.skills.pick();
      if (plan) adding = { ...plan, keep: false, err: "" };
    } catch (e) { say(`Couldn't read that file: ${errMessage(e)}`); }
    o.rerender();
    card?.querySelector(".imp")?.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  async function add() {
    if (!adding) return;
    const a = adding;
    a.err = nameProblem(a.name, (skills ?? []).map((k) => k.name)) ?? (a.description.trim() ? "" : "Give it a one-line description: your agent reads it to know when to use the skill.");
    if (a.err) { o.rerender(); return; }
    try {
      await host.skills.add(a, { name: a.name, description: a.description, keepSkipped: a.keep });
      adding = null;
      await load();
      say(`Added ${a.name}. Type /${a.name} in the message box to use it.`);
    } catch (e) { a.err = errMessage(e); o.rerender(); }
  }

  // ───────────── rows ─────────────

  function row(r: ItemRef, title: string, desc: string, tag: string, builtin = false): HTMLElement[] {
    const s = host.getSettings();
    const key = keyOf(r);
    if (confirming === key) {
      return [h("div.sp.sp--confirm", { role: "group", "aria-label": `Delete ${title}?` },
        h("span.sp__q", null, r.kind === "skill" ? `Delete the skill ${r.name} and its folder?` : `Delete “${title}”?`),
        h("button.btn.btn--sm.btn--danger", { type: "button", dataset: { fid: `del-yes:${key}` }, onclick: () => void del(r) }, "Delete"),
        h("button.btn.btn--sm", { type: "button", onclick: () => { confirming = null; o.rerender(); } }, "Cancel"))];
    }
    const slot = slotOf(s, r);
    const on = isOn(s, r);
    const pinBtn = h(`button.sp__pin${slot ? ".sp__pin--on" : ""}`, {
      type: "button", "aria-pressed": String(!!slot), dataset: { fid: `pin:${key}` }, disabled: on ? null : true,
      "aria-label": slot ? `Unpin ${title} (shortcut ${slotLabel(slot)})` : `Pin ${title}`,
      title: slot ? `Pinned: listed on a new chat, and ${slotLabel(slot)}. Click to unpin.` : "Pin: list it on a new chat, with a shortcut",
      onclick: () => { const p = togglePin(s, r); if (p) void o.save(p); else say("Four are pinned. Unpin one first."); },
    }, slot ? h("span.sp__slot", null, slotLabel(slot)) : icon("bookmark"));
    const sw = h("input", { type: "checkbox", role: "switch", checked: on, "aria-label": `Use ${title}`, dataset: { fid: `on:${key}` }, onchange: (e: Event) => void o.save(setOn(s, r, (e.target as HTMLInputElement).checked)) });
    const out: HTMLElement[] = [h(`div.sp${on ? "" : ".sp--off"}${editing === key ? ".sp--open" : ""}`, null,
      h("div.sp__tx", null,
        h(`div.sp__t${r.kind === "skill" ? ".sp__t--cmd" : ""}`, { title }, title),
        h("div.sp__d", { title: desc || null }, h("span.sp__tag", null, tag), desc ? ` · ${desc}` : "")),
      h("div.sp__acts", null,
        pinBtn,
        h("label.switch.switch--sm", { title: on ? "On: in the / menu" : "Off" }, sw, h("span.switch__track", null, h("span.switch__thumb"))),
        builtin ? null : h("button.iconbtn.iconbtn--sm", { type: "button", "aria-label": `Edit ${title}`, title: "Edit", "aria-expanded": String(editing === key), dataset: { fid: `edit:${key}` },
          onclick: () => { if (editing === key) { editing = null; o.rerender(); } else if (r.kind === "skill") void editSkill(r.name); else { editing = key; o.rerender(); } } }, icon("pencil"))))];
    if (editing === key) out.push(r.kind === "prompt" ? promptEditor(r.id) : skillEditor(r.name));
    return out;
  }

  /** Delete lives in the editor, so a row stays one line at 300 px; it asks first, in the row's place. */
  const delBtn = (key: string, title: string) => h("button.btn.btn--sm.btn--quiet-danger.pe__del", { type: "button", "aria-label": `Delete ${title}`, dataset: { fid: `del:${key}` }, onclick: () => { confirming = key; editing = null; o.rerender(); } }, icon("trash"), "Delete");

  function promptEditor(id: string): HTMLElement {
    const list = host.getSettings().prompts;
    const i = list.findIndex((p) => p.id === id);
    const p = list[i];
    if (!p) return h("div");
    const ta = h("textarea.input.pe__text", { rows: "3", placeholder: "The message to send", "aria-label": "Prompt text", dataset: { fid: `ptext:${id}` }, onchange: () => void o.save({ prompts: editPrompt(host.getSettings().prompts, i, { text: (ta as HTMLTextAreaElement).value }) }) }) as HTMLTextAreaElement;
    ta.value = p.text;
    return h("div.pe", null,
      h("input.input.input--sm", { type: "text", value: p.title, placeholder: "Title (the button's label)", "aria-label": "Prompt title", dataset: { fid: `ptitle:${id}` }, onchange: (e: Event) => void o.save({ prompts: editPrompt(host.getSettings().prompts, i, { title: (e.target as HTMLInputElement).value }) }) }),
      ta,
      h("div.pe__acts", null, delBtn(`p:${id}`, p.title || "this prompt"), h("button.btn.btn--sm", { type: "button", onclick: () => { editing = null; o.rerender(); } }, "Done")));
  }

  function skillEditor(name: string): HTMLElement {
    const ta = h("textarea.input.sp__src", { rows: "14", spellcheck: "false", "aria-label": `${name}: SKILL.md`, oninput: () => { source = (ta as HTMLTextAreaElement).value; } }) as HTMLTextAreaElement;
    ta.value = source ?? "";
    const saveSkill = async () => {
      try { await host.skills.write(name, ta.value); editing = null; source = null; await load(); say(`Saved ${name}.`); } catch (e) { say(`Couldn't save: ${errMessage(e)}`); }
    };
    return h("div.pe", null,
      h("div.sp__path", { title: `${host.skills.dir}/${name}/SKILL.md` }, icon("file"), h("span", null, `${name}/SKILL.md`)),
      ta,
      h("div.pe__acts", null,
        delBtn(`s:${name}`, `/${name}`),
        h("button.btn.btn--sm.btn--solid", { type: "button", onclick: () => void saveSkill() }, "Save"),
        h("button.btn.btn--sm", { type: "button", onclick: () => { editing = null; source = null; o.rerender(); } }, "Cancel")));
  }

  // ───────────── adding a skill ─────────────

  function preview(a: NonNullable<typeof adding>): HTMLElement {
    const keepable = a.skip.filter((x) => x.keepable);
    const field = (label: string, value: string, fid: string, set: (v: string) => void, ph: string) =>
      h("label.field", null, h("span.field__l", null, label), h("input.input.input--sm", { type: "text", value, placeholder: ph, spellcheck: "false", dataset: { fid }, oninput: (e: Event) => set((e.target as HTMLInputElement).value) }));
    return h("div.imp", { role: "region", "aria-label": "Add a skill" },
      h("div.imp__warn", { role: "note" }, icon("shield"), h("span", null, SAFETY)),
      field("Name", a.name, "imp:name", (v) => { a.name = v.trim(); }, "annotate-paper"),
      field("Description", a.description, "imp:desc", (v) => { a.description = v; }, "What it does and when to use it, in one line"),
      h("div.imp__h", null, a.loose ? `${a.source.split("/").pop()}, which becomes the skill's SKILL.md` : "SKILL.md", h("span.imp__n", null, `${a.text.length.toLocaleString()} characters`)),
      h("pre.imp__pre", { tabindex: "0", "aria-label": "The skill's full text" }, a.text),
      a.loose ? null : h("div.imp__files", null,
        h("div.imp__h", null, "Copies"),
        h("ul.imp__list", null, ["SKILL.md", ...a.copy].map((f) => h("li", null, icon("check"), h("span.imp__f", null, f)))),
        a.skip.length ? [h("div.imp__h", null, "Leaves out"), h("ul.imp__list.imp__list--skip", null, a.skip.map((x) => h("li", null, icon("close"), h("span.imp__f", null, x.path), h("span.imp__why", null, x.reason))))] : null,
        keepable.length ? h("label.imp__keep", null,
          h("input", { type: "checkbox", checked: a.keep, onchange: (e: Event) => { a.keep = (e.target as HTMLInputElement).checked; o.rerender(); } }),
          h("span", null, `Also copy the ${keepable.length === 1 ? "file" : `${keepable.length} files`} marked as scripts or other files. Only if you have read ${keepable.length === 1 ? "it" : "them"}: your agent may run ${keepable.length === 1 ? "it" : "them"}.`)) : null),
      a.err ? h("p.imp__err", { role: "alert" }, a.err) : null,
      h("div.imp__acts", null,
        h("button.btn.btn--sm.btn--solid", { type: "button", dataset: { fid: "imp:add" }, onclick: () => void add() }, "Add skill"),
        h("button.btn.btn--sm", { type: "button", onclick: () => { adding = null; o.rerender(); } }, "Cancel")));
  }

  return () => {
    if (!skills && !loadErr) void load();
    const s = host.getSettings();
    const rows: Kid[] = [];
    if (loadErr) rows.push(h("div.inlineerr", { role: "alert" }, icon("warn"), h("div.inlineerr__tx", null, h("div.inlineerr__t", null, "Couldn't read your skills"), h("div.inlineerr__d", null, loadErr)), h("button.btn.btn--sm", { type: "button", onclick: () => { loadErr = ""; void load(); } }, "Try again")));
    else if (!skills) rows.push(h("div.sk-group", { "aria-busy": "true" }, h("div.sk.sk--field"), h("div.sk.sk--field")));
    else {
      for (const k of skills.filter((x) => !x.builtin)) rows.push(row({ kind: "skill", name: k.name }, `/${k.name}`, k.description, "skill"));
      for (const k of skills.filter((x) => x.builtin)) rows.push(row({ kind: "skill", name: k.name }, `/${k.name}`, k.description, "built-in", true));
    }
    for (const p of s.prompts) rows.push(row({ kind: "prompt", id: p.id }, p.title || (p.text ? p.text.slice(0, 60) : "Untitled prompt"), p.title ? p.text : "", "prompt"));
    const dir = host.skills.dir;
    const sec = section("Skills and prompts", "Type / in the message box to use them. Pinned ones are listed on a new chat and run from anywhere in Zotero with their shortcut.",
      h("div.sp__list", null, ...rows),
      adding ? preview(adding) : h("div.sp__btns", null,
        h("button.btn.btn--sm", { type: "button", dataset: { fid: "sp:add" }, onclick: () => void pick() }, icon("plus"), "Add skill…"),
        h("button.btn.btn--sm", { type: "button", dataset: { fid: "sp:new" }, onclick: () => { const id = nextId("p") + Date.now().toString(36); editing = `p:${id}`; void o.save({ prompts: addPrompt(host.getSettings().prompts, id) }); } }, icon("plus"), "New prompt"),
        o.createSkill ? h("button.btn.btn--sm", { type: "button", onclick: o.createSkill }, icon("sparkle"), "Create with the agent") : null),
      msg ? h("p.sec__hint.sp__msg", { role: "status" }, msg) : null,
      h("div.sec__sub", { role: "group", "aria-label": "Skills folder" },
        h("span.sec__subt", null, "Skills folder"),
        h("div.folder", { title: dir }, icon("folder"), h("span.sp__dir", null, shortPath(dir, 44))),
        h("div.folder__acts", null, h("button.btn.btn--sm", { type: "button", onclick: () => void host.skills.reveal().catch((e) => say(errMessage(e))) }, isMac() ? "Reveal in Finder" : "Reveal in file manager")),
        hint("Each skill is a folder with a SKILL.md of plain instructions. A new chat copies the skills that are on into its folder, so your agent can read them.")));
    sec.id = "skills";
    card = sec;
    return sec;
  };
}
