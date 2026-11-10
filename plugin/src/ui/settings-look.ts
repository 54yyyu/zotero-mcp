// The Appearance card: glass, accent colour, background (a gradient or the user's own picture), text size, density.
// Every change shows at once: a saved one is applied by `save`, a slider or the colour picker being dragged previews
// through `look.apply(draft)` and saves on release.
import type { Appearance, SettingsHost } from "../types.ts";
import { ACCENTS, BACKGROUNDS } from "./appearance.ts";
import type { Look } from "./appearance.ts";
import { errMessage, h } from "./dom.ts";
import { row, section, seg, slider, swatches, switchRow } from "./settings-parts.ts";
import type { Kid } from "./dom.ts";
import { t } from "../i18n.ts";

const HEX = /^#[0-9a-f]{6}$/i;

/** `first`: a row drawn at the top of the card (the interface language, which is saved with the other settings). */
export function appearanceCard(host: SettingsHost, look: Look, save: (patch: Partial<Appearance>) => Promise<void>, render: () => void, first: () => Kid): () => HTMLElement {
  let msg = "";
  let busy = false;
  const fail = (e: unknown) => { msg = errMessage(e); render(); };

  const chooseImage = async () => {
    busy = true; msg = ""; render();
    try {
      const got = await host.chooseImage();
      if (got) { look.setImage(got.dataUrl); await save({ background: "image", image: got.name }); }
    } catch (e) { msg = errMessage(e); }
    busy = false; render();
  };
  const removeImage = async () => {
    const a = host.getSettings().appearance;
    try { await host.removeImage(); look.setImage(null); await save({ image: "", background: a.background === "image" ? "none" : a.background }); } catch (e) { fail(e); }
  };

  return () => {
    const a = host.getSettings().appearance;
    const custom = !!a.accent && !ACCENTS.some((x) => x.id === a.accent);
    const color = h("input", {
      type: "color", value: a.accent || ACCENTS[0]!.light, "aria-label": t("look.custom"), title: t("look.custom"), dataset: { fid: "accent:custom" },
      oninput: () => look.apply({ ...a, accent: (color as HTMLInputElement).value.toLowerCase() }),
      onchange: () => void save({ accent: (color as HTMLInputElement).value.toLowerCase() }),
    });
    // The hex field: exact colours, and a way in wherever the system colour picker does not open.
    const hex = h("input.input.input--sm.hex", {
      type: "text", value: a.accent || ACCENTS[0]!.light, maxlength: "7", spellcheck: "false", "aria-label": t("look.hex"), dataset: { fid: "accent:hex" },
      onchange: () => {
        const v = (hex as HTMLInputElement).value.trim().replace(/^(?!#)/, "#");
        if (HEX.test(v)) void save({ accent: v.toLowerCase() }); else { msg = t("look.hexBad"); render(); }
      },
    });
    const tiles = [
      { id: "none", label: t("bg.plain"), mod: "none" },
      ...BACKGROUNDS.map((b) => ({ id: b.id, label: t(b.name), mod: b.id })),
      ...(a.image ? [{ id: "image", label: t("look.yourPicture", { name: a.image }), mod: "image" }] : []),
    ];
    const sec = section(t("look.title"), t("look.desc"),
      first(),
      switchRow(t("look.glass"), t("look.glassHint"), a.glass, (on) => void save({ glass: on })),
      row(t("look.accent"),
        swatches(t("look.accent"), "sw", ACCENTS.map((c) => ({ id: c.id, label: t(c.name), style: `--cl:${c.light};--cd:${c.dark}`, mod: c.id ? undefined : "mono" })), custom ? "custom" : a.accent, (id) => void save({ accent: id }),
          h(`label.sw.sw--custom${custom ? ".sw--on" : ""}`, null, color)),
        t("look.accentHint")),
      custom ? row(t("look.hexLabel"), hex) : null,
      row(t("look.background"),
        [swatches(t("look.background"), "bgsw", tiles, a.background, (id) => void save({ background: id })),
          h("div.row__ctl", null,
            h("button.btn.btn--sm", { type: "button", disabled: busy ? true : null, dataset: { fid: "image:choose" }, onclick: () => void chooseImage() }, busy ? t("look.opening") : t("look.chooseImage")),
            a.image ? h("button.btn.btn--sm.btn--quiet", { type: "button", "aria-label": t("look.removeImage"), onclick: () => void removeImage() }, t("common.remove")) : null)],
        t("look.backgroundHint")),
      a.background === "image" ? [
        slider(t("look.visibility"), { value: a.imageVisibility, min: 0, max: 100, unit: "%", help: t("look.visibilityHint"), onInput: (v) => look.apply({ ...a, imageVisibility: v }), onChange: (v) => void save({ imageVisibility: v }) }),
        slider(t("look.blur"), { value: a.imageBlur, min: 0, max: 20, unit: " px", onInput: (v) => look.apply({ ...a, imageBlur: v }), onChange: (v) => void save({ imageBlur: v }) }),
      ] : null,
      row(t("look.textSize"), seg(t("look.textSize"), [{ id: "small", label: t("look.small") }, { id: "default", label: t("common.default") }, { id: "large", label: t("look.large") }], a.textSize, (id) => void save({ textSize: id as Appearance["textSize"] }))),
      row(t("look.density"), seg(t("look.density"), [{ id: "compact", label: t("look.compact") }, { id: "comfortable", label: t("look.comfortable") }], a.density, (id) => void save({ density: id as Appearance["density"] }))),
      msg ? h("p.sec__hint", { role: "alert" }, msg) : null);
    sec.id = "look"; // a change of language redraws the panel and comes back here
    return sec;
  };
}
