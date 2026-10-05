// The Appearance card: glass, accent colour, background (a gradient or the user's own picture), text size, density.
// Every change shows at once: a saved one is applied by `save`, a slider or the colour picker being dragged previews
// through `look.apply(draft)` and saves on release.
import type { Appearance, SettingsHost } from "../types.ts";
import { ACCENTS, BACKGROUNDS } from "./appearance.ts";
import type { Look } from "./appearance.ts";
import { errMessage, h } from "./dom.ts";
import { row, section, seg, slider, swatches, switchRow } from "./settings-parts.ts";

const HEX = /^#[0-9a-f]{6}$/i;

export function appearanceCard(host: SettingsHost, look: Look, save: (patch: Partial<Appearance>) => Promise<void>, render: () => void): () => HTMLElement {
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
      type: "color", value: a.accent || ACCENTS[0]!.light, "aria-label": "Custom colour", title: "Custom colour", dataset: { fid: "accent:custom" },
      oninput: () => look.apply({ ...a, accent: (color as HTMLInputElement).value.toLowerCase() }),
      onchange: () => void save({ accent: (color as HTMLInputElement).value.toLowerCase() }),
    });
    // The hex field: exact colours, and a way in wherever the system colour picker does not open.
    const hex = h("input.input.input--sm.hex", {
      type: "text", value: a.accent || ACCENTS[0]!.light, maxlength: "7", spellcheck: "false", "aria-label": "Accent colour as hex", dataset: { fid: "accent:hex" },
      onchange: () => {
        const v = (hex as HTMLInputElement).value.trim().replace(/^(?!#)/, "#");
        if (HEX.test(v)) void save({ accent: v.toLowerCase() }); else { msg = "A colour is # and six hex digits, like #2563c9."; render(); }
      },
    });
    const tiles = [
      { id: "none", label: "Plain", mod: "none" },
      ...BACKGROUNDS.map((b) => ({ id: b.id, label: b.name, mod: b.id })),
      ...(a.image ? [{ id: "image", label: `Your picture (${a.image})`, mod: "image" }] : []),
    ];
    return section("Appearance", "How the panel looks. Changes show as you make them.",
      switchRow("Glass", "Frosted, translucent cards and menus.", a.glass, (on) => void save({ glass: on })),
      row("Accent colour",
        swatches("Accent colour", "sw", ACCENTS.map((c) => ({ id: c.id, label: c.name, style: `--cl:${c.light};--cd:${c.dark}`, mod: c.id ? undefined : "mono" })), custom ? "custom" : a.accent, (id) => void save({ accent: id }),
          h(`label.sw.sw--custom${custom ? ".sw--on" : ""}`, null, color)),
        "The send button, selections, the working dot and, once you pick a colour, links. Mono is black and white."),
      custom ? row("Hex", hex) : null,
      row("Background",
        [swatches("Background", "bgsw", tiles, a.background, (id) => void save({ background: id })),
          h("div.row__ctl", null,
            h("button.btn.btn--sm", { type: "button", disabled: busy ? true : null, dataset: { fid: "image:choose" }, onclick: () => void chooseImage() }, busy ? "Opening…" : "Choose an image…"),
            a.image ? h("button.btn.btn--sm.btn--quiet", { type: "button", "aria-label": "Remove the background picture", onclick: () => void removeImage() }, "Remove") : null)],
        "Behind the chat. A picture is shrunk and kept in Zotero's profile, never synced."),
      a.background === "image" ? [
        slider("Picture visibility", { value: a.imageVisibility, min: 0, max: 100, unit: "%", help: "Lower fades it into the panel so text stays easy to read.", onInput: (v) => look.apply({ ...a, imageVisibility: v }), onChange: (v) => void save({ imageVisibility: v }) }),
        slider("Picture blur", { value: a.imageBlur, min: 0, max: 20, unit: " px", onInput: (v) => look.apply({ ...a, imageBlur: v }), onChange: (v) => void save({ imageBlur: v }) }),
      ] : null,
      row("Text size", seg("Text size", [{ id: "small", label: "Small" }, { id: "default", label: "Default" }, { id: "large", label: "Large" }], a.textSize, (id) => void save({ textSize: id as Appearance["textSize"] }))),
      row("Density", seg("Density", [{ id: "compact", label: "Compact" }, { id: "comfortable", label: "Comfortable" }], a.density, (id) => void save({ density: id as Appearance["density"] }))),
      msg ? h("p.sec__hint", { role: "alert" }, msg) : null);
  };
}
