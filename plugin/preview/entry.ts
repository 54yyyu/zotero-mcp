// The preview page: the panel over a FakeHost. URL parameters pick the scenario:
//   ?theme=dark  &doctor=ok|zotero-api|write-access|cli|node|backend|many
//   &ctx=item|selection|area|none  &speed=0  &stress=1  &nohistory=1  &welcome=1
//   &look={"glass":false,"background":"dawn",...}  (an appearance laid over the default)  &image=1 (a saved background picture)
// `window.__zmc` exposes the host's `sim` and the panel handle to Playwright.
import { mountPanel } from "../src/ui/index.ts";
import { FakeHost } from "../src/ui/fake-host.ts";
import type { DoctorMode } from "../src/ui/fake-host.ts";

const q = new URLSearchParams(location.search);
const host = new FakeHost({
  theme: q.get("theme") === "dark" ? "dark" : "light",
  doctor: (q.get("doctor") as DoctorMode | null) ?? "ok",
  context: (q.get("ctx") as "item" | "selection" | "area" | "none" | null) ?? "item",
  speed: q.has("speed") ? Number(q.get("speed")) : 18,
  catalogDelay: q.has("catalogDelay") ? Number(q.get("catalogDelay")) : undefined,
  stress: q.has("stress"),
  noHistory: q.has("nohistory"),
  welcome: q.has("welcome"),
});
if (q.has("image")) host.sim.image = host.sim.pickImage?.dataUrl ?? null; // a picture saved by an earlier session
if (q.has("look")) void host.setSettings({ appearance: { ...host.getSettings().appearance, ...JSON.parse(q.get("look")!) } });
const mount = document.getElementById("host") as HTMLElement;
const shadow = mount.attachShadow({ mode: "open" });
const panel = mountPanel(shadow, host);
mount.addEventListener("zmc-close", () => { mount.dataset.closed = "1"; });
(window as unknown as { __zmc: unknown }).__zmc = { host, sim: host.sim, panel, shadow };
