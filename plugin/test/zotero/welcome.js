// The first run in real Gecko (run with --mock-agent, and no saved settings): the welcome shows instead of the chat, the
// checks settle, an agent can be chosen, and Start chatting (or Skip) hands over to the chat and remembers it.
async function main(ctx) {
  const { Zotero, win } = ctx;
  const out = {};
  const check = (cond, msg) => { if (!cond) throw new Error("FAILED: " + msg); };
  const injected = ctx.plugin.windows.get(win);

  await ctx.resize(1280, 900);
  injected.show();
  const root = await ctx.waitFor(() => win.document.getElementById("zmc-root")?.shadowRoot?.querySelector(".wel") && win.document.getElementById("zmc-root").shadowRoot, "welcome rendered");
  const host = ctx.plugin.panel().host;
  const $ = (sel) => root.querySelector(sel);
  const $$ = (sel) => [...root.querySelectorAll(sel)];
  check(host.getSettings().welcomed === false, "a fresh profile has not been welcomed");
  check(!$(".cin") || $(".chat").hidden, "the chat is not showing under the welcome");
  await ctx.snapshot("welcome-1-start");

  // the checks run for real (curl to the local API, node, the CLI) and every row settles
  await ctx.waitFor(() => $$(".wrow").length >= 4 && !$(".wrow--pending") && !$(".sk"), "checks settled", 30000);
  out.rows = $$(".wrow").map((r) => r.querySelector(".wrow__t")?.textContent + ":" + (r.classList.contains("wrow--ok") ? "ok" : "bad"));
  await ctx.snapshot("welcome-2-checked");

  // the agent cards: three, one chosen, choosing another is saved
  check($$(".wcard").length === 3 && $$(".wcard[aria-checked=true]").length === 1, "three cards, one chosen");
  const before = host.getSettings().backend;
  const other = $$(".wcard").find((c) => c.getAttribute("aria-checked") === "false");
  other.click();
  await ctx.waitFor(() => host.getSettings().backend !== before, "choice saved");
  $$(".wcard").find((c) => c.textContent.includes(before === "claude-code" ? "Claude Code" : "pi")).click();
  await ctx.waitFor(() => host.getSettings().backend === before, "choice restored");
  out.cards = "ok";

  // hand over: Start chatting when everything is ready, otherwise Skip setup; both finish the welcome for good
  await ctx.waitFor(() => !$(".wrow--pending"), "rechecked");
  const go = $(".wfoot .btn--solid") || $(".wfoot__skip");
  go.click();
  await ctx.waitFor(() => $("textarea") && !$(".chat").hidden && !$(".wel"), "the chat is showing");
  check(host.getSettings().welcomed === true, "welcomed is saved");
  check(JSON.parse(Zotero.Prefs.get("extensions.zotero-chat.settings", true)).welcomed === true, "welcomed reached the prefs");
  await ctx.snapshot("welcome-3-chat");
  out.handover = "ok";
  return out;
}
