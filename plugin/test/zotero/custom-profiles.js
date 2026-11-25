// Drive the packaged add-on in two Zotero processes with the same isolated preferences.
async function main(ctx) {
  const { Zotero, Services, PathUtils, plugin, win } = ctx;
  const settingsKey = "extensions.zotero-chat.settings";
  const restart = Services.prefs.getStringPref("extensions.zotero-chat.testCustomRestart", "") === "1";
  const saved = () => JSON.parse(Services.prefs.getStringPref(settingsKey, "{}"));
  const check = (value, reason) => { if (!value) throw new Error(reason); };
  const mock = PathUtils.join(PathUtils.parent(PathUtils.parent(ctx.env("ZMC_TEST_SCRIPT"))), "mock-agent.mjs");
  const a = { label: "Local Literature", command: "node", args: [mock, "legacy arg"], env: { MOCK_VARIANT: "claude" } };
  const b = { label: "Code Review ACP", command: "node", args: [mock, "review arg"], env: { MOCK_VARIANT: "codex" } };
  const bModel = "gpt-6-luna[high]";
  const pick = (root, selector, text) => {
    const button = [...root.querySelectorAll(selector)].find((el) => (el.querySelector(".radio__t") ?? el).textContent.trim() === text);
    check(button, `Missing button "${text}" in ${selector}`);
    button.click();
  };
  const set = (input, value) => {
    check(input, `Missing input for "${value}"`);
    input.value = value;
    input.dispatchEvent(new input.ownerDocument.defaultView.Event(input.localName === "select" ? "change" : "input", { bubbles: true }));
  };
  if (!restart) {
    Services.prefs.setStringPref(settingsKey, JSON.stringify({
      welcomed: true, uiLanguage: "en", backend: "custom", customAgent: a,
      model: { custom: "sonnet" }, mode: { custom: "plan" }, effort: { custom: "high" },
    }));
  }
  const pw = Zotero.Utilities.Internal.openPreferences("zotero-chat-pane");
  const pane = await ctx.waitFor(() => {
    const root = pw.document.getElementById("zmc-prefs")?.shadowRoot;
    return root?.querySelector(".set__agent-tabs") && root;
  }, "Zotero's Agent settings pane");
  const field = (key) => pane.querySelector(`[data-fid="custom:${key}"]`);
  const active = () => pane.querySelector(".set__agent-tabs [aria-checked=true]")?.textContent;
  const tab = (label) => pick(pane, ".set__agent-tabs .seg__opt", label);
  const button = (label) => pick(pane, "button", label);
  const fill = (profile) => {
    set(field("label"), profile.label);
    set(field("command"), profile.command);
    set(field("args"), JSON.stringify(profile.args));
    set(field("env"), JSON.stringify(profile.env));
  };
  const model = () => pane.querySelector('section[aria-label="Agent"] select[aria-label="Model"]');
  const show = async (label, modelId) => {
    if (!model()) button(`Show ${label}'s options`);
    await ctx.waitFor(() => model()?.querySelector(`option[value="${modelId}"]`), `${label}'s own ACP models`);
  };
  const checkB = (s) => {
    const id = Object.keys(s.customAgents ?? {}).find((id) => s.customAgents[id].label === b.label);
    check(id && s.backend === id, "The second custom agent is the selected backend");
    check(s.customAgents.custom.label === "Literature ACP", "Renamed legacy profile persisted");
    check(JSON.stringify(s.customAgents[id]) === JSON.stringify(b), "Saved command, argv and environment match the second profile");
    check(s.model[id] === bModel && s.mode[id] === "read-only", "Second profile kept its own model and mode");
    check(s.model.custom === "sonnet" && s.mode.custom === "plan" && s.effort.custom === "high", "Legacy model, mode and effort migrated");
    check(!("customAgent" in s), "Only the migrated settings schema remains");
    return id;
  };

  if (restart) {
    const id = checkB(saved());
    check(active() === b.label && field("command").value === b.command, "Zotero reloaded the selected profile into the UI");
    await show(b.label, bModel);
    check(model().value === bModel, "The second profile's model is selected after restart");
    tab("Literature ACP");
    await ctx.waitFor(() => active() === "Literature ACP", "first tab after restart");
    check(JSON.parse(field("args").value).at(-1) === "legacy arg", "The first profile is still selectable after restart");
    pw.close();
    return { restart: true, selectedAtStartup: id, profilesRestored: 2 };
  }

  check(active() === a.label && field("command").value === a.command, "Old Custom preferences populate a named tab");
  set(field("label"), "Literature ACP");
  button("Save custom agent");
  await ctx.waitFor(() => active() === "Literature ACP", "renamed legacy agent");
  check(saved().model.custom === "sonnet", "Renaming a profile preserves its model choice");
  ctx.log("Legacy profile migrated");

  button("Add custom agent");
  fill(b);
  set(field("args"), "{");
  button("Save custom agent");
  check(Object.keys(saved().customAgents).length === 1 && field("args").value === "{", "Invalid JSON leaves the draft and saved profiles alone");
  set(field("args"), JSON.stringify(b.args));
  button("Save custom agent");
  await ctx.waitFor(() => active() === b.label, "the second named tab");
  const second = saved().backend;
  await show(b.label, bModel);
  set(model(), bModel);
  pick(pane, "section[aria-label=Agent] .radio", "Read-only");
  await ctx.waitFor(() => saved().model[second] === bModel && saved().mode[second] === "read-only", "independent preferences of the second profile");
  checkB(saved());

  // An endpoint change clears only that profile's selections; a label change did not.
  set(field("args"), JSON.stringify([...b.args, "--changed"]));
  button("Save custom agent");
  await ctx.waitFor(() => !saved().model[second], "changed endpoint clears its model selection");
  check(!saved().mode[second] && saved().model.custom === "sonnet", "Changed endpoint clears only its own choices");
  set(field("args"), JSON.stringify(b.args));
  button("Save custom agent");
  await show(b.label, bModel);
  set(model(), bModel);
  pick(pane, "section[aria-label=Agent] .radio", "Read-only");
  await ctx.waitFor(() => saved().model[second] === bModel && saved().mode[second] === "read-only", "second profile preferences restored");
  ctx.log("Both ACP configurations and their options saved independently");

  button("Add custom agent");
  fill({ ...a, label: "Discard draft" });
  button("Cancel");
  check(active() === b.label && field("label").value === b.label, "Cancel returns to the previous profile");
  button("Add custom agent");
  check(!field("label").value, "A cancelled draft does not return");
  fill({ ...a, label: "Temporary ACP" });
  button("Save custom agent");
  await ctx.waitFor(() => active() === "Temporary ACP", "temporary profile");
  pick(pane, ".custom__actions button", "Delete");
  pick(pane, ".custom__actions button", "Cancel");
  check(active() === "Temporary ACP", "Cancelling deletion keeps the profile");
  pick(pane, ".custom__actions button", "Delete");
  pick(pane, ".custom__actions button.btn--danger", "Delete");
  await ctx.waitFor(() => ![...pane.querySelectorAll(".set__agent-tabs .seg__opt")].some((el) => el.textContent === "Temporary ACP"), "deleted tab disappears");
  tab(b.label);
  await ctx.waitFor(() => active() === b.label, "second profile selected again");
  checkB(saved());
  ctx.log("Add cancellation and profile deletion verified");

  const { host, bundle } = plugin.panel();
  const run = bundle.spawner.run.bind(bundle.spawner);
  // The chat still starts through Gecko. Skip the unrelated skill installation, which can download zotero-cli via uv.
  bundle.spawner.run = (cmd, args, opts) => args.includes("install-skill") && opts.cwd === ctx.env("ZMC_DEFAULT_CHAT_FOLDER")
    ? Promise.resolve({ code: 0, stdout: "", stderr: "" })
    : run(cmd, args, opts);
  plugin.windows.get(win).show();
  const root = await ctx.waitFor(() => {
    const shadow = win.document.getElementById("zmc-root")?.shadowRoot;
    return shadow?.querySelector("textarea.cin") && shadow;
  }, "real chat panel");
  const send = async (id, n, expected) => {
    set(root.querySelector("textarea.cin"), "SCENARIO:whoami");
    const send = await ctx.waitFor(() => {
      const el = root.querySelector("button.send:not(:disabled)");
      return el && !el.classList.contains("send--stop") && el;
    }, "chat enabled for the selected custom agent");
    send.click();
    await ctx.waitFor(async () => {
      const chat = (await host.sessions()).find((s) => s.backend === id);
      if (!chat) return;
      const events = await host.loadEvents(chat.id);
      const error = events.find((ev) => ev.t === "notice" && ev.level === "error");
      if (error) throw new Error(error.message);
      const turns = events.filter((ev) => ev.t === "turn_end" && ev.stop === "end_turn");
      return turns.length === n
        && events.filter((ev) => ev.t === "text" && ev.turn === turns.at(-1).turn).map((ev) => ev.delta).join("").includes(expected);
    }, `a reply from ${id} with ${expected}`);
  };
  await send(second, 1, `"model":"${bModel}","mode":"read-only"`);
  tab("Literature ACP");
  await ctx.waitFor(() => host.getSettings().backend === "custom", "panel follows the Settings pane's default");
  await send(second, 2, `"model":"${bModel}","mode":"read-only"`);
  check(root.querySelector(".stat__t").textContent === b.label, "The already running chat keeps its agent");

  root.querySelector(".pick--model").click();
  await ctx.waitFor(() => root.querySelectorAll(".mdd__agent").length === 6, "both custom agents in the model menu");
  pick(root, ".mdd__agent", "Literature ACP");
  await ctx.waitFor(() => root.querySelector(".mdd__note button.btn--solid"), "switch asks for a new chat");
  root.querySelector(".mdd__note button.btn--solid").click();
  await ctx.waitFor(() => root.querySelector(".stat__t")?.textContent === "Literature ACP", "the new chat uses the first agent");
  await send("custom", 1, '"model":"sonnet","mode":"plan","effort":"high"');
  const sessions = await host.sessions();
  check(sessions.some((s) => s.backend === second && s.backendLabel === b.label), "The previous chat is still saved under its custom name");
  check(sessions.length === 2, "Switching in chat keeps both conversations");
  root.querySelector('button[aria-label="History"]').click();
  const old = await ctx.waitFor(() => [...root.querySelectorAll(".hrow__main")].find((el) => el.querySelector(".hrow__m").textContent.includes(b.label)), "the previous custom chat in history");
  old.click();
  await ctx.waitFor(() => root.querySelector(".zmc").dataset.view === "chat", "read the previous transcript");
  set(root.querySelector("textarea.cin"), "Archived chat");
  check(root.querySelector("button.send").disabled && root.querySelector(".stat__t").textContent === b.label, "The saved custom chat is readable under its original name");
  ctx.log("Both ACP processes answered using their own options");
  tab(b.label);
  await ctx.waitFor(() => active() === b.label, "second agent selected for the restart");
  checkB(saved());
  pw.close();
  return { migrated: true, profiles: 2, isolatedChoices: true, answeredThroughEachAgent: true, selected: second };
}
