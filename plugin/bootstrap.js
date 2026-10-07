/* Loads plugin.js (built from src/zotero/plugin.ts) and forwards the bootstrap hooks to it. */
var ZoteroChatPlugin;

async function startup({ id, version, rootURI }) {
  try {
    await Zotero.initializationPromise;
    await Zotero.uiReadyPromise;
    let t0 = Date.now();
    let scope = { Zotero, Services, ChromeUtils, IOUtils, PathUtils, Components, Cc: Components.classes, Ci: Components.interfaces };
    Services.scriptloader.loadSubScript(rootURI + "plugin.js", scope);
    ZoteroChatPlugin = scope.ZoteroChat.createPlugin({ id, version, rootURI });
    ZoteroChatPlugin.timing.loadMs = Date.now() - t0;
    await ZoteroChatPlugin.startup();
  }
  catch (e) {
    Zotero.logError(e);
    // The dev harness waits for a result file; a failed startup must say so instead of timing out.
    let out = Services.env.get("ZMC_TEST_OUT");
    if (out) await IOUtils.writeUTF8(out, JSON.stringify({ ok: false, error: "startup failed: " + e + "\n" + (e.stack || "") }));
  }
}

function onMainWindowLoad({ window }) {
  ZoteroChatPlugin?.onMainWindowLoad(window);
}

function onMainWindowUnload({ window }) {
  ZoteroChatPlugin?.onMainWindowUnload(window);
}

async function shutdown() {
  if (ZoteroChatPlugin) {
    await ZoteroChatPlugin.shutdown();
    ZoteroChatPlugin = undefined;
  }
}

function install() {}
function uninstall() {}
