/* Run by Zotero's Settings window (in the pane's sandbox) before the pane's markup is inserted. The plugin mounts the
   settings when Zotero sends the pane its "load" event; a capturing listener on the document sees it on the way down. */
document.addEventListener("load", function mount(e) {
  if (e.target.id !== "zmc-prefs") return;
  document.removeEventListener("load", mount, true);
  Services.obs.notifyObservers({ wrappedJSObject: { root: e.target, win: window } }, "zotero-chat:prefpane");
}, true);
