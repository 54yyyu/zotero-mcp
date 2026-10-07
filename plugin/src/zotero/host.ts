// PanelHost: what ui/ gets. This is the only place that knows both the UI's needs and Zotero's APIs.
import type { AgentRuntime, BackendId, Catalog, ContextChip, PanelHost, Spawner } from "../types.ts";
import { DRAWING_GUIDE, FORMAT_GUIDE, TOOL_SHEET, buildBrief, createRuntime, prepareWorkspace, resumeCommand } from "../agent/index.ts";
import { ContextTracker } from "./context.ts";
import { describeContext } from "./describe.ts";
import { dropChips } from "./drop.ts";
import { createDoctor, findCli } from "./doctor.ts";
import * as keychain from "./keychain.ts";
import { saveNote } from "./note.ts";
import { pageChip, pageHit } from "./page.ts";
import { createPapers } from "./paper.ts";
import { openTarget } from "./open.ts";
import { chipForHit, search } from "./search.ts";
import { createGeckoSpawner } from "./spawn-gecko.ts";
import { createSettingsHost } from "./settings-host.ts";

export interface HostBundle {
  host: PanelHost;
  spawner: Spawner;
  context: ContextTracker;
  /** chooseImage without the picker: the in-Zotero test imports a file through it. */
  importImage(path: string): Promise<{ name: string; dataUrl: string }>;
  /** The user is about to write (the composer has focus): start preparing the focused PDF's text. */
  prefetchPaper(): void;
  /** A catalog this panel has read, for Zotero's Settings pane, which must not start an agent to show one. */
  knownCatalog(b: BackendId): Catalog | undefined;
  dispose(): void;
}

export function createHost(opts: { id: string; version: string; win: any; dataDir: string }): HostBundle {
  const { win, dataDir } = opts;
  const sh = createSettingsHost(opts);
  const { settings, store, images, skills, chatFolder, defaultFolder } = sh;
  const spawner = createGeckoSpawner();
  const context = new ContextTracker(opts.id);
  context.start(win);
  const base = createRuntime({ spawner, bridgeDir: PathUtils.join(dataDir, "bridges") });
  const known = new Map<BackendId, Catalog>();
  // Wherever zotero-cli is, the agent's shell must find it: its folder goes on the agent's PATH.
  const runtime: AgentRuntime = {
    ...base,
    catalog: (b) => base.catalog(b).then((c) => { known.set(b, c); return c; }),
    async start(o) {
      const cli = await findCli(spawner, await spawner.baseEnv());
      return base.start(cli ? { ...o, path: [cli.slice(0, cli.lastIndexOf("/")), ...(o.path ?? [])] } : o);
    },
  };
  const doctor = createDoctor({ win, spawner, runtime, settings });
  const papers = createPapers({ dataDir, defaultFolder, chatFolder });
  // The context settings decide what the tracker hands over, wherever they were changed.
  const offSettings = sh.host.onSettingsChange(() => context.refresh());

  const host: PanelHost = {
    ...sh.host,
    runtime,
    currentContext() {
      const s = settings();
      if (!s.followFocus) return [];
      return context.current(win)
        .filter((c) => s.attachSelection || c.kind !== "selection")
        .map((c) => (c.kind === "area" && !s.attachAreas ? { ...c, image: undefined } : c));
    },
    onContextChange: (cb) => context.on(cb),
    async search(q) {
      const reader = context.activeReader(win);
      const page = reader ? pageHit(reader, q) : null;
      const hits = await search(win, q, reader?._item ?? null);
      return page ? [page, ...hits] : hits;
    },
    async chipFor(hit) {
      if (hit.kind === "page") return pageChip(win, context.activeReader(win), hit);
      let text: string | undefined;
      if (hit.kind === "annotation" && hit.ref.annotationKey) {
        const a = Zotero.Items.getByLibraryAndKey(hit.ref.libraryID, hit.ref.annotationKey);
        text = a?.annotationText || a?.annotationComment || undefined;
      }
      return chipForHit(hit, text);
    },
    open: (target) => openTarget(win, target),
    dropChips: (data) => dropChips(context, data),
    paperContext: (chips, ready) => papers.context(chips, ready),
    describeContext: (chips: ContextChip[]) => describeContext(chips),

    async setSettings(patch) { await sh.host.setSettings(patch); context.refresh(); },
    async resetSettings() { await sh.host.resetSettings(); context.refresh(); },

    sessions: () => store.sessions(),
    loadEvents: (id) => store.loadEvents(id),
    appendEvent: (s, ev) => store.appendEvent(s, ev),
    deleteSession: (id) => store.deleteSession(id),
    async saveFile(name, data, mime) {
      const fp = Cc["@mozilla.org/filepicker;1"].createInstance(Ci.nsIFilePicker);
      const ext = /\.([a-z0-9]{1,5})$/i.exec(name)?.[1]?.toLowerCase() ?? "";
      fp.init(win.browsingContext, "Save", Ci.nsIFilePicker.modeSave);
      fp.defaultString = name;
      if (ext) { fp.appendFilter(`${ext.toUpperCase()} (${mime})`, `*.${ext}`); fp.defaultExtension = ext; }
      const result: number = await new Promise((r) => fp.open(r));
      if (result !== Ci.nsIFilePicker.returnOK && result !== Ci.nsIFilePicker.returnReplace) return null;
      let path: string = fp.file.path;
      if (ext && !path.toLowerCase().endsWith(`.${ext}`)) path += `.${ext}`;
      if (typeof data === "string") await IOUtils.writeUTF8(path, data);
      else await IOUtils.write(path, data);
      return path;
    },
    saveNote: (note) => saveNote(win, context, note),
    resumeCommand: (s) => resumeCommand(s.backend, s.cwd, s.agentSessionId),

    doctor,
    async prepareSession(resumeIn) {
      const env = await spawner.baseEnv();
      const zoteroCli = await findCli(spawner, env);
      // A chat being resumed runs in the folder it started in; agents look a session up by its folder.
      const dir = resumeIn || chatFolder();
      const cwd = await prepareWorkspace(spawner, dir, { ...(zoteroCli ? { zoteroCli } : {}), refresh: dir === defaultFolder() });
      const s = settings();
      // The skills that are on go into the agent's folder (Claude and Codex also find them there by themselves).
      await skills.sync(cwd, (await skills.host.list()).filter((k) => !s.skills?.[k.name]?.off).map((k) => k.name)).catch((e) => Zotero.logError(e));
      const key = s.auth[s.backend] === "api-key" ? await keychain.getApiKey(s.backend) : null;
      const varName = keychain.API_KEY_ENV[s.backend];
      return { cwd, brief: `${buildBrief()}\n\n${TOOL_SHEET}\n\n${DRAWING_GUIDE}\n\n${FORMAT_GUIDE}`, env: key && varName ? { [varName]: key } : {} };
    },
  };
  return { host, spawner, context, prefetchPaper: () => papers.prefetch(host.currentContext()), importImage: images.importImage, knownCatalog: (b) => known.get(b), dispose: () => { offSettings(); sh.dispose(); context.stop(); } };
}
