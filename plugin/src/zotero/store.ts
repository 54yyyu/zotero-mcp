// Chat history: <dir>/index.json (the list) and <dir>/<id>.jsonl (one ChatEvent per line, replayed through the same
// reducer that rendered it live). Lives in the profile, not the Zotero data dir, so it is never synced or backed up with the library.
import type { ChatEvent, SavedSession } from "../types.ts";

export function createStore(dir: string) {
  const index = PathUtils.join(dir, "index.json");
  const file = (id: string) => PathUtils.join(dir, `${id.replace(/[^A-Za-z0-9_-]/g, "_")}.jsonl`);
  let chain: Promise<unknown> = Promise.resolve();
  const serial = <T>(fn: () => Promise<T>): Promise<T> => {
    const run = chain.then(fn, fn);
    chain = run.catch(() => {});
    return run;
  };

  async function readIndex(): Promise<SavedSession[]> {
    try {
      return JSON.parse(await IOUtils.readUTF8(index)).sessions ?? [];
    } catch {
      return [];
    }
  }

  return {
    sessions: () => serial(async () => (await readIndex()).sort((a, b) => b.updatedAt - a.updatedAt)),

    async loadEvents(id: string): Promise<ChatEvent[]> {
      try {
        const text = await IOUtils.readUTF8(file(id));
        return text.split("\n").filter(Boolean).flatMap((l: string) => { try { return [JSON.parse(l) as ChatEvent]; } catch { return []; } });
      } catch {
        return [];
      }
    },

    appendEvent: (session: SavedSession, ev: ChatEvent) => serial(async () => {
      await IOUtils.makeDirectory(dir, { createAncestors: true, ignoreExisting: true });
      await IOUtils.writeUTF8(file(session.id), JSON.stringify(ev) + "\n", { mode: "appendOrCreate" });
      const all = await readIndex();
      const i = all.findIndex((s) => s.id === session.id);
      const entry = { ...session, updatedAt: Date.now() };
      if (i >= 0) all[i] = entry; else all.push(entry);
      await IOUtils.writeUTF8(index, JSON.stringify({ sessions: all }));
    }),

    clearAll: () => serial(async () => { await IOUtils.remove(dir, { recursive: true, ignoreAbsent: true }); }),

    deleteSession: (id: string) => serial(async () => {
      await IOUtils.remove(file(id), { ignoreAbsent: true });
      await IOUtils.writeUTF8(index, JSON.stringify({ sessions: (await readIndex()).filter((s) => s.id !== id) }));
    }),
  };
}
