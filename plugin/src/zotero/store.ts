// Chat history: <dir>/index.json (the list) and <dir>/<id>.jsonl (one ChatEvent per line, replayed through the same
// reducer that rendered it live). Lives in the profile, not the Zotero data dir, so it is never synced or backed up with the library.
//
// The index is the only file that is rewritten, so it is the one that must survive a crash: it is written to a temporary
// file and moved into place, and a list that cannot be read is set aside, never written over. It is rewritten when a
// chat's entry says something new (a new chat, its title, its agent session) and when a message is sent or a turn ends
// (the "last used" time the history sorts by), not for every event of a streaming answer.
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
  /** Each chat's entry as this store last wrote it (without the time): an event that changes none of it leaves the index alone. */
  const written = new Map<string, string>();
  const entryOf = (s: SavedSession) => JSON.stringify({ ...s, updatedAt: 0 });

  async function readIndex(): Promise<SavedSession[]> {
    let text: string;
    try {
      text = await IOUtils.readUTF8(index);
    } catch {
      return []; // no list yet
    }
    try {
      const sessions: unknown = JSON.parse(text).sessions;
      if (Array.isArray(sessions)) return sessions as SavedSession[];
    } catch { /* below */ }
    // Not a list we wrote whole (a crash mid-write, before writes were atomic; a disk error): keep it for whoever wants to
    // look, rather than replace every chat's entry with the one being saved now.
    await IOUtils.move(index, `${index}.unreadable-${Date.now()}`).catch(() => {});
    return [];
  }

  const writeIndex = (sessions: SavedSession[]) => IOUtils.writeUTF8(index, JSON.stringify({ sessions }), { tmpPath: `${index}.tmp` });

  async function append(path: string, line: string): Promise<void> {
    try {
      await IOUtils.writeUTF8(path, line, { mode: "appendOrCreate" });
    } catch {
      // The folder is not there (the first chat, or history was cleared): make it, once.
      await IOUtils.makeDirectory(dir, { createAncestors: true, ignoreExisting: true });
      await IOUtils.writeUTF8(path, line, { mode: "appendOrCreate" });
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
      await append(file(session.id), JSON.stringify(ev) + "\n");
      const entry = entryOf(session);
      if (ev.t !== "user" && ev.t !== "turn_end" && written.get(session.id) === entry) return;
      const all = await readIndex();
      const i = all.findIndex((s) => s.id === session.id);
      const next = { ...session, updatedAt: Date.now() };
      if (i >= 0) all[i] = next; else all.push(next);
      await writeIndex(all);
      written.set(session.id, entry);
    }),

    clearAll: () => serial(async () => {
      written.clear();
      await IOUtils.remove(dir, { recursive: true, ignoreAbsent: true });
    }),

    deleteSession: (id: string) => serial(async () => {
      written.delete(id);
      await IOUtils.remove(file(id), { ignoreAbsent: true });
      await writeIndex((await readIndex()).filter((s) => s.id !== id));
    }),

    /** A user rename: the new title and the lock that stops every automatic source. */
    renameSession: (id: string, title: string) => serial(async () => {
      const all = await readIndex();
      const i = all.findIndex((s) => s.id === id);
      const cur = i >= 0 ? all[i] : undefined;
      if (!cur) return;
      all[i] = { ...cur, title, titleLocked: true, updatedAt: Date.now() };
      written.delete(id);
      await writeIndex(all);
    }),
  };
}
