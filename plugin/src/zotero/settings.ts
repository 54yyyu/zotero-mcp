// Prefs under extensions.zotero-chat.*. Secrets never go here (see keychain.ts).
export const PREFIX = "extensions.zotero-chat.";

export const prefs = {
  get(key: string): any {
    return Zotero.Prefs.get(PREFIX + key, true);
  },
  set(key: string, value: unknown): void {
    Zotero.Prefs.set(PREFIX + key, value, true);
  },
  json<T>(key: string, fallback: T): T {
    try {
      const raw = prefs.get(key);
      return raw ? (JSON.parse(raw) as T) : fallback;
    } catch {
      return fallback;
    }
  },
  setJson(key: string, value: unknown): void {
    prefs.set(key, JSON.stringify(value));
  },
};
