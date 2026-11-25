import type { SavedSession } from "../types.ts";
import type { Chat } from "./chat.ts";

interface ConversationDeps {
  makeChat(onChange: () => void): Chat;
  limit(): number;
  foregroundChanged(): void;
  backgroundChanged(): void;
}

/** Owns the chats in one panel. Only busy, saved chats can stay live offscreen. */
export class Conversations {
  current: Chat;
  private background = new Map<string, Chat>();
  private navigation = 0;
  private disposed = false;
  private closing: Promise<void> = Promise.resolve();
  private deps: ConversationDeps;

  constructor(deps: ConversationDeps) {
    this.deps = deps;
    this.current = this.create();
  }

  isCurrent(chat: Chat): boolean { return !this.disposed && this.current === chat; }
  isBackground(id: string): boolean { return this.background.has(id); }
  waitForClose(): Promise<void> { return this.closing; }

  newChat(): void {
    this.navigation++;
    this.switchTo(this.create());
  }

  async open(saved: SavedSession): Promise<boolean> {
    const navigation = ++this.navigation;
    if (this.current.saved?.id === saved.id) return true;
    const live = this.background.get(saved.id);
    if (live) {
      this.background.delete(saved.id);
      this.switchTo(live);
      return true;
    }
    const chat = this.create();
    await chat.load(saved);
    if (navigation !== this.navigation) { this.close(chat); return false; }
    this.switchTo(chat);
    return true;
  }

  rename(saved: SavedSession): void {
    this.current.adoptRename(saved.id, saved.title);
    this.background.get(saved.id)?.adoptRename(saved.id, saved.title);
  }

  /** Stop writing before the history view queues the deletion. */
  remove(id: string): boolean {
    this.navigation++;
    const current = this.current.saved?.id === id;
    const chat = current ? this.current : this.background.get(id);
    if (!chat) return false;
    if (current) this.current = this.create();
    this.background.delete(id);
    this.close(chat);
    return current;
  }

  /** A smaller limit takes effect immediately. Keep the earliest chats left running. */
  trim(): void {
    const excess = [...this.background].slice(this.deps.limit());
    for (const [id, chat] of excess) {
      this.background.delete(id);
      this.close(chat);
    }
    if (excess.length) this.deps.backgroundChanged();
  }

  clear(): boolean {
    this.navigation++;
    const current = this.current.saved !== null;
    const chats = [...this.background.values(), ...(current ? [this.current] : [])];
    this.background.clear();
    if (current) this.current = this.create();
    for (const chat of chats) {
      chat.forget();
      this.close(chat);
    }
    return current;
  }

  dispose(): Promise<void> {
    this.disposed = true;
    this.navigation++;
    const chats = [this.current, ...this.background.values()];
    this.background.clear();
    for (const chat of chats) this.close(chat);
    return this.closing;
  }

  private create(): Chat {
    const chat = this.deps.makeChat(() => {
      if (this.isCurrent(chat)) this.deps.foregroundChanged();
      else if (!chat.busy && chat.saved && this.background.get(chat.saved.id) === chat) {
        this.background.delete(chat.saved.id);
        this.close(chat);
        this.deps.backgroundChanged();
      }
    });
    return chat;
  }

  private switchTo(next: Chat): void {
    const previous = this.current;
    this.current = next;
    if (previous.busy && previous.saved && this.background.size < this.deps.limit()) {
      this.background.set(previous.saved.id, previous);
    } else this.close(previous);
    this.deps.backgroundChanged();
  }

  private close(chat: Chat): void {
    this.closing = Promise.all([this.closing, chat.dispose()]).then(() => {});
  }
}
