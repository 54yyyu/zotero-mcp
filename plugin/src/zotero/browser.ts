const WINDOW_URI = "chrome://zotero-agent-browser/content/window.xhtml";
export const PAGE_ACTOR = "ZoteroAgentPage";
/** Gecko nsSandboxFlags.h. The browser has one visible tab and no auxiliary window manager. */
const SANDBOXED_AUXILIARY_NAVIGATION = 0x2;
const LOAD_TIMEOUT = 8000;

export class BrowserError extends Error {}

export type BrowserLabel = "browser.title" | "browser.consent" | "browser.address" | "browser.back" | "browser.take" | "browser.return";

export const BROWSER_TOOLS = [
  { name: "navigate", description: "Open an HTTP(S) URL in the visible Zotero browser.", inputSchema: {
    type: "object", properties: { url: { type: "string" } }, required: ["url"], additionalProperties: false,
  } },
  { name: "read_page", description: "Read bounded visible text and links from the current Zotero browser page.", inputSchema: {
    type: "object", properties: {}, additionalProperties: false,
  }, annotations: { readOnlyHint: true, openWorldHint: true } },
  { name: "follow_link", description: "Navigate to the link with this index in the last Zotero browser page result.", inputSchema: {
    type: "object", properties: { index: { type: "integer", minimum: 1 } }, required: ["index"], additionalProperties: false,
  } },
  { name: "hand_off", description: "Pause agent access so the user can sign in or interact in the same Zotero browser window.", inputSchema: {
    type: "object", properties: {}, additionalProperties: false,
  } },
] as const;

interface Page {
  url: string;
  title: string;
  text: string;
  truncated: boolean;
  links: { href: string; url: string; label: string }[];
}

interface View {
  win: any;
  browser: any;
  context: { id: number; dispose(): void };
  manual: boolean;
  setManual(manual: boolean): void;
  revision: number;
  snapshot?: { documentId: number; links: Page["links"] };
}

function webURI(value: unknown): any {
  if (typeof value === "string" && value.length <= 4096 && !/[\u0000-\u001f\u007f]/.test(value)) {
    try {
      const uri = Services.io.newURI(value);
      if ((uri.schemeIs("https") || uri.schemeIs("http")) && uri.host && !uri.userPass) return uri;
    } catch { /* invalid URL */ }
  }
  throw new BrowserError("Use an absolute HTTP or HTTPS URL without credentials.");
}

async function bounded<T>(win: any, promise: Promise<T>, message: string): Promise<T> {
  let timer: number | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => { timer = win.setTimeout(() => reject(new BrowserError(message)), LOAD_TIMEOUT); }),
    ]);
  } finally {
    if (timer !== undefined) win.clearTimeout(timer);
  }
}

export class AgentBrowser {
  #view: View | undefined;
  #window: any;
  #busy = false;
  #approved: boolean | undefined;
  #closed = false;
  #owner: any;
  #register: () => void;
  #label: (key: BrowserLabel) => string;

  constructor(owner: any, register: () => void, label: (key: BrowserLabel) => string) {
    this.#owner = owner;
    this.#register = register;
    this.#label = label;
  }

  close(): void {
    this.#closed = true;
    if (this.#window && !this.#window.closed) this.#window.close();
  }

  async call(name: string, args: Record<string, unknown>, turnActive: () => boolean): Promise<unknown> {
    const checkControl = () => {
      if (this.#closed || !turnActive()) throw new BrowserError("Zotero Browser tools require an active chat turn.");
      if (this.#view?.manual) throw new BrowserError("The user has control. Wait for them to choose Return control to agent in the browser.");
    };
    checkControl();
    if (this.#busy) throw new BrowserError("A browser operation is already running. Call browser tools one at a time.");
    this.#busy = true;
    try {
      switch (name) {
        case "navigate": {
          const uri = webURI(args["url"]);
          const view = this.#view ?? await this.#open(checkControl);
          checkControl();
          return this.#navigate(view, uri);
        }
        case "read_page":
          return await this.#read(this.#requireView(), checkControl);
        case "follow_link": {
          const view = this.#requireView();
          const index = args["index"];
          if (typeof index !== "number" || !Number.isSafeInteger(index) || index < 1) throw new BrowserError("Choose a link index from read_page.");
          const snapshot = view.snapshot;
          const link = snapshot?.links[index - 1];
          if (!snapshot || !link || snapshot.documentId !== view.browser.browsingContext.currentWindowGlobal?.innerWindowId) {
            throw new BrowserError("That link is unavailable. Read the current page and choose a link from that result.");
          }
          return this.#navigate(view, webURI(link.href));
        }
        case "hand_off": {
          const view = this.#requireView();
          view.setManual(true);
          view.win.focus();
          return { status: "user_control", message: "Ask the user to finish in the Zotero Browser window, choose Return control to agent, and tell you to continue." };
        }
        default:
          throw new BrowserError("Unknown Zotero Browser tool.");
      }
    } finally {
      this.#busy = false;
    }
  }

  #requireView(): View {
    if (!this.#view) throw new BrowserError("Open a page with navigate first.");
    return this.#view;
  }

  async #open(checkControl: () => void): Promise<View> {
    if (this.#approved === undefined) {
      this.#approved = Services.prompt.confirm(this.#owner, this.#label("browser.title"), this.#label("browser.consent"));
      checkControl();
    }
    if (!this.#approved) throw new BrowserError("The user declined browsing in this chat.");
    this.#register();
    const context = Zotero.HTTP.newCookieContext();
    let win: any;
    try {
      win = this.#owner.openDialog(WINDOW_URI, "", "chrome,dialog=no,resizable,centerscreen", null);
    } catch (e) {
      context.dispose();
      throw e;
    }
    this.#window = win;
    let disposed = false;
    const dispose = () => {
      if (disposed) return;
      disposed = true;
      this.#view = undefined;
      this.#window = undefined;
      context.dispose();
      Services.clearData.deleteDataFromOriginAttributesPattern({ userContextId: context.id });
    };
    try {
      await bounded(this.#owner, new Promise<void>((resolve) => {
        const loaded = () => {
          if (win.document.documentURI !== WINDOW_URI) return;
          win.removeEventListener("load", loaded);
          resolve();
        };
        win.addEventListener("load", loaded);
        if (win.document.readyState === "complete") loaded();
      }), "The Zotero Browser window did not open.");
      win.addEventListener("unload", dispose, { once: true });
      checkControl();
      return this.#view = this.#attach(win, context);
    } catch (e) {
      if (!win.closed) win.close();
      dispose();
      throw e;
    }
  }

  #attach(win: any, context: View["context"]): View {
    const doc = win.document;
    const browser = doc.createXULElement("browser");
    browser.setAttribute("type", "content");
    browser.setAttribute("remote", "true");
    browser.setAttribute("flex", "1");
    browser.setAttribute("disableglobalhistory", "true");
    browser.setAttribute("messagemanagergroup", "zoteroAgentBrowser");
    browser.setAttribute("usercontextid", String(context.id));
    doc.getElementById("content").append(browser);
    browser.browsingContext.sandboxFlags |= SANDBOXED_AUXILIARY_NAVIGATION;
    const control = doc.getElementById("control");
    const back = doc.getElementById("back");
    const address = doc.getElementById("address");
    doc.title = this.#label("browser.title");
    address.setAttribute("aria-label", this.#label("browser.address"));
    back.setAttribute("label", this.#label("browser.back"));
    back.addEventListener("command", () => { if (browser.canGoBack) browser.goBack(); });
    const label = this.#label;
    const view: View = {
      win, browser, context, manual: false, revision: 0,
      setManual(manual) {
        view.manual = manual;
        view.revision++;
        view.snapshot = undefined;
        control.setAttribute("label", label(manual ? "browser.return" : "browser.take"));
      },
    };
    control.addEventListener("command", () => view.setManual(!view.manual));
    const location = (uri: any) => {
      view.revision++;
      view.snapshot = undefined;
      address.value = uri?.spec ?? "";
      back.disabled = !browser.canGoBack;
    };
    const listener = {
      onLocationChange(progress: any, _request: any, uri: any) { if (progress.isTopLevel) location(uri); },
      QueryInterface: ChromeUtils.generateQI(["nsIWebProgressListener", "nsISupportsWeakReference"]),
    };
    browser.webProgress.addProgressListener(listener, Ci.nsIWebProgress.NOTIFY_LOCATION);
    win.addEventListener("unload", () => browser.webProgress?.removeProgressListener(listener), { once: true });
    view.setManual(false);
    location(browser.currentURI);
    return view;
  }

  #navigate(view: View, uri: any): unknown {
    view.snapshot = undefined;
    const triggeringPrincipal = Services.scriptSecurityManager.createNullPrincipal({ userContextId: view.context.id });
    view.browser.loadURI(uri, { triggeringPrincipal });
    return { status: "navigating", message: "Call read_page for the loaded page." };
  }

  async #read(view: View, checkControl: () => void): Promise<unknown> {
    const browser = view.browser;
    const deadline = Date.now() + LOAD_TIMEOUT;
    while (Date.now() < deadline) {
      checkControl();
      if (this.#view !== view) throw new BrowserError("The browser window was closed.");
      const frame = browser.browsingContext?.currentWindowGlobal;
      if (!browser.webProgress.isLoadingDocument && frame && /^https?:/.test(frame.documentURI?.spec ?? "")) {
        const revision = view.revision;
        const actor = frame.getActor(PAGE_ACTOR);
        const page = await bounded<Page | { needsLogin: true } | null>(
          this.#owner, actor.sendQuery("readPage"), "The Zotero Browser page did not respond.",
        );
        checkControl();
        if (this.#view !== view) throw new BrowserError("The browser window was closed.");
        if (revision === view.revision && frame === browser.browsingContext.currentWindowGlobal && page) {
          if ("needsLogin" in page) {
            view.snapshot = undefined;
            return { status: "login_required", message: "Use hand_off to let the user sign in." };
          }
          const { links, ...content } = page;
          view.snapshot = { documentId: frame.innerWindowId, links };
          return { ...content, links: links.map(({ url, label }, i) => ({ index: i + 1, url, label })) };
        }
      }
      await new Promise((resolve) => this.#owner.setTimeout(resolve, 80));
    }
    throw new BrowserError("No readable web page loaded. Check the Zotero Browser window or try read_page again.");
  }
}
