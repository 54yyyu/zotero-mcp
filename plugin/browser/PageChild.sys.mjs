// Runs in the browser's content process. The caller can request only a bounded view of the top page.
const TEXT_LIMIT = 16_000;
const LINK_LIMIT = 60;
const URL_LABEL_LIMIT = 512;

function webURL(value) {
  try {
    const url = new URL(value);
    if ((url.protocol === "http:" || url.protocol === "https:") && !url.username && !url.password) return url;
  } catch { /* invalid link */ }
  return null;
}

export class ZoteroAgentPageChild extends JSWindowActorChild {
  receiveMessage({ name }) {
    if (name !== "readPage") return;
    const doc = this.document;
    const url = webURL(doc.URL);
    if (!url || !doc.nodePrincipal.isContentPrincipal || !doc.body || doc.readyState === "loading") return null;
    // Form values and raw markup stay in Gecko. If a password form is visible, ask for user control.
    for (const input of doc.querySelectorAll('input[type="password"]')) {
      if (input.getClientRects().length && doc.defaultView.getComputedStyle(input).visibility === "visible") {
        return { needsLogin: true };
      }
    }
    const links = [];
    for (const link of doc.querySelectorAll("a[href]")) {
      const target = webURL(link.href);
      if (!target || target.href.length > 4096 || !link.getClientRects().length) continue;
      const label = link.innerText?.trim() || link.getAttribute("aria-label") || "";
      links.push({ href: target.href, url: (target.origin + target.pathname).slice(0, URL_LABEL_LIMIT), label: label.slice(0, 200) });
      if (links.length === LINK_LIMIT) break;
    }
    const text = doc.body.innerText ?? "";
    return {
      url: (url.origin + url.pathname).slice(0, URL_LABEL_LIMIT),
      title: doc.title.slice(0, 300),
      text: text.slice(0, TEXT_LIMIT),
      truncated: text.length > TEXT_LIMIT,
      links,
    };
  }
}
