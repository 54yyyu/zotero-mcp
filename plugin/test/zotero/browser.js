// The installed XPI, real Gecko and a mock ACP bridge. All pages and credentials are fixtures.
async function main({ plugin, win, Services, waitFor, log }) {
  const expect = (ok, message) => { if (!ok) throw new Error(message); };
  const { HttpServer } = win.ChromeUtils.importESModule("chrome://remote/content/server/httpd.sys.mjs");
  const site = new HttpServer();
  const html = (body) => (_req, res) => {
    res.setHeader("Content-Type", "text/html; charset=utf-8", false);
    res.write("<!doctype html>" + body);
  };
  site.registerPathHandler("/", html('<title>Publisher fixture</title><p>Guest article &#26085;&#26412;&#35486;</p><a href="/login?state=remove-me#secret">Sign in</a>'));
  site.registerPathHandler("/login", html('<a style="display:block;padding:20px" href="/signed-in">Sign in to fixture</a><input type="password" value="fixture-password">'));
  site.registerPathHandler("/signed-in", (_req, res) => {
    res.setStatusLine("1.1", 302, "Found");
    res.setHeader("Set-Cookie", "account=fixture; Path=/; HttpOnly", false);
    res.setHeader("Location", "/article?code=remove-me", false);
  });
  site.registerPathHandler("/article", (req, res) => {
    const signedIn = req.hasHeader("Cookie") && req.getHeader("Cookie").includes("account=fixture");
    html(signedIn ? '<title>Article</title><p>Signed-in article</p><a href="/next">Next page</a>' : "<p>Guest view</p>")(req, res);
  });
  site.registerPathHandler("/next", html("<p>Second page</p>"));
  site.start(-1);
  const base = "http://127.0.0.1:" + site.identity.primaryPort;
  const { host } = plugin.panel(win);
  const sessions = new Set();
  async function start(backend = "claude-code") {
    const opts = await host.prepareSession(undefined, backend);
    const session = await host.runtime.start({ backend, ...opts, env: { ...opts.env, ...(backend === "pi" ? { MOCK_VARIANT: "pi" } : {}) } });
    sessions.add(session);
    return session;
  }
  async function run(session, calls) {
    let answer = "";
    const off = session.on((event) => { if (event.t === "text") answer += event.delta; });
    try {
      await session.prompt({ text: "SCENARIO:browser " + JSON.stringify(calls) });
      return JSON.parse(answer);
    } finally {
      off();
    }
  }
  const window = () => Services.wm.getMostRecentWindow("zotero:agent-browser");
  let consent = false;
  let prompts = 0;
  const observer = {
    observe(subject, topic) {
      if (topic !== "domwindowopened") return;
      subject.addEventListener("load", () => {
        const doc = subject.document;
        const dialog = doc.querySelector("dialog");
        if (doc.documentURI !== "chrome://global/content/commonDialog.xhtml" || !dialog) return;
        prompts++;
        dialog.getButton(consent ? "accept" : "cancel").click();
      }, { once: true });
    },
  };
  Services.ww.registerNotification(observer);
  try {
    const off = await start();
    expect(await run(off, [["connect"]]) === null, "Browser was offered with the setting off");
    await off.close();
    await host.setSettings({ browser: true, uiLanguage: "en" });
    const unsupported = await start("pi");
    expect(await run(unsupported, [["connect"]]) === null, "Browser was offered without ACP HTTP support");
    await unsupported.close();

    const denied = await start();
    const deniedResult = await run(denied, [["navigate", { url: base }], ["navigate", { url: base }]]);
    expect(deniedResult.every((r) => r.error?.includes("declined")) && prompts === 1 && !window(), "Declining browsing did not stop it");
    await denied.close();
    consent = true;

    const session = await start();
    log("connecting browser");
    const first = await run(session, [["connect"], ["navigate", { url: "file:///tmp/private" }], ["navigate", { url: base }], ["read_page"]]);
    const [connection, blocked, opened, page] = first;
    expect(connection.init.result?.protocolVersion === "2025-11-25" && connection.initialized === 202 && connection.tools?.length === 4, "MCP handshake failed");
    expect(connection.untrusted === 403 && connection.anonymous === 403, "Loopback endpoint accepted an untrusted request");
    expect(blocked.error?.includes("HTTP") && opened.status === "navigating", "URL validation or navigation failed: " + JSON.stringify([blocked, opened]));
    expect(page.text?.includes("Guest article 日本語") && page.links[0].url === base + "/login", "Rendered page or redacted link missing: " + JSON.stringify(page));
    const browserWindow = window();
    const browser = browserWindow.document.querySelector("browser");
    const contextId = browser.browsingContext.originAttributes.userContextId;
    expect(contextId > 0 && browser.isRemoteBrowser, "Expected a remote browser with an isolated cookie context");

    log("following link and handing off");
    const [followed, login, handoff, read, navigate] = await run(session, [
      ["follow_link", { index: 1 }], ["read_page"], ["hand_off"], ["read_page"], ["navigate", { url: base }],
    ]);
    expect(followed.status === "navigating" && login.status === "login_required" && !JSON.stringify(login).includes("fixture-password"), "Sign-in page was readable or missing");
    expect(handoff.status === "user_control" && [read, navigate].every((r) => r.error?.includes("user has control")), "Agent retained control during handoff");
    const control = browserWindow.document.getElementById("control");
    const address = browserWindow.document.getElementById("address");
    expect(control.getAttribute("label") === "Return control to agent", "Handoff is not visible in the browser");
    const rect = browser.getBoundingClientRect();
    const mouse = browserWindow.windowUtils;
    mouse.sendMouseEvent("mousedown", rect.x + 30, rect.y + 30, 0, 1, 0);
    mouse.sendMouseEvent("mouseup", rect.x + 30, rect.y + 30, 0, 1, 0);
    await waitFor(() => address.value.startsWith(base + "/article"), "user sign-in redirect");
    control.click();
    const [stale, article] = await run(session, [["follow_link", { index: 1 }], ["read_page"]]);
    expect(stale.error?.includes("unavailable"), "A link from the pre-handoff page could still be followed");
    expect(article.url === base + "/article" && article.text?.includes("Signed-in article") && window() === browserWindow, "Agent did not resume in the authenticated tab");
    expect(Services.cookies.getCookiesFromHost("127.0.0.1", { userContextId: contextId }).some((cookie) => cookie.name === "account"), "Fixture never received an authenticated cookie");
    const [, next] = await run(session, [["follow_link", { index: 1 }], ["read_page"]]);
    expect(next.text?.includes("Second page"), "Agent could not follow a link after handoff");

    await session.close();
    await waitFor(() => browserWindow.closed, "browser closes with the chat");
    expect(!Services.cookies.getCookiesFromHost("127.0.0.1", { userContextId: contextId }).some((cookie) => cookie.name === "account"), "Cookie survived browser close");
    const resumed = await host.runtime.start({ backend: "claude-code", ...await host.prepareSession(), resumeSessionId: session.sessionId });
    sessions.add(resumed);
    const [, guest] = await run(resumed, [["navigate", { url: base + "/article" }], ["read_page"]]);
    expect(guest.text?.includes("Guest view"), "Resumed chat did not get a clean browser");
    const resumedWindow = window();
    await host.setSettings({ browser: false });
    await waitFor(() => resumedWindow.closed, "browser closes when disabled");
    return { rendered: true, httpAuth: true, handoff: true, cookieIsolation: true, closed: true };
  } finally {
    Services.ww.unregisterNotification(observer);
    for (const session of sessions) await session.close();
    await site.stop();
  }
}
