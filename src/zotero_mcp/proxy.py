"""EZProxy URL rewriting — Python port of Zotero's proxy scheme system.

Ported from:
  https://github.com/zotero/zotero/blob/main/chrome/content/zotero/xpcom/proxy.js

Mirrors Zotero.Proxy / Zotero.Proxies using dataclasses and config.json for
persistence instead of SQLite.

Scheme parameters (same as Zotero):
    %h  hostname  — in HTTPS HttpsHyphens mode dots become dashes
    %p  full path + query + fragment
    %d  directory portion of path
    %f  filename portion of path
    %a  anything

Typical multi-host EZProxy scheme (HttpsHyphens):
    https://%h.ezproxy.uni.edu/%p

    https://www.jstor.org/stable/123
    -> https://www-jstor-org.ezproxy.uni.edu/stable/123
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"
_DOI_HOSTS = {"doi.org", "dx.doi.org"}
_BROWSER_ORDER = ["chrome", "chromium", "firefox", "brave", "edge", "opera"]

# Regex fragments that replace scheme parameters when building the match regexp.
# %h gets its own pattern only for multi-host proxies (set in compile_regexp).
_SCHEME_PARAMS: dict[str, str] = {
    "%p": r"(.*?)",
    "%d": r"(.*?)",
    "%f": r"(.*?)",
    "%a": r"(.*?)",
}
# Matches the hyphen-separated or dot-separated hostname captured by %h
_HOST_PARAM_RE = r"([a-zA-Z0-9]+[.\-][a-zA-Z0-9.\-]+)"


# ---------------------------------------------------------------------------
# Proxy — mirrors Zotero.Proxy
# ---------------------------------------------------------------------------


@dataclass
class Proxy:
    """A single EZProxy proxy definition.

    Attributes:
        scheme:         URL template with %h/%p/%d/%f/%a parameters.
        hosts:          Proper hostnames associated with this proxy
                        (grows automatically when auto_associate is True).
        multi_host:     Auto-set to True when scheme contains %h.
        auto_associate: Whether unknown hosts should be added to this proxy.
        proxy_id:       Stable numeric ID used when saving to config.json.
    """

    scheme: str
    hosts: list[str] = field(default_factory=list)
    multi_host: bool = False
    auto_associate: bool = False
    proxy_id: Optional[int] = None

    # Set by compile_regexp(); not persisted.
    regexp: Optional[re.Pattern] = field(default=None, init=False, repr=False)
    parameters: list[str] = field(default_factory=list, init=False, repr=False)
    indices: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if "%h" in self.scheme:
            self.multi_host = True
        if self.scheme:
            self.compile_regexp()

    # ------------------------------------------------------------------
    # compile_regexp — port of Zotero.Proxy.prototype.compileRegexp
    # ------------------------------------------------------------------

    def compile_regexp(self) -> None:
        """Compile self.regexp to match proxied URLs; record parameter positions."""
        params_to_check = dict(_SCHEME_PARAMS)
        if self.multi_host:
            params_to_check["%h"] = _HOST_PARAM_RE

        scheme = self.scheme
        indices: dict[str, int] = {}
        parameters: list[str] = []

        for param in params_to_check:
            idx = scheme.find(param)
            # Remove preceding '%' escape (e.g. '%%p' → literal '%p', not a param)
            while 0 < idx and scheme[idx - 1] == "%":
                scheme = scheme[: idx - 1] + scheme[idx:]
                idx = scheme.find(param, idx)
            if idx != -1:
                indices[param] = idx
                parameters.append(param)

        parameters.sort(key=lambda p: indices[p])
        self.scheme = scheme  # store de-escaped scheme (matches Zotero behaviour)
        self.parameters = parameters
        self.indices = indices

        if "://" in scheme:
            re_str = "^" + re.escape(scheme) + "$"
        else:
            re_str = "^https?" + re.escape("://" + scheme) + "$"

        # Substitute params in reverse index order so earlier positions stay valid
        for param in reversed(parameters):
            re_str = re_str.replace(param, params_to_check[param], 1)

        self.regexp = re.compile(re_str, re.IGNORECASE)

    # ------------------------------------------------------------------
    # to_proper — port of Zotero.Proxy.prototype.toProper
    # ------------------------------------------------------------------

    def to_proper(self, url: str) -> str:
        """Return the canonical (unproxied) URL for a proxied *url*.

        Returns *url* unchanged if it does not match this proxy's pattern.
        """
        if not self.regexp:
            return url
        m = self.regexp.match(url)
        if not m:
            return url
        # groups[0] = full URL string, groups[1..] = capture groups
        groups: tuple = (url,) + m.groups()

        protocol = "https://" if "https" in groups[0].lower() else "http://"

        if self.multi_host:
            host_raw = groups[self.parameters.index("%h") + 1]
        else:
            host_raw = self.hosts[0] if self.hosts else ""

        proper_base = protocol + host_raw + "/"

        # EZProxy HttpsHyphens: dashes in the captured hostname represent dots
        if "https" in protocol:
            _, _, rest = proper_base.partition("://")
            host_part, _, _ = rest.partition("/")
            host_part = host_part.replace("-", ".")
            proper_base = protocol + host_part + "/"

        if "%p" in self.indices:
            return proper_base + groups[self.parameters.index("%p") + 1]

        d = groups[self.parameters.index("%d") + 1]
        f = groups[self.parameters.index("%f") + 1]
        return proper_base + (d.lstrip("/") + "/" if d else "") + f

    # ------------------------------------------------------------------
    # to_proxy — port of Zotero.Proxy.prototype.toProxy
    # ------------------------------------------------------------------

    def to_proxy(self, url: str) -> str:
        """Return the proxied form of *url* through this proxy's scheme.

        Returns *url* unchanged if it already matches the proxy pattern.
        """
        if self.regexp and self.regexp.match(url):
            return url

        parsed = urlparse(url)
        proxy_url = self.scheme

        # Process in reverse so substituting a later param doesn't shift earlier indices
        for param in reversed(self.parameters):
            if param == "%h":
                host = parsed.hostname or ""
                # HttpsHyphens mode: dots → dashes in HTTPS hostnames
                value = host.replace(".", "-") if parsed.scheme == "https" else host
            elif param == "%p":
                path = parsed.path.lstrip("/")
                query = f"?{parsed.query}" if parsed.query else ""
                fragment = f"#{parsed.fragment}" if parsed.fragment else ""
                value = path + query + fragment
            elif param == "%d":
                p = parsed.path
                value = p[: p.rfind("/")]
            elif param == "%f":
                p = parsed.path
                value = p[p.rfind("/") + 1 :]
            else:
                value = ""

            idx = self.indices[param]
            proxy_url = proxy_url[:idx] + value + proxy_url[idx + 2 :]

        return proxy_url

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "proxy_id": self.proxy_id,
            "scheme": self.scheme,
            "hosts": self.hosts,
            "multi_host": self.multi_host,
            "auto_associate": self.auto_associate,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Proxy":
        return cls(
            scheme=data["scheme"],
            hosts=data.get("hosts", []),
            multi_host=data.get("multi_host", False),
            auto_associate=data.get("auto_associate", False),
            proxy_id=data.get("proxy_id"),
        )


# ---------------------------------------------------------------------------
# ProxyRegistry — port of Zotero.Proxies
# ---------------------------------------------------------------------------


class ProxyRegistry:
    """Manages a collection of Proxy objects and provides URL rewriting.

    Also holds auth settings (browser, cookie_string) for make_session().
    Persists everything to the existing ~/.config/zotero-mcp/config.json
    under the 'ezproxy' key.

    Config.json shape::

        {
          "ezproxy": {
            "browser": "auto",
            "proxies": [
              {
                "proxy_id": 1,
                "scheme": "https://%h.ezproxy.uni.edu/%p",
                "hosts": [],
                "multi_host": true,
                "auto_associate": true
              }
            ]
          }
        }
    """

    def __init__(self, browser: str = "auto", cookie_string: Optional[str] = None) -> None:
        self.proxies: list[Proxy] = []
        self.hosts: dict[str, Proxy] = {}  # proper hostname → Proxy
        self.browser = browser
        self.cookie_string = cookie_string
        self._next_id = 1

    # -- collection management (mirrors Zotero.Proxies) ------------------

    def add(self, proxy: Proxy) -> None:
        """Add *proxy* and update the hostname index."""
        if proxy not in self.proxies:
            if proxy.proxy_id is None:
                proxy.proxy_id = self._next_id
            self._next_id = max(self._next_id, (proxy.proxy_id or 0) + 1)
            self.proxies.append(proxy)
        if proxy.proxy_id:
            for host in proxy.hosts:
                self.hosts[host] = proxy

    def remove(self, proxy: Proxy) -> bool:
        try:
            self.proxies.remove(proxy)
        except ValueError:
            return False
        self.hosts = {h: p for h, p in self.hosts.items() if p is not proxy}
        return True

    def refresh_host_map(self, proxy: Proxy) -> None:
        """Re-sync hostname index after proxy.hosts changes."""
        if not proxy.proxy_id:
            return
        self.hosts = {h: p for h, p in self.hosts.items() if p is not proxy or h in proxy.hosts}
        self.add(proxy)

    # -- URL conversion (mirrors Zotero.Proxies) -------------------------

    def proxy_to_proper(self, url: str, only_if_proxied: bool = False) -> str:
        """Convert a proxied URL to its canonical form."""
        for proxy in self.proxies:
            if proxy.regexp and proxy.regexp.match(url):
                return proxy.to_proper(url)
        return "" if only_if_proxied else url

    def proper_to_proxy(self, url: str, only_if_proxied: bool = False) -> str:
        """Convert a canonical URL to its proxied form.

        Checks the hostname index first; then falls through to any multi-host proxy.
        """
        host = urlparse(url).hostname or ""
        if host in self.hosts and self.hosts[host].proxy_id:
            return self.hosts[host].to_proxy(url)
        for proxy in self.proxies:
            if proxy.multi_host:
                return proxy.to_proxy(url)
        return "" if only_if_proxied else url

    # -- persistence (config.json) --------------------------------------

    def load_from_config(self, config_path: Optional[str] = None) -> None:
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        ezp = cfg.get("ezproxy", {})
        self.browser = ezp.get("browser", "auto")
        self.cookie_string = ezp.get("cookie_string")
        self.proxies = []
        self.hosts = {}
        self._next_id = 1
        for data in ezp.get("proxies", []):
            self.add(Proxy.from_json(data))

    def save_to_config(self, config_path: Optional[str] = None) -> None:
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg: dict = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        entry: dict = {"browser": self.browser, "proxies": [p.to_json() for p in self.proxies]}
        if self.cookie_string:
            entry["cookie_string"] = self.cookie_string
        cfg["ezproxy"] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_config_file(cls, config_path: Optional[str] = None) -> "ProxyRegistry":
        registry = cls()
        registry.load_from_config(config_path)
        return registry


# ---------------------------------------------------------------------------
# DOI resolution
# ---------------------------------------------------------------------------


def resolve_doi(url: str, timeout: int = 10) -> str:
    """Follow redirects on a doi.org URL to get the actual publisher URL."""
    if urlparse(url).netloc not in _DOI_HOSTS:
        return url
    resp = requests.head(
        url,
        allow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; zotero-mcp/1.0)"},
    )
    return resp.url


# ---------------------------------------------------------------------------
# Cookie loading / session
# ---------------------------------------------------------------------------


def _proxy_cookie_domain(registry: ProxyRegistry) -> str:
    """Derive the EZProxy domain to scope cookie loading from the first proxy scheme."""
    for proxy in registry.proxies:
        m = re.search(r"%h\.([^/%]+)", proxy.scheme)
        if m:
            return m.group(1)
    return ""


def _load_browser_cookies(domain: str, browser: str = "auto"):
    try:
        import browser_cookie3  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "browser-cookie3 is required for automatic cookie extraction. Install with:  pip install browser-cookie3"
        ) from exc

    candidates = _BROWSER_ORDER if browser == "auto" else [browser]
    last_err: Exception | None = None
    for b in candidates:
        loader = getattr(browser_cookie3, b, None)
        if loader is None:
            continue
        try:
            jar = loader(domain_name=domain)
            if jar is not None:
                return jar
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(
        f"Could not load cookies for '{domain}' from any browser "
        f"({', '.join(candidates)}). "
        "Log into EZProxy in your browser first. "
        f"Last error: {last_err}"
    )


def make_session(registry: ProxyRegistry) -> requests.Session:
    """Return a requests.Session pre-loaded with EZProxy authentication cookies."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    if registry.cookie_string:
        domain = _proxy_cookie_domain(registry)
        for part in registry.cookie_string.split(";"):
            part = part.strip()
            if "=" in part:
                name, _, value = part.partition("=")
                session.cookies.set(name.strip(), value.strip(), domain=domain or None)
    else:
        domain = _proxy_cookie_domain(registry)
        if not domain:
            raise ValueError(
                "Cannot determine proxy domain for cookie extraction. "
                "Add a proxy with a scheme like 'https://%h.ezproxy.uni.edu/%p'."
            )
        jar = _load_browser_cookies(domain, registry.browser)
        session.cookies.update(jar)

    return session


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_paper(
    url: str,
    registry: ProxyRegistry,
    *,
    resolve_dois: bool = True,
    follow_redirects: bool = True,
    timeout: int = 30,
) -> dict:
    """Fetch a paper via EZProxy using browser authentication cookies.

    Steps: resolve DOI → rewrite URL through proxy → fetch with auth session.

    Returns a JSON-serialisable dict with:
        url, resolved_url, proxied_url, final_url,
        status_code, content_type, content_length,
        title (HTML only), body (decoded text for HTML/plain, else None).
    """
    resolved = resolve_doi(url, timeout=timeout) if resolve_dois else url
    proxied = registry.proper_to_proxy(resolved)

    session = make_session(registry)
    resp = session.get(proxied, allow_redirects=follow_redirects, timeout=timeout, stream=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    content = resp.content

    return {
        "url": url,
        "resolved_url": resolved,
        "proxied_url": proxied,
        "final_url": resp.url,
        "status_code": resp.status_code,
        "content_type": content_type,
        "content_length": len(content),
        "body": content.decode("utf-8", errors="replace"),
    }


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


def configure_proxy(
    scheme: str,
    *,
    hosts: Optional[list[str]] = None,
    auto_associate: bool = True,
    browser: str = "auto",
    cookie_string: Optional[str] = None,
    config_path: Optional[str] = None,
) -> ProxyRegistry:
    """Create a ProxyRegistry with one proxy and persist it to config.json.

    Example::

        from zotero_mcp.proxy import configure_proxy
        registry = configure_proxy("https://%h.ezproxy.myuniversity.edu/%p")
    """
    registry = ProxyRegistry(browser=browser, cookie_string=cookie_string)
    registry.add(Proxy(scheme=scheme, hosts=hosts or [], auto_associate=auto_associate))
    registry.save_to_config(config_path)
    return registry
