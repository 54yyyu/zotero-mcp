from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import browser_cookie3
import requests

from zotero_mcp import utils as _utils

BROWSERS: dict[str, Callable] = {
    "firefox": browser_cookie3.firefox,
    "chrome": browser_cookie3.chrome,
    "chromium": browser_cookie3.chromium,
    "brave": browser_cookie3.brave,
    "edge": browser_cookie3.edge,
    "opera": browser_cookie3.opera,
    "vivaldi": browser_cookie3.vivaldi,
    "librewolf": browser_cookie3.librewolf,
}


@dataclass
class Proxy:
    domain: str
    browser: str = "firefox"

    def _bk(self) -> Callable:
        bk = BROWSERS.get(self.browser)
        if bk is None:
            raise ValueError(f"Unsupported browser: {self.browser}. Options: {list(BROWSERS)}")
        return bk

    def rewrite(self, url: str) -> str:
        """Rewrite a canonical URL through the proxy (dots -> hyphens in hostname)."""
        parsed = urlparse(url)
        host = (parsed.hostname or "").replace(".", "-")
        path = parsed.path.lstrip("/")
        if parsed.query:
            path += "?" + parsed.query
        return f"https://{host}.{self.domain}/{path}"

    def fetch(self, url: str) -> dict:
        """Fetch a paper URL through the proxy. Returns dict with status, url, content."""
        if self.domain not in url:
            url = self.rewrite(url)
        cookies = self._bk()(domain_name=self.domain)
        session = requests.Session()
        session.cookies.update(cookies)
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return {
            "status": resp.status_code,
            "url": str(resp.url),
            "content": _utils.clean_html(resp.text, collapse_whitespace=True),
        }


def fetch_via_proxy(
    url: str,
    proxy_domain: str,
    browser: str = "firefox",
) -> dict:
    """Fetch a paper URL through your institution's proxy using browser cookies."""
    return Proxy(domain=proxy_domain, browser=browser).fetch(url)
