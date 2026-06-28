"""Tests for the institutional-proxy paper fetch (src/zotero_mcp/proxy.py).

The fetch path reads cookies from the user's browser and forwards them through
a configured proxy. These tests mock both the browser cookie reader and
``requests`` so nothing touches a real browser, keychain, or network.
"""

import json

import pytest

from zotero_mcp import proxy as proxy_mod
from zotero_mcp.proxy import Proxy, fetch_via_proxy


class _FakeCookieJar:
    def __init__(self):
        self.updated = None

    def update(self, cookies):
        self.updated = cookies


class _FakeResponse:
    def __init__(self, url, status=200, text="<html><body>Hello <b>world</b></body></html>"):
        self.url = url
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Records the args fetch() passes to .get() so tests can assert on them."""

    last = None

    def __init__(self):
        self.cookies = _FakeCookieJar()
        self.get_url = None
        self.get_kwargs = None
        _FakeSession.last = self

    def get(self, url, **kwargs):
        self.get_url = url
        self.get_kwargs = kwargs
        return _FakeResponse(url)


@pytest.fixture
def fake_cookies(monkeypatch):
    """Install a fake browser cookie reader; return the dict of recorded calls."""
    calls = {}

    def reader(domain_name=None):
        calls["domain_name"] = domain_name
        return {"sessionid": "secret"}

    monkeypatch.setitem(proxy_mod.BROWSERS, "firefox", reader)
    return calls


@pytest.fixture
def fake_session(monkeypatch):
    monkeypatch.setattr(proxy_mod.requests, "Session", _FakeSession)
    return _FakeSession


# --- rewrite ---------------------------------------------------------------


def test_rewrite_dots_to_hyphens_with_query():
    p = Proxy(domain="proxy.edu")
    out = p.rewrite("https://www.nature.com/articles/abc?x=1&y=2")
    assert out == "https://www-nature-com.proxy.edu/articles/abc?x=1&y=2"


def test_rewrite_root_path():
    assert Proxy(domain="proxy.edu").rewrite("https://example.com") == "https://example-com.proxy.edu/"


# --- fetch: cookie handling + request shape --------------------------------


def test_fetch_reads_cookies_for_proxy_domain(fake_cookies, fake_session):
    Proxy(domain="proxy.edu", browser="firefox").fetch("https://www.nature.com/x")
    # Cookies are read for the proxy domain, not the paper's origin host.
    assert fake_cookies["domain_name"] == "proxy.edu"
    assert _FakeSession.last.cookies.updated == {"sessionid": "secret"}


def test_fetch_sends_timeout(fake_cookies, fake_session):
    Proxy(domain="proxy.edu").fetch("https://www.nature.com/x")
    assert _FakeSession.last.get_kwargs.get("timeout") == 30


def test_fetch_rewrites_url_through_proxy(fake_cookies, fake_session):
    Proxy(domain="proxy.edu").fetch("https://www.nature.com/x")
    assert _FakeSession.last.get_url == "https://www-nature-com.proxy.edu/x"


def test_fetch_skips_rewrite_when_already_proxied(fake_cookies, fake_session):
    url = "https://www-nature-com.proxy.edu/x"
    Proxy(domain="proxy.edu").fetch(url)
    assert _FakeSession.last.get_url == url


def test_fetch_strips_html(fake_cookies, fake_session):
    result = Proxy(domain="proxy.edu").fetch("https://www.nature.com/x")
    assert result["status"] == 200
    assert result["content"] == "Hello world"
    assert "<b>" not in result["content"]


def test_fetch_raises_on_http_error(fake_cookies, monkeypatch):
    class ErrSession(_FakeSession):
        def get(self, url, **kwargs):
            return _FakeResponse(url, status=500)

    monkeypatch.setattr(proxy_mod.requests, "Session", ErrSession)
    with pytest.raises(RuntimeError):
        Proxy(domain="proxy.edu").fetch("https://www.nature.com/x")


def test_fetch_unsupported_browser_raises(fake_session):
    with pytest.raises(ValueError, match="Unsupported browser"):
        Proxy(domain="proxy.edu", browser="netscape").fetch("https://www.nature.com/x")


# --- fetch_via_proxy: proxy domain is required -----------------------------


def test_fetch_via_proxy_requires_domain():
    with pytest.raises(TypeError):
        fetch_via_proxy("https://www.nature.com/x")  # no proxy_domain


def test_fetch_via_proxy_passes_through(fake_cookies, fake_session):
    fetch_via_proxy("https://www.nature.com/x", proxy_domain="proxy.edu")
    assert _FakeSession.last.get_url == "https://www-nature-com.proxy.edu/x"


# --- tool config loading ---------------------------------------------------


def test_load_proxy_config_missing(monkeypatch, tmp_path):
    from zotero_mcp.tools import proxy as tool

    monkeypatch.setattr(tool, "_DEFAULT_CONFIG_PATH", tmp_path / "nope.json")
    assert tool._load_proxy_config() is None


def test_load_proxy_config_reads_proxy_section(monkeypatch, tmp_path):
    from zotero_mcp.tools import proxy as tool

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"proxy": {"domain": "proxy.edu", "browser": "chrome"}}))
    monkeypatch.setattr(tool, "_DEFAULT_CONFIG_PATH", cfg)
    assert tool._load_proxy_config() == {"domain": "proxy.edu", "browser": "chrome"}
