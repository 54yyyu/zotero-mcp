"""Unit tests for proxy.py — Proxy scheme system and ProxyRegistry persistence."""

import json
import pytest

from zotero_mcp.proxy import Proxy, ProxyRegistry, configure_proxy


SCHEME = "https://%h.ezproxy.uni.edu/%p"


# ---------------------------------------------------------------------------
# Proxy.to_proxy — proper URL → proxied URL
# ---------------------------------------------------------------------------

class TestToProxy:
    def setup_method(self):
        self.proxy = Proxy(scheme=SCHEME)

    def test_simple(self):
        assert self.proxy.to_proxy("https://www.jstor.org/stable/123") == \
            "https://www-jstor-org.ezproxy.uni.edu/stable/123"

    def test_dots_become_dashes_https(self):
        result = self.proxy.to_proxy("https://link.springer.com/article/10.1234/s")
        assert "link-springer-com" in result

    def test_path_and_query_preserved(self):
        url = "https://www.nature.com/articles/abc?q=1&p=2#sec1"
        result = self.proxy.to_proxy(url)
        assert result == "https://www-nature-com.ezproxy.uni.edu/articles/abc?q=1&p=2#sec1"

    def test_no_subdomain_host(self):
        result = self.proxy.to_proxy("https://nature.com/articles/abc")
        assert result == "https://nature-com.ezproxy.uni.edu/articles/abc"

    def test_already_proxied_unchanged(self):
        proxied = "https://www-jstor-org.ezproxy.uni.edu/stable/123"
        assert self.proxy.to_proxy(proxied) == proxied

    def test_empty_path(self):
        result = self.proxy.to_proxy("https://www.example.com/")
        assert result == "https://www-example-com.ezproxy.uni.edu/"


# ---------------------------------------------------------------------------
# Proxy.to_proper — proxied URL → canonical URL
# ---------------------------------------------------------------------------

class TestToProper:
    def setup_method(self):
        self.proxy = Proxy(scheme=SCHEME)

    def test_simple(self):
        assert self.proxy.to_proper("https://www-jstor-org.ezproxy.uni.edu/stable/123") == \
            "https://www.jstor.org/stable/123"

    def test_dashes_become_dots(self):
        result = self.proxy.to_proper("https://link-springer-com.ezproxy.uni.edu/article/10.1234/s")
        assert result == "https://link.springer.com/article/10.1234/s"

    def test_query_and_fragment_preserved(self):
        proxied = "https://www-nature-com.ezproxy.uni.edu/articles/abc?q=1&p=2#sec1"
        assert self.proxy.to_proper(proxied) == "https://www.nature.com/articles/abc?q=1&p=2#sec1"

    def test_non_matching_url_unchanged(self):
        url = "https://www.google.com/search"
        assert self.proxy.to_proper(url) == url

    def test_roundtrip(self):
        original = "https://www.jstor.org/stable/123?q=test"
        assert self.proxy.to_proper(self.proxy.to_proxy(original)) == original


# ---------------------------------------------------------------------------
# Proxy — scheme without protocol prefix
# ---------------------------------------------------------------------------

class TestSchemeWithoutProtocol:
    def test_scheme_without_protocol_compiles(self):
        proxy = Proxy(scheme="%h.ezproxy.uni.edu/%p")
        assert proxy.regexp is not None

    def test_matches_http_and_https(self):
        proxy = Proxy(scheme="%h.ezproxy.uni.edu/%p")
        assert proxy.regexp.match("http://www-jstor-org.ezproxy.uni.edu/stable/123")
        assert proxy.regexp.match("https://www-jstor-org.ezproxy.uni.edu/stable/123")


# ---------------------------------------------------------------------------
# Proxy — %d and %f parameters
# ---------------------------------------------------------------------------

class TestDirFileParams:
    def test_dir_file_scheme(self):
        proxy = Proxy(scheme="https://%h.ezproxy.uni.edu/%d/%f", multi_host=True)
        proxied = proxy.to_proxy("https://www.example.com/articles/paper.html")
        assert "www-example-com.ezproxy.uni.edu" in proxied


# ---------------------------------------------------------------------------
# ProxyRegistry — collection management
# ---------------------------------------------------------------------------

class TestProxyRegistry:
    def test_add_assigns_id(self):
        registry = ProxyRegistry()
        proxy = Proxy(scheme=SCHEME)
        registry.add(proxy)
        assert proxy.proxy_id is not None

    def test_ids_increment(self):
        registry = ProxyRegistry()
        p1 = Proxy(scheme=SCHEME)
        p2 = Proxy(scheme="https://%h.other.edu/%p")
        registry.add(p1)
        registry.add(p2)
        assert p1.proxy_id != p2.proxy_id

    def test_remove(self):
        registry = ProxyRegistry()
        proxy = Proxy(scheme=SCHEME)
        registry.add(proxy)
        assert registry.remove(proxy)
        assert proxy not in registry.proxies

    def test_remove_nonexistent_returns_false(self):
        registry = ProxyRegistry()
        assert not registry.remove(Proxy(scheme=SCHEME))

    def test_host_map_updated_on_add(self):
        registry = ProxyRegistry()
        proxy = Proxy(scheme=SCHEME, hosts=["www.jstor.org"])
        registry.add(proxy)
        assert "www.jstor.org" in registry.hosts

    def test_host_map_cleared_on_remove(self):
        registry = ProxyRegistry()
        proxy = Proxy(scheme=SCHEME, hosts=["www.jstor.org"])
        registry.add(proxy)
        registry.remove(proxy)
        assert "www.jstor.org" not in registry.hosts


# ---------------------------------------------------------------------------
# ProxyRegistry — URL conversion
# ---------------------------------------------------------------------------

class TestRegistryUrlConversion:
    def setup_method(self):
        self.registry = ProxyRegistry()
        self.registry.add(Proxy(scheme=SCHEME))

    def test_proper_to_proxy(self):
        result = self.registry.proper_to_proxy("https://www.jstor.org/stable/123")
        assert result == "https://www-jstor-org.ezproxy.uni.edu/stable/123"

    def test_proxy_to_proper(self):
        result = self.registry.proxy_to_proper("https://www-jstor-org.ezproxy.uni.edu/stable/123")
        assert result == "https://www.jstor.org/stable/123"

    def test_proper_to_proxy_via_host_map(self):
        registry = ProxyRegistry()
        proxy = Proxy(scheme=SCHEME, hosts=["www.jstor.org"])
        registry.add(proxy)
        result = registry.proper_to_proxy("https://www.jstor.org/stable/123")
        assert "ezproxy.uni.edu" in result

    def test_unmatched_proxy_to_proper_passthrough(self):
        url = "https://www.google.com/search"
        assert self.registry.proxy_to_proper(url) == url

    def test_unmatched_proper_to_proxy_only_if_proxied(self):
        # No multi-host proxies → empty string when only_if_proxied
        registry = ProxyRegistry()
        proxy = Proxy(scheme=SCHEME, hosts=["www.jstor.org"])
        registry.add(proxy)
        result = registry.proper_to_proxy("https://www.unknown.com/", only_if_proxied=True)
        assert result == ""


# ---------------------------------------------------------------------------
# ProxyRegistry — persistence
# ---------------------------------------------------------------------------

class TestRegistryPersistence:
    def test_roundtrip(self, tmp_path):
        cfg_path = str(tmp_path / "config.json")
        registry = ProxyRegistry(browser="firefox")
        registry.add(Proxy(scheme=SCHEME, hosts=["www.jstor.org"], auto_associate=True))
        registry.save_to_config(cfg_path)

        loaded = ProxyRegistry.from_config_file(cfg_path)
        assert loaded.browser == "firefox"
        assert len(loaded.proxies) == 1
        assert loaded.proxies[0].scheme == SCHEME
        assert "www.jstor.org" in loaded.proxies[0].hosts

    def test_merges_existing_config_keys(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"semantic_search": {"zotero_db_path": "/keep/me"}}))

        registry = ProxyRegistry()
        registry.add(Proxy(scheme=SCHEME))
        registry.save_to_config(str(cfg_path))

        with open(cfg_path) as f:
            data = json.load(f)
        assert data["semantic_search"]["zotero_db_path"] == "/keep/me"
        assert "ezproxy" in data

    def test_cookie_string_persisted(self, tmp_path):
        cfg_path = str(tmp_path / "config.json")
        registry = ProxyRegistry(cookie_string="ezproxy=TOKEN123")
        registry.add(Proxy(scheme=SCHEME))
        registry.save_to_config(cfg_path)

        loaded = ProxyRegistry.from_config_file(cfg_path)
        assert loaded.cookie_string == "ezproxy=TOKEN123"

    def test_load_empty_config(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{}")
        registry = ProxyRegistry.from_config_file(str(cfg_path))
        assert registry.proxies == []

    def test_load_missing_file_is_noop(self, tmp_path):
        registry = ProxyRegistry()
        registry.load_from_config(str(tmp_path / "nonexistent.json"))
        assert registry.proxies == []

    def test_compiled_regexp_works_after_load(self, tmp_path):
        cfg_path = str(tmp_path / "config.json")
        ProxyRegistry.from_config_file  # pre-check
        r1 = ProxyRegistry()
        r1.add(Proxy(scheme=SCHEME))
        r1.save_to_config(cfg_path)

        r2 = ProxyRegistry.from_config_file(cfg_path)
        result = r2.proper_to_proxy("https://www.jstor.org/stable/123")
        assert result == "https://www-jstor-org.ezproxy.uni.edu/stable/123"


# ---------------------------------------------------------------------------
# configure_proxy helper
# ---------------------------------------------------------------------------

class TestEzproxyConfigureHelper:
    def test_creates_registry_and_saves(self, tmp_path):
        cfg_path = str(tmp_path / "config.json")
        registry = configure_proxy(SCHEME, browser="chrome", config_path=cfg_path)
        assert registry.browser == "chrome"
        assert len(registry.proxies) == 1

        loaded = ProxyRegistry.from_config_file(cfg_path)
        assert loaded.proxies[0].scheme == SCHEME

    def test_with_hosts(self, tmp_path):
        cfg_path = str(tmp_path / "config.json")
        registry = configure_proxy(SCHEME, hosts=["www.jstor.org"], config_path=cfg_path)
        assert "www.jstor.org" in registry.proxies[0].hosts
