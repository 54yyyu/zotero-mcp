#!/usr/bin/env python3
import sys
from urllib.parse import urlparse, urlunparse


class Proxy:
    def __init__(self, proxy_host: str):
        self.proxy_host = proxy_host

    def rewrite(self, url: str) -> str:
        p = urlparse(url)
        host = (p.hostname or "").replace(".", "-")
        return urlunparse(p._replace(netloc=f"{host}.{self.proxy_host}"))


if __name__ == "__main__":
    proxy = Proxy(sys.argv[1])
    for url in sys.argv[2:]:
        print(proxy.rewrite(url))
