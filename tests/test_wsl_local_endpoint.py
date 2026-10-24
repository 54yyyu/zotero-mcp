"""Regression tests for #148: reach the Zotero local API from WSL2.

pyzotero hardcodes ``http://127.0.0.1:23119/api`` for local mode. Under WSL2
NAT that address is the Linux container, not the Windows host running Zotero,
so every API-backed tool fails with "Cannot connect to Zotero local API".
Zotero's local server also rejects a request whose ``Host`` header is not
``localhost:23119`` with a 400, so the client must connect to the Windows host
IP while presenting ``localhost:23119``.

The Windows host is the default gateway in ``/proc/net/route``, and WSL is
detected from ``WSL_DISTRO_NAME`` / ``/proc/version``. These tests feed a fake
route table through ``builtins.open`` and set the environment marker, so they
fail on the unpatched code at the assertion that the endpoint was rewritten
(where it still reads 127.0.0.1) instead of at import.
"""

import builtins
import io

from pyzotero import zotero

from zotero_mcp import client as zclient

# A default route (destination 00000000) whose gateway is the WSL2 Windows
# host. The value is little-endian hex, so 0170A8C0 is 192.168.112.1.
_WSL_ROUTE_TABLE = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
    "eth0\t00000000\t0170A8C0\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
    "eth0\t0000A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n"
)


def _fake_proc_files(monkeypatch, route_contents, *, wsl_version=True):
    """Serve fake /proc files, deferring every other open() to the real one."""
    real_open = builtins.open
    version = (
        "Linux version 5.15.0-microsoft-standard-WSL2 ..."
        if wsl_version
        else "Linux version 5.15.0-generic ..."
    )

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/net/route":
            return io.StringIO(route_contents)
        if str(path) == "/proc/version":
            return io.StringIO(version)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def _capture_client(monkeypatch, *, wsl=True):
    """Build get_zotero_client() in local mode and hand back the live client."""
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "0")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    if wsl:
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    else:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)

    return zclient.get_zotero_client()


def test_local_endpoint_targets_wsl_host(monkeypatch):
    """Under WSL2 the local endpoint must name the Windows host, not 127.0.0.1."""
    _fake_proc_files(monkeypatch, _WSL_ROUTE_TABLE)
    zot = _capture_client(monkeypatch)
    try:
        assert zot.endpoint == "http://192.168.112.1:23119/api"
    finally:
        zot.client.close()


def test_local_requests_carry_the_host_header(monkeypatch):
    """Zotero rejects a Host that isn't localhost:23119, so it must be injected."""
    _fake_proc_files(monkeypatch, _WSL_ROUTE_TABLE)
    zot = _capture_client(monkeypatch)
    try:
        assert zot.client.headers.get("host") == "localhost:23119"
    finally:
        zot.client.close()


def test_wsl_detected_from_proc_version_without_env(monkeypatch):
    """WSL must still be detected when WSL_DISTRO_NAME is absent."""
    _fake_proc_files(monkeypatch, _WSL_ROUTE_TABLE, wsl_version=True)
    zot = _capture_client(monkeypatch, wsl=False)
    try:
        assert zot.endpoint == "http://192.168.112.1:23119/api"
    finally:
        zot.client.close()


def test_no_rewrite_on_a_native_linux_host(monkeypatch):
    """A default route alone is not WSL2; native Linux must be left alone.

    The gateway here (10.0.0.1) is a plausible NAT router on any Linux box.
    Rewriting the endpoint for it would break every non-WSL local install, so
    the fix must only fire when WSL is actually detected.
    """
    native_route = (
        "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
        "eth0\t00000000\t0100000A\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
    )
    _fake_proc_files(monkeypatch, native_route, wsl_version=False)
    zot = _capture_client(monkeypatch, wsl=False)
    try:
        assert zot.endpoint == "http://127.0.0.1:23119/api"
        assert zot.client.headers.get("host") is None
    finally:
        zot.client.close()
