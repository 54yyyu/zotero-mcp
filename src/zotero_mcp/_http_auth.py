"""Shared-secret authentication and bind safeguards for HTTP transports."""

import hashlib
import hmac
import ipaddress
import os
import re
import sys

from fastmcp.server.auth import AccessToken, TokenVerifier

_TOKEN_ENV = "ZOTERO_MCP_AUTH_TOKEN"


def validate_auth_token(value: object) -> str:
    """Reject malformed secrets without including their contents in errors."""
    # RFC 6750 bearer credentials. Do not trim or coerce configured secrets:
    # a JSON null/number or an accidental newline must not become a credential.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", value):
        raise ValueError(
            f"{_TOKEN_ENV} must be a non-empty bearer token string with no whitespace. "
            "Remove the setting to disable shared-secret authentication."
        )
    return value


class _SharedSecretVerifier(TokenVerifier):
    """Verify a single configured secret without retaining its plaintext."""

    def __init__(self, token: str):
        super().__init__(required_scopes=[])
        self._token_digest = hashlib.sha256(token.encode("utf-8")).digest()

    async def verify_token(self, token: str) -> AccessToken | None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        if not hmac.compare_digest(digest, self._token_digest):
            return None
        return AccessToken(token=token, client_id="zotero-mcp", scopes=[])


def _is_loopback_host(host: str) -> bool:
    """Recognize explicit loopback addresses without trusting DNS aliases."""
    if host.lower().removesuffix(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def configure_http_auth(mcp, host: str, allow_unauthenticated: bool = False) -> None:
    """Configure authentication before FastMCP constructs either HTTP app."""
    token = os.environ.get(_TOKEN_ENV)
    if token is not None:
        mcp.auth = _SharedSecretVerifier(validate_auth_token(token))

    # Preserve an auth provider configured by FastMCP itself when no shared
    # secret is supplied. The escape hatch never disables configured auth.
    if mcp.auth is not None:
        return
    if not _is_loopback_host(host) and not allow_unauthenticated:
        raise ValueError(
            "Refusing to serve unauthenticated HTTP on a non-loopback host. "
            f"Set {_TOKEN_ENV}, use --host 127.0.0.1 or --host ::1, "
            "or explicitly opt in with --allow-unauthenticated."
        )
    print(
        "Warning: HTTP authentication is disabled. Anyone who can reach this "
        "listener, including through a tunnel, can use its enabled tools. "
        f"Set {_TOKEN_ENV} to require a bearer token.",
        file=sys.stderr,
    )
