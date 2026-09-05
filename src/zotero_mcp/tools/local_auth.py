"""Local API write authorization tools.

Writing through Zotero's local API needs a key the user grants in a dialog in
the Zotero app. These two tools let an agent obtain that key and check whether
it already has one, instead of guessing why a write was refused.
"""

import httpx
from pyzotero.errors import (
    LocalAPIDeniedError,
    ParamNotPassedError,
    TooManyRequestsError,
    UnsupportedParamsError,
)

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context

# Under FastMCP's ~60s client timeout: past that the transport gives up and
# the user gets an opaque failure instead of "you didn't answer the dialog".
_MIN_TIMEOUT = 5
_MAX_TIMEOUT = 55


@mcp.tool(
    name="zotero_authorize_local_writes",
    description=(
        "Request permission to write to the local Zotero library (Zotero 10+). "
        "BLOCKS until the user answers a dialog that appears in the Zotero "
        "app, so tell them to switch to Zotero before calling this. "
        "\"Always Allow\" grants a reusable key that is saved for future "
        "sessions; \"Allow\" grants a key good for exactly ONE write. "
        "Call this when a write tool reports that no writable backend is "
        "configured and zotero_write_capabilities says the server supports "
        "local writes but no key is held. Not needed in web API mode. "
        "app_name: the name shown to the user in the dialog. "
        "timeout: seconds to wait for the answer (5-55, default 45). "
        "Zotero rate-limits this to about 5 prompts per minute — do not retry "
        "in a loop. Example: zotero_authorize_local_writes()."
    )
)
def authorize_local_writes(
    app_name: str = _client.DEFAULT_APP_NAME,
    timeout: int = 45,
    *,
    ctx: Context
) -> str:
    """Obtain a local API key, prompting the user in the Zotero app.

    Deliberately NOT wrapped in @with_zotero_api_lock, unlike every other
    write-adjacent tool here. This call blocks on a human decision; holding
    the shared API lock for that long would push every concurrent tool past
    its bounded acquire and turn one dialog into a wave of "Zotero is busy"
    errors. The narrower hazard — two dialogs at once — is covered by the
    dedicated lock inside client.authorize_local_api.
    """
    if not _utils.is_local_mode():
        return (
            "Local writes do not apply here: the server is not in local mode "
            "(ZOTERO_LOCAL is not set). Writes already go through the web API."
        )

    timeout = max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, int(timeout)))

    try:
        ctx.info(f"Requesting local API authorization as {app_name!r}")
        result = _client.authorize_local_api(app_name=app_name, timeout=timeout)
    except _client.ZoteroAuthInProgressError as e:
        return f"Error: {e}"
    except LocalAPIDeniedError:
        return (
            "Authorization denied — the request was declined in Zotero. "
            "Call this tool again and choose \"Allow\" or \"Always Allow\"."
        )
    except TooManyRequestsError:
        return (
            "Zotero is rate-limiting authorization prompts (about 5 per "
            "minute). Wait a minute before trying again."
        )
    except UnsupportedParamsError:
        return (
            "This Zotero build does not expose the local write API, which "
            "needs Zotero 10 or newer. Set ZOTERO_API_KEY and "
            "ZOTERO_LIBRARY_ID to write through the web API instead."
        )
    except ParamNotPassedError:
        return "Error: app_name cannot be empty — it identifies this app in the dialog."
    except httpx.TimeoutException:
        return (
            f"No answer within {timeout}s. The dialog is probably still open "
            "in Zotero — answer it there, then call this tool again. It will "
            "not open a second dialog while the first is pending."
        )
    except Exception as e:
        ctx.error(f"Local authorization failed: {e}")
        return f"Error requesting local API authorization: {e}"

    if result.get("remember"):
        return (
            "Local write access granted and saved.\n\n"
            "Writes now go straight to the running Zotero instead of the "
            "cloud. The key was stored in ~/.config/zotero-mcp/config.json "
            "and will be reused in future sessions. Revoke it with "
            "`zotero-mcp authorize-local --revoke`, or from Zotero's "
            "Settings > Advanced > Clear Write Authorizations."
        )
    return (
        "Local write access granted for a SINGLE write.\n\n"
        "\"Allow\" was chosen rather than \"Always Allow\", so this key is "
        "consumed by the next successful write and the one after that will "
        "fail. It has deliberately not been saved to disk, because a consumed "
        "key there would take priority over working web credentials on every "
        "later run. Ask the user to call this tool again and choose "
        "\"Always Allow\" for a key that persists."
    )


@mcp.tool(
    name="zotero_write_capabilities",
    description=(
        "Report whether and how this server can write to Zotero: local API, "
        "web API, hybrid (local reads + cloud writes), or nothing. Read-only "
        "and instant — it never opens a dialog. Call it when a write tool "
        "reports that no writable backend is configured, to find out which "
        "fix applies before asking the user for anything: if the local "
        "server supports writes but no key is held, call "
        "zotero_authorize_local_writes; otherwise web API credentials are "
        "needed. Takes no arguments. Example: zotero_write_capabilities()."
    )
)
def write_capabilities(*, ctx: Context) -> str:
    """Describe the currently available write path."""
    caps = _client.get_write_capabilities()

    modes = {
        "local": "Local API — writes go directly to the running Zotero.",
        "hybrid": "Hybrid — local reads, writes through the Zotero web API.",
        "web": "Web API — reads and writes both go through zotero.org.",
        "none": "None — writes are not possible with the current configuration.",
    }
    lines = [
        "# Zotero write capabilities",
        "",
        f"**Mode:** {caps['mode']} — {modes[caps['mode']]}",
        "",
        f"- Local mode (ZOTERO_LOCAL): {'yes' if caps['local_mode'] else 'no'}",
    ]

    if caps["local_mode"]:
        supported = caps["server_supports_local_writes"]
        lines.append(
            f"- Running Zotero supports local writes: "
            f"{'yes' if supported else 'no (needs Zotero 10 or newer)'}"
        )
        if caps["has_local_key"]:
            remember = caps["local_key_remember"]
            validity = {
                True: "reusable",
                False: "SINGLE USE — consumed by the next write",
                None: "validity unknown",
            }[remember]
            lines.append(
                f"- Local API key: held, from {caps['local_key_source']} ({validity})"
            )
        else:
            lines.append("- Local API key: none")
        if not caps["local_write_enabled"]:
            lines.append("- Local writes are switched off by ZOTERO_LOCAL_WRITE")

    lines.append(
        f"- Web API credentials: {'set' if caps['has_web_credentials'] else 'not set'}"
    )

    if caps["mode"] == "none":
        lines += ["", "**To enable writes:**"]
        if caps["server_supports_local_writes"]:
            lines += [
                "1. Call zotero_authorize_local_writes and have the user "
                "choose \"Always Allow\" in Zotero.",
                "2. Or set ZOTERO_API_KEY and ZOTERO_LIBRARY_ID for web API writes.",
            ]
        else:
            lines.append(
                "- Set ZOTERO_API_KEY and ZOTERO_LIBRARY_ID for web API writes "
                "(this Zotero has no local write API)."
            )
    elif caps["mode"] == "hybrid" and caps["server_supports_local_writes"]:
        lines += [
            "",
            "This Zotero supports local writes. Calling "
            "zotero_authorize_local_writes would keep writes off the cloud "
            "round-trip.",
        ]

    return "\n".join(lines)
