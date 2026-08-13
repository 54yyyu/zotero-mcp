"""Parent assignment for Zotero items."""

from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers


@mcp.tool(
    name="zotero_set_item_parent",
    description=(
        "Set or clear the parent of a Zotero item. Pass a parent item key to "
        "assign or change the parent, or null to make the item top-level. "
        "Zotero validates whether the requested parent-child relationship is allowed."
    ),
)
@with_zotero_api_lock
def set_item_parent(
    item_key: str,
    parent_key: str | None,
    *,
    ctx: Context,
) -> str:
    """Set ``parentItem`` through Zotero's versioned item-update API."""
    try:
        _read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        item = write_zot.item(item_key)
        payload = {
            "key": item["key"],
            "version": item["version"],
            "parentItem": parent_key if parent_key is not None else False,
        }
        response = write_zot.update_item(payload)
        if not _helpers._handle_write_response(response, ctx):
            return f"Failed to set parent for item '{item_key}'."

        if parent_key is None:
            return f"Successfully cleared the parent of item `{item_key}`."
        return f"Successfully set the parent of item `{item_key}` to `{parent_key}`."
    except Exception as e:
        ctx.error(f"Error setting parent for item {item_key}: {e}")
        return f"Error setting parent for item '{item_key}': {e}"
