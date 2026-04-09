"""Zotero MCP server — thin entry-point that registers all tools.

The actual tool implementations live in :mod:`zotero_mcp.tools.*`.
This module re-exports public names so that existing callers
(``from zotero_mcp.server import mcp``, tests that call
``server.some_function()``, etc.) keep working.

Tool modules use module-level attribute access (e.g. ``_client.get_zotero_client()``)
so that tests can patch the canonical location directly.
"""

# -- FastMCP app instance ---------------------------------------------------
from zotero_mcp._app import mcp  # noqa: F401 — re-export

# -- Register every tool module by importing the package --------------------
import zotero_mcp.tools  # noqa: F401 — side-effect: registers all @mcp.tool

# -- Re-export client helpers (used by tests as server.X) -------------------
from zotero_mcp.client import (  # noqa: F401
    get_zotero_client,
    get_web_zotero_client,
    get_active_library,
    set_active_library,
    clear_active_library,
    convert_to_markdown,
    format_item_metadata,
    generate_bibtex,
    get_attachment_details,
)
from zotero_mcp.utils import (  # noqa: F401
    format_creators,
    format_item_result,
    clean_html,
    is_local_mode,
)

# -- Re-export private helpers (used by tests) ------------------------------
from zotero_mcp.tools._helpers import (  # noqa: F401
    CROSSREF_TYPE_MAP,
    _get_write_client,
    _handle_write_response,
    _normalize_limit,
    _normalize_str_list_input,
    _resolve_collection_names,
    _normalize_doi,
    _normalize_arxiv_id,
    _download_and_attach_pdf,
    _attach_pdf_linked_url,
    _try_unpaywall,
    _try_arxiv_from_crossref,
    _try_semantic_scholar,
    _try_pmc,
    _try_attach_oa_pdf,
    _extra_has_citekey,
    _format_citekey_result,
    _format_bbt_result,
)

# -- Re-export tool functions (used by tests as server.func_name) -----------
from zotero_mcp.tools.search import (  # noqa: F401
    search_items,
    search_by_tag,
    search_by_citation_key,
    advanced_search,
    semantic_search,
    update_search_database,
    get_search_database_status,
)
from zotero_mcp.tools.retrieval import (  # noqa: F401
    get_item_metadata,
    get_item_fulltext,
    get_collections,
    get_collection_items,
    get_item_children,
    get_items_children,
    get_tags,
    list_libraries,
    switch_library,
    validate_library_switch,
    list_feeds,
    get_feed_items,
    get_recent,
)
from zotero_mcp.tools.annotations import (  # noqa: F401
    get_annotations,
    _get_annotations,
    get_notes,
    _batch_resolve_parent_titles,
    _format_search_results,
    search_notes,
    create_note,
    update_note,
    delete_note,
    create_annotation,
)
from zotero_mcp.tools.write import (  # noqa: F401
    batch_update_tags,
    create_collection,
    search_collections,
    manage_collections,
    add_by_doi,
    add_by_url,
    update_item,
    find_duplicates,
    merge_duplicates,
    get_pdf_outline,
    add_from_file,
)
from zotero_mcp.tools.connectors import (  # noqa: F401
    chatgpt_connector_search,
    connector_fetch,
)
