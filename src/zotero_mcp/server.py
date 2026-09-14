"""Zotero MCP server — thin entry-point that registers all tools.

The actual tool implementations live in :mod:`zotero_mcp.tools.*`.
This module re-exports public names so that existing callers
(``from zotero_mcp.server import mcp``, tests that call
``server.some_function()``, etc.) keep working.

Tool modules use module-level attribute access (e.g. ``_client.get_zotero_client()``)
so that tests can patch the canonical location directly.
"""

# -- FastMCP app instance ---------------------------------------------------
# -- Register every tool module by importing the package --------------------
import zotero_mcp.tools  # noqa: F401 — side-effect: registers all @mcp.tool
from zotero_mcp._app import mcp  # noqa: F401 — re-export

# -- Re-export client helpers (used by tests as server.X) -------------------
from zotero_mcp.client import (  # noqa: F401
    authorize_local_api,
    clear_active_library,
    convert_to_markdown,
    format_item_metadata,
    generate_bibtex,
    get_active_library,
    get_attachment_details,
    get_local_write_client,
    get_web_zotero_client,
    get_write_capabilities,
    get_zotero_client,
    set_active_library,
)

# -- Re-export private helpers (used by tests) ------------------------------
from zotero_mcp.tools._helpers import (  # noqa: F401
    CROSSREF_TYPE_MAP,
    _attach_pdf_linked_url,
    _download_and_attach_pdf,
    _extra_has_citekey,
    _format_bbt_result,
    _format_citekey_result,
    _get_write_client,
    resolve_write_client,
    write_unavailable_message,
    item_template_for,
    attach_files,
    trash_item,
    format_zotero_error,
    describe_write_failure,
    _handle_write_response,
    _normalize_arxiv_id,
    _normalize_doi,
    _normalize_limit,
    _normalize_str_list_input,
    _resolve_collection_names,
    _try_arxiv_from_crossref,
    _try_attach_oa_pdf,
    _try_pmc,
    _try_semantic_scholar,
    _try_unpaywall,
)
from zotero_mcp.tools.annotations import (  # noqa: F401
    _batch_resolve_parent_titles,
    _format_search_results,
    _get_annotations,
    create_annotation,
    create_area_annotation,
    create_note,
    delete_annotation,
    delete_note,
    get_annotations,
    get_notes,
    get_notes_tool,
    get_page_layout,
    manage_note,
    search_notes,
    update_annotation,
    update_note,
)
from zotero_mcp.tools.connectors import (  # noqa: F401
    chatgpt_connector_search,
    connector_fetch,
)
from zotero_mcp.tools.local_auth import (  # noqa: F401
    authorize_local_writes,
    write_capabilities,
)
from zotero_mcp.tools.read_pdf import (  # noqa: F401
    read_pdf_pages,
)
from zotero_mcp.tools.retrieval import (  # noqa: F401
    get_collection_items,
    get_collections,
    get_feed_items,
    get_item_children,
    get_item_fulltext,
    get_item_metadata,
    get_item_related,
    get_recent,
    get_tags,
    list_feeds,
    list_libraries,
    switch_library,
    validate_library_switch,
)

# -- Re-export tool functions (used by tests as server.func_name) -----------
from zotero_mcp.tools.search import (  # noqa: F401
    advanced_search,
    get_search_database_status,
    search_by_citation_key,
    search_by_tag,
    search_items,
    semantic_search,
    update_search_database,
)
from zotero_mcp.tools.write import (  # noqa: F401
    add_by_bibtex,
    add_by_csl_json,
    add_by_doi,
    add_by_isbn,
    add_by_url,
    add_from_file,
    add_item,
    add_item_relation,
    attach_file,
    batch_update,
    batch_update_extra,
    batch_update_tags,
    create_collection,
    delete_collection,
    delete_item,
    find_duplicates,
    get_pdf_outline,
    manage_collections,
    merge_duplicates,
    remove_item_relation,
    search_collections,
    update_item,
)

from zotero_mcp.toolsets import apply_toolsets
from zotero_mcp.utils import (  # noqa: F401
    clean_html,
    format_creators,
    format_item_result,
    is_local_mode,
)

# Apply the optional-toolset profile now that every tool module above has
# registered its tools. This uses the stdio default; `zotero-mcp serve`
# re-applies once the real transport is known, which is what lets the ChatGPT
# connector tools appear only when the server is actually reachable by ChatGPT.
# Callers that import `mcp` directly still get the configured profile rather
# than the full surface.
apply_toolsets(mcp)
