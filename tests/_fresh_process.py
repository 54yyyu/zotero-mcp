"""End a test step the way a process exit would.

ChromaDB keeps one client (and its background writer) per persist directory.
After an in-process update the test still holds the ``search`` object, so
``release_chroma_clients()`` alone does not stop the writer, and on Windows the
test's own next read of the directory can fail with "Nothing found on disk".
"""

import gc

from zotero_mcp.chroma_client import release_chroma_clients


def as_fresh_process(*searches):
    """End a step the way a process exit would: drop its ChromaDB clients, then the cache."""
    for s in searches:
        s.chroma_client = None
    release_chroma_clients()
    gc.collect()
