"""SQLite list_tags must list what the API's /tags endpoint lists (#552).

/tags counts tags on trashed items. The SQLite query excluded them, so a tag
carried only by a trashed item appeared on one backend and not the other.
"""

import _search_corpus as corpus
from zotero_mcp.local_db import LocalZoteroReader


def test_tags_on_trashed_items_are_listed(tmp_path):
    items = [
        corpus.Item("LIVE0001", title="Live Paper", tags=["kept-tag"]),
        corpus.Item("DELETED1", title="Trashed Paper", tags=["trash-only-tag"]),
    ]
    db = tmp_path / "zotero.sqlite"
    corpus.build_sqlite(db, items)
    with LocalZoteroReader(db_path=str(db)) as reader:
        tags = set(reader.list_tags(group_id=0))
    assert {"kept-tag", "trash-only-tag"} <= tags
