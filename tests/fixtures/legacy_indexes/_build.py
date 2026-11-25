"""Run inside the throwaway environment of one old zotero-mcp release; see make_legacy_indexes.py.

    python _build.py <work dir>

Builds the fake Zotero under <work dir>/zotero, points HOME at <work dir>/home, and runs the
release's own ``zotero-mcp update-db --fulltext --force-rebuild`` with chunking on. The only
thing replaced is the embedding model (fake_embedding.FakeDefaultEmbedding: no download, no
network).
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
work = Path(sys.argv[1]).resolve()

home = work / "home"
os.environ["HOME"] = str(home)
os.environ["USERPROFILE"] = str(home)
os.environ["XDG_CONFIG_HOME"] = str(home / ".config")
for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ZOTERO_API_KEY"):
    os.environ.pop(var, None)

import fake_zotero  # noqa: E402

db = fake_zotero.build(work / "zotero")
os.environ["ZOTERO_DB_PATH"] = str(db)
os.environ["ZOTERO_LOCAL"] = "true"

config_dir = home / ".config" / "zotero-mcp"
config_dir.mkdir(parents=True)
(config_dir / "config.json").write_text(
    json.dumps(
        {"semantic_search": {"embedding_model": "default", "chunking": fake_zotero.CHUNKING}},
        indent=2,
        sort_keys=True,
    )
)

import socket  # noqa: E402


def _no_network(*args, **kwargs):
    raise OSError("the legacy-index build is offline")


# Nothing may leave the machine, and the user's running Zotero is not part of this either.
socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network
socket.getaddrinfo = _no_network


class _OfflineZotero:
    """What the old releases get for the Zotero Web/local API: an empty library."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: {} if name in ("item_versions", "deleted") else None


from fake_embedding import FakeDefaultEmbedding  # noqa: E402

import zotero_mcp.chroma_client as chroma_client  # noqa: E402

chroma_client.ChromaClient._create_embedding_function = lambda self: FakeDefaultEmbedding()

import zotero_mcp.client as zotero_client  # noqa: E402
import zotero_mcp.semantic_search as semantic_search  # noqa: E402

zotero_client.get_zotero_client = lambda *a, **k: _OfflineZotero()
semantic_search.get_zotero_client = lambda *a, **k: _OfflineZotero()

from zotero_mcp.cli import main  # noqa: E402

sys.argv = ["zotero-mcp", "update-db", "--fulltext", "--force-rebuild"]
main()
