"""Regression test for issue #362.

``zotero_get_search_database_status`` reported "0 documents / Not initialized"
against a fully populated database, while the CLI ``db-status`` reported the
correct count for the *same* database.

Root cause
----------
``read_collection_status`` opens the collection with
``get_collection(name, embedding_function=_NoEmbeddingFunction())`` precisely so
it does NOT reconstruct (and download) the persisted embedding model — counting
rows never needs to embed anything. ChromaDB >=1.x, however, validates the
provided embedding function against the persisted config in
``validate_embedding_function_conflict_on_get``::

    if (embedding_function.name() != "default"
            and persisted_ef_config.get("name") is not None
            and persisted_ef_config.get("name") != embedding_function.name()):
        raise ValueError("... Embedding function conflict ...")

``_NoEmbeddingFunction`` did not implement ``name()``, so it reported
``NotImplemented`` — which is ``!= "default"`` and ``!=`` the persisted name —
tripping the conflict. The broad ``except Exception`` inside
``read_collection_status`` then swallowed the ``ValueError`` and returned
``{count: 0, initialized: False}``.

Fix
---
``_NoEmbeddingFunction.name()`` returns ``"default"``, which short-circuits the
conflict check for *any* persisted backend. The no-op function is still never
invoked, because ``count()`` does not embed.
"""

import shutil
import tempfile

import pytest

# chromadb is an optional extra (``[semantic]``); skip where it isn't installed.
chromadb = pytest.importorskip("chromadb")

from chromadb import Documents, EmbeddingFunction, Embeddings  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from chromadb.utils.embedding_functions import (  # noqa: E402
    register_embedding_function,
)

from zotero_mcp import chroma_client  # noqa: E402


@register_embedding_function
class _PersistedFakeEF(EmbeddingFunction):
    """A registered EF whose persisted name is *not* ``"default"``.

    This recreates the embedding-function conflict that a real populated
    collection produces, without downloading the ~80MB ONNX default model.
    """

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [[0.1, 0.2, 0.3] for _ in input]

    @staticmethod
    def name() -> str:
        return "persisted_fake_362"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config) -> "_PersistedFakeEF":
        return _PersistedFakeEF()


def _populate(persist_dir, collection_name, n=5):
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    col = client.get_or_create_collection(
        name=collection_name, embedding_function=_PersistedFakeEF()
    )
    col.add(
        ids=[str(i) for i in range(n)],
        documents=[f"doc {i}" for i in range(n)],
    )
    return n


def test_no_embedding_function_name_is_default():
    """The fix hinges on this exact value: ``"default"`` is the only name that
    ChromaDB's conflict check short-circuits on, regardless of the persisted
    backend (default / openai / gemini / ...)."""
    assert chroma_client._NoEmbeddingFunction.name() == "default"


def test_read_collection_status_reports_populated_count():
    # tempfile (not pytest's tmp_path, which is flagged unreliable on CI in
    # this repo's conftest); persist_directory is passed explicitly.
    persist_dir = tempfile.mkdtemp(prefix="zotero_mcp_status_362_")
    try:
        n = _populate(persist_dir, "zotero_library", n=5)

        status = chroma_client.read_collection_status(
            config_path=None, persist_directory=persist_dir
        )

        assert status.get("error") is None, status
        assert status["initialized"] is True
        assert status["count"] == n
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)
