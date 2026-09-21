"""The user-visible failure, reproduced against a real persistent ChromaDB.

A collection is built with a remote-provider embedding function, then reopened
by a ``ChromaClient`` configured with ``embedding_model="default"`` — which is
what happens on a missing or partial config, a bare ``create_chroma_client()``,
or an env override. ChromaDB skips its embedding-function conflict check for
its own ``DefaultEmbeddingFunction``, so the collection survives and ChromaDB
instead rebuilds the *persisted* remote embedding function from the stored
config and drives it itself (``CollectionCommon._embed``):

    config_ef.embed_query(input=input)     # queries
    config_ef(input=input)                 # documents

Our ``embed_query(self, text: str)`` did not accept ``input``, so every search
died with::

    TypeError: RemoteEmbeddingFunction.embed_query() got an unexpected keyword
    argument 'input' in query.

while indexing kept working, because the document side goes through
``__call__``. ``ChromaClient.search`` masked the defect for a correctly
configured client by calling our method positionally itself.

Offline: the stub provider's ``_embed_batch`` returns fixed vectors and makes
no request, and the default embedding function downloads its model only on
``__call__``, which nothing here reaches (ChromaDB prefers the persisted
embedding function over a supplied default one).
"""

import importlib.util
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from chromadb.utils.embedding_functions import register_embedding_function  # noqa: E402

from zotero_mcp.chroma_client import ChromaClient  # noqa: E402
from zotero_mcp.embeddings.base import RemoteEmbeddingFunction  # noqa: E402

COLLECTION = "embed_query_contract"
PROVIDER_NAME = "stub_remote_provider"


# Registration is process-global for the whole test session; PROVIDER_NAME is unique to this file.
@register_embedding_function
class _StubRemoteEF(RemoteEmbeddingFunction):
    """A registered remote provider that never leaves the process.

    Registered (module scope, like the real providers) because the whole point
    of this test is that ChromaDB resolves the persisted provider *name* back
    to a class and rebuilds it. Calls are recorded on the class, not the
    instance: the instance ChromaDB rebuilds from the persisted config is not
    the one the test constructed.
    """

    calls: list[tuple[list[str], bool]] = []

    def __init__(self, model_name: str = "stub-embed-1"):
        self._init_common(
            model_name=model_name,
            base_url=None,
            request_batch_size=None,
            rate_limit_rps=None,
            max_parallel_requests=None,
            max_retries=0,
        )

    @staticmethod
    def name() -> str:
        return PROVIDER_NAME

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "_StubRemoteEF":
        return _StubRemoteEF(model_name=config.get("model_name", "stub-embed-1"))

    def _embed_batch(self, texts, is_query=False):
        type(self).calls.append((list(texts), is_query))
        # Deterministic, offline, and ordered so that "alpha" is the nearest
        # neighbour of the query "alpha" — proving the vectors are really used.
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]


@pytest.fixture(autouse=True)
def _reset_calls():
    _StubRemoteEF.calls.clear()
    yield
    _StubRemoteEF.calls.clear()


@pytest.fixture()
def seeded_dir(tmp_path, monkeypatch):
    """A persisted collection whose embedding function is the stub provider."""
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)

    path = str(tmp_path)
    # allow_reset mirrors ChromaClient's own Settings: ChromaDB caches one System
    # per path and rejects a reopen (the `reopened` fixture below) whose Settings differ.
    client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False, allow_reset=True))
    collection = client.get_or_create_collection(name=COLLECTION, embedding_function=_StubRemoteEF())
    collection.add(ids=["A", "B"], documents=["alpha", "bravooo"])
    assert collection.count() == 2
    del collection
    del client
    _StubRemoteEF.calls.clear()
    return path


@pytest.fixture()
def reopened(seeded_dir):
    """The same directory, reopened as a ``default``-model ChromaClient."""
    return ChromaClient(
        collection_name=COLLECTION,
        persist_directory=seeded_dir,
        embedding_model="default",
    )


def test_default_configured_client_keeps_the_collection_and_its_provider(reopened):
    """No silent wipe: the documents and the persisted provider both survive."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    assert isinstance(reopened.embedding_function, DefaultEmbeddingFunction), (
        "the client under test must be the one configured with 'default'"
    )
    assert reopened.collection.count() == 2

    persisted_ef = reopened.collection.configuration.get("embedding_function")
    assert persisted_ef is not None
    assert persisted_ef.name() == PROVIDER_NAME
    assert persisted_ef.get_config()["model_name"] == "stub-embed-1"


def test_search_through_the_rebuilt_remote_provider_does_not_raise(reopened):
    """The regression: this raised ``TypeError: ... keyword argument 'input'``.

    ``ChromaClient.search`` takes the ``query_texts`` branch here (the supplied
    embedding function is not one of ours), so ChromaDB itself calls
    ``embed_query(input=[...])`` on the provider it rebuilt from the config.
    """
    results = reopened.search(query_texts=["alpha"], n_results=2)

    assert results["ids"] == [["A", "B"]]
    assert _StubRemoteEF.calls == [(["alpha"], True)], (
        "the query must reach the rebuilt remote provider exactly once, flagged as a query"
    )
