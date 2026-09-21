"""``embed_query`` must honour ChromaDB's EmbeddingFunction contract.

ChromaDB's protocol is ``embed_query(self, input: D) -> Embeddings`` — a
sequence in, a list of vectors out — and ChromaDB calls it *by keyword*
(``embed_query(input=...)`` in ``CollectionCommon._embed``). Our classes used to
define ``embed_query(self, text: str) -> list[float]`` instead, which shadowed
the protocol method and blew up with ``TypeError: ... unexpected keyword
argument 'input'`` whenever ChromaDB — rather than our own
``ChromaClient.search`` — drove the query path. The single-string helper now
lives under the name ``embed_query_text``.

Everything here is offline: no provider is constructed through its real
``__init__``, and ``_embed_batch`` (or, for the local HuggingFace provider,
the sentence-transformers model) is always stubbed.
"""

import inspect

import pytest

pytest.importorskip("chromadb")

import numpy as np  # noqa: E402  (a chromadb dependency)

from zotero_mcp.embeddings.providers import (  # noqa: E402
    CUSTOM_EMBEDDING_FUNCTIONS,
    GeminiEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction,
)

REMOTE_CLASSES = (
    OpenAIEmbeddingFunction,
    GeminiEmbeddingFunction,
    OllamaEmbeddingFunction,
)


class _Model:
    """Stand-in for a sentence-transformers model (HuggingFace's only I/O)."""

    def __init__(self, calls):
        self.calls = calls

    def encode(self, input, convert_to_numpy=True):
        self.calls.append((list(input), False))
        return np.array([[float(len(text)), 1.0] for text in input])


def _rows(embeddings):
    """Plain float lists, whichever row type the provider returned.

    ChromaDB's ``__init_subclass__`` normalizes whatever ``__call__`` returns
    into numpy rows, so the base-class (HuggingFace) path yields ndarrays while
    the remote path yields the lists its ``_embed_batch`` produced.
    """
    return [[float(x) for x in row] for row in embeddings]


def _make(cls, calls):
    """A provider instance that records every embed call, built without I/O.

    ``__new__`` skips ``__init__`` — the pattern the existing provider tests
    already use — so no API client is constructed and no model is downloaded.
    Everything the query path reads resolves to a class attribute or a
    ``getattr(..., default)``; the rate limiter builds itself on first use.
    """
    ef = cls.__new__(cls)
    ef.model_name = "stub-model"
    if cls is HuggingFaceEmbeddingFunction:
        ef.model = _Model(calls)
    else:
        ef._embed_batch = lambda texts, is_query=False: (
            calls.append((list(texts), is_query)) or [[float(len(text)), 1.0] for text in texts]
        )
    return ef


# ---------------------------------------------------------------------------
# 1. The contract: a sequence in, one vector per element out, by keyword.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CUSTOM_EMBEDDING_FUNCTIONS, ids=lambda c: c.__name__)
def test_embed_query_takes_input_by_keyword_and_returns_one_vector_each(cls):
    """The exact call ChromaDB makes: ``embed_query(input=[...])``."""
    calls = []
    ef = _make(cls, calls)

    result = ef.embed_query(input=["a", "bbb"])

    assert _rows(result) == [[1.0, 1.0], [3.0, 1.0]]
    assert [text for call in calls for text in call[0]] == ["a", "bbb"]


@pytest.mark.parametrize("cls", REMOTE_CLASSES, ids=lambda c: c.__name__)
def test_embed_query_routes_through_prepare_query_as_a_query(cls):
    """Query semantics survive the batch wrapper: every element goes through
    ``_prepare_query`` and reaches ``_embed_batch`` with ``is_query=True``.

    This is what would silently degrade search quality if ``embed_query`` were
    implemented by simply calling ``__call__`` on the batch.
    """
    calls = []
    ef = _make(cls, calls)
    ef._prepare_query = lambda text: f"Q<{text}>"
    ef._prepare_document = lambda text: f"D<{text}>"

    ef.embed_query(input=["one", "two"])

    assert [call[0] for call in calls] == [["Q<one>"], ["Q<two>"]]
    assert all(call[1] is True for call in calls)


# ---------------------------------------------------------------------------
# 2. A bare string must fail loudly, not be embedded character by character.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CUSTOM_EMBEDDING_FUNCTIONS, ids=lambda c: c.__name__)
def test_embed_query_rejects_a_bare_string(cls):
    """``ef.embed_query("text")`` was the old spelling; a string is iterable,
    so without this guard it would be embedded one character at a time and
    return nonsense instead of raising."""
    calls = []
    ef = _make(cls, calls)

    with pytest.raises(TypeError, match="embed_query_text"):
        ef.embed_query("hello")

    assert calls == [], "nothing may be embedded before the guard trips"


# ---------------------------------------------------------------------------
# 3. Tripwire for the next provider added to the package.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CUSTOM_EMBEDDING_FUNCTIONS, ids=lambda c: c.__name__)
def test_embed_query_parameter_is_named_input(cls):
    """ChromaDB passes the parameter by keyword, so its *name* is load-bearing.

    A provider that redefines ``embed_query(self, text)`` reintroduces the
    exact defect this module pins, and only fails at query time against a
    persisted collection — long after CI.
    """
    params = list(inspect.signature(cls.embed_query).parameters)[1:]  # drop self
    assert params[:1] == ["input"], (
        f"{cls.__name__}.embed_query must name its first parameter 'input' "
        f"(ChromaDB calls embed_query(input=...)); got {params}"
    )
