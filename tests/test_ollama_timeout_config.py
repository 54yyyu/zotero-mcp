"""Regression tests for issue #423.

``semantic_search.embedding_config.timeout`` never reached
``OllamaEmbeddingFunction``: the ollama branch of ``_create_embedding_function``
built the function with only ``model_name`` and ``base_url``, so a configured
timeout silently fell back to ``DEFAULT_TIMEOUT`` (120s). With chunking enabled
each ``/api/embed`` call carries a whole item batch worth of chunks, which on a
modest GPU runs well past 120s — so every batch timed out, was queued for
retry, and the run finished having written nothing.

The second half of the fix bounds the request itself: the indexer hands the
embedding function (items x max_chunks_per_item) documents, and sending that as
one HTTP call means a single request has to outlast the entire GPU pass. These
tests pin both the passthrough and the chunking.
"""

import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("chromadb")

from zotero_mcp.chroma_client import (  # noqa: E402
    ChromaClient,
    OllamaEmbeddingFunction,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _client(embedding_config):
    """A ChromaClient without __init__'s PersistentClient side effects.

    ``_create_embedding_function`` reads only these two attributes, so bypassing
    __init__ keeps the test to the branch under test instead of standing up a
    real on-disk ChromaDB.
    """
    client = ChromaClient.__new__(ChromaClient)
    client.embedding_model = "ollama"
    client.embedding_config = embedding_config
    return client


class TestTimeoutPassthrough:
    def test_configured_timeout_reaches_the_embedding_function(self):
        ef = _client({"timeout": 3600})._create_embedding_function()
        assert ef.timeout == 3600, (
            "embedding_config.timeout must reach OllamaEmbeddingFunction; "
            "dropping it silently pins every request to DEFAULT_TIMEOUT"
        )

    def test_timeout_defaults_when_unset(self):
        ef = _client({})._create_embedding_function()
        assert ef.timeout == OllamaEmbeddingFunction.DEFAULT_TIMEOUT

    def test_configured_timeout_is_used_on_the_wire(self, monkeypatch):
        seen = []

        def fake_post(url, json=None, timeout=None):
            seen.append(timeout)
            return _FakeResponse({"embeddings": [[0.1, 0.2]]})

        monkeypatch.setattr(requests, "post", fake_post)
        _client({"timeout": 3600})._create_embedding_function()(["alpha"])

        assert seen == [3600], "the configured timeout must be the HTTP timeout"

    def test_model_name_and_base_url_still_forwarded(self):
        ef = _client(
            {"model_name": "nomic-embed-text", "base_url": "http://gpu:11434"}
        )._create_embedding_function()
        assert ef.model_name == "nomic-embed-text"
        assert ef.base_url == "http://gpu:11434"


class TestRequestBatching:
    def test_large_input_is_split_across_requests(self, monkeypatch):
        sent = []

        def fake_post(url, json=None, timeout=None):
            window = json["input"]
            sent.append(list(window))
            return _FakeResponse({"embeddings": [[float(len(t))] for t in window]})

        monkeypatch.setattr(requests, "post", fake_post)

        ef = OllamaEmbeddingFunction(request_batch_size=10)
        texts = [f"doc-{i}" for i in range(25)]
        ef(texts)

        assert [len(w) for w in sent] == [10, 10, 5]
        # Every document goes out exactly once, in the caller's order.
        assert [t for w in sent for t in w] == texts

    def test_vectors_are_returned_in_input_order(self, monkeypatch):
        def fake_post(url, json=None, timeout=None):
            # Encode each input's index so misordering is detectable.
            return _FakeResponse(
                {"embeddings": [[float(t.split("-")[1]), 0.0] for t in json["input"]]}
            )

        monkeypatch.setattr(requests, "post", fake_post)

        ef = OllamaEmbeddingFunction(request_batch_size=3)
        result = ef([f"doc-{i}" for i in range(7)])

        assert [round(float(vec[0])) for vec in result] == list(range(7))

    def test_configured_request_batch_size_is_honoured(self):
        ef = _client({"request_batch_size": 8})._create_embedding_function()
        assert ef.request_batch_size == 8

    def test_request_batch_size_defaults_when_unset(self):
        ef = _client({})._create_embedding_function()
        assert ef.request_batch_size == OllamaEmbeddingFunction.DEFAULT_REQUEST_BATCH_SIZE

    def test_short_response_raises_instead_of_misaligning(self, monkeypatch):
        # A response with fewer vectors than inputs would shift every later
        # document onto the wrong vector — a corruption that only surfaces
        # much later as inexplicably bad search results.
        monkeypatch.setattr(
            requests,
            "post",
            lambda *a, **k: _FakeResponse({"embeddings": [[0.1, 0.2]]}),
        )

        with pytest.raises(ValueError, match="returned 1 embeddings for 2 inputs"):
            OllamaEmbeddingFunction()(["alpha", "beta"])


class TestConfigRoundTrip:
    def test_timeout_and_batch_size_survive_persist_and_rebuild(self):
        original = OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            base_url="http://gpu:11434",
            timeout=3600,
            request_batch_size=8,
        )
        rebuilt = OllamaEmbeddingFunction.build_from_config(original.get_config())

        assert rebuilt.timeout == 3600
        assert rebuilt.request_batch_size == 8
        assert rebuilt.model_name == "nomic-embed-text"
        assert rebuilt.base_url == "http://gpu:11434"

    def test_config_stays_valid_for_chromadbs_builtin(self):
        # ChromaDB ships an OllamaEmbeddingFunction under the same registry
        # name; if it wins the lookup it rebuilds from our dict and asserts
        # when url/model_name/timeout are missing (#382). Extra keys are read
        # with .get() and ignored, so adding request_batch_size stays safe.
        config = OllamaEmbeddingFunction(timeout=3600).get_config()
        assert {"url", "model_name", "timeout"} <= set(config)
        assert config["timeout"] == 3600
