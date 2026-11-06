"""Remote embedding providers report an unreachable API as such.

A university API may answer only on the campus network. Off it, every
request fails with a connection error, a timeout, a 401 or a 403. Each remote
provider turns those failures, once retries are spent, into
``EmbeddingAPIUnreachable``, which names the host but never the key or the
full URL. Other errors keep their own type.

Provider clients are fakes that raise the SDKs' real exception types. Nothing
here makes a network call. ``test_update_stops_when_api_unreachable.py``
covers what an update, a search and the status tools do with the error.
"""

from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")
openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")
requests = pytest.importorskip("requests")

from zotero_mcp.embeddings.base import EmbeddingAPIUnreachable  # noqa: E402
from zotero_mcp.embeddings.providers.ollama import OllamaEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction  # noqa: E402
from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter  # noqa: E402

API_KEY = "sk-secret-test-key-123"
BASE_URL = "https://embed.uni.example/v1"
HOST = "embed.uni.example"


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _no_wait_limiter():
    clock = FakeClock()
    return AdaptiveRateLimiter(clock=clock.time, sleep=clock.sleep)


def _request():
    return httpx.Request("POST", f"{BASE_URL}/embeddings")


def connection_error():
    return openai.APIConnectionError(request=_request())


def timeout_error():
    return openai.APITimeoutError(request=_request())


def status_error(cls, code):
    response = httpx.Response(code, request=_request())
    return cls("refused", response=response, body=None)


class _Data:
    def __init__(self, embedding):
        self.embedding = embedding


class _Response:
    def __init__(self, n):
        self.data = [_Data([0.1, 0.2, 0.3]) for _ in range(n)]


class FakeEmbeddings:
    """``client.embeddings`` with no ``with_raw_response``: plain create()."""

    def __init__(self, script):
        # script(call_number, inputs) -> Exception to raise, or None to succeed
        self.script = script
        self.calls = []

    def create(self, **request):
        self.calls.append(list(request["input"]))
        outcome = self.script(len(self.calls), request["input"])
        if outcome is not None:
            raise outcome
        return _Response(len(request["input"]))


class FakeOpenAIClient:
    def __init__(self, script):
        self.embeddings = FakeEmbeddings(script)


def make_openai_ef(script, **kwargs):
    ef = OpenAIEmbeddingFunction(api_key=API_KEY, base_url=BASE_URL, **kwargs)
    ef.client = FakeOpenAIClient(script)
    ef.limiter = _no_wait_limiter()
    return ef


def always(exc_factory):
    return lambda n, inputs: exc_factory()


# ---------------------------------------------------------------------------
# The probe and the query path name the failure; a run's documents do not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, reason",
    [
        (connection_error, "connection failed"),
        (timeout_error, "timed out"),
        (lambda: status_error(openai.AuthenticationError, 401), "HTTP 401"),
        (lambda: status_error(openai.PermissionDeniedError, 403), "HTTP 403"),
    ],
)
def test_probe_names_the_host_and_reason(factory, reason):
    ef = make_openai_ef(always(factory), max_retries=5)

    with pytest.raises(EmbeddingAPIUnreachable) as info:
        ef.check_reachable()

    assert (info.value.host, info.value.reason) == (HOST, reason)
    assert API_KEY not in str(info.value) and BASE_URL not in str(info.value)
    assert len(ef.client.embeddings.calls) <= 2


def test_probe_ignores_errors_that_are_not_reachability():
    ef = make_openai_ef(always(lambda: status_error(openai.BadRequestError, 400)))

    ef.check_reachable()  # other errors are left to the run, as before


def test_host_never_carries_credentials():
    ef = make_openai_ef(always(connection_error))
    ef.base_url = "https://user:pa55word@embed.uni.example:8443/v1?key=abc"

    with pytest.raises(EmbeddingAPIUnreachable) as info:
        ef.check_reachable()
    assert info.value.host == HOST
    assert "pa55word" not in str(info.value) and "abc" not in str(info.value)


def test_default_host_when_no_base_url():
    ef = OpenAIEmbeddingFunction(api_key=API_KEY)
    ef.client = FakeOpenAIClient(always(connection_error))
    ef.limiter = _no_wait_limiter()

    with pytest.raises(EmbeddingAPIUnreachable) as info:
        ef.check_reachable()
    assert info.value.host == "api.openai.com"


def test_query_gives_up_after_the_probe_budget():
    ef = make_openai_ef(always(connection_error), max_retries=5)

    with pytest.raises(EmbeddingAPIUnreachable):
        ef.embed_query_text("query")

    assert len(ef.client.embeddings.calls) == ef.probe_max_retries + 1


def test_document_requests_keep_their_error_and_retries():
    ef = make_openai_ef(always(connection_error), max_retries=2)

    with pytest.raises(openai.APIConnectionError):
        ef(["some text"])

    assert len(ef.client.embeddings.calls) == 3


def test_ollama_on_localhost_says_the_server_is_not_running():
    ef = OllamaEmbeddingFunction(base_url="http://localhost:11434", max_retries=0)
    ef.limiter = _no_wait_limiter()

    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(requests, "post", refuse)
        with pytest.raises(EmbeddingAPIUnreachable) as info:
            ef.check_reachable()

    assert info.value.host == "localhost"
    assert "running" in str(info.value) and "VPN" not in str(info.value)


def test_gemini_classifies_transport_and_auth_errors():
    pytest.importorskip("google.genai")
    from google.genai import errors

    from zotero_mcp.embeddings.providers.gemini import GeminiEmbeddingFunction

    ef = GeminiEmbeddingFunction(api_key=API_KEY)
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/x")
    assert ef._unreachable_reason(httpx.ConnectError("down", request=request)) == "connection failed"
    assert ef._unreachable_reason(httpx.ConnectTimeout("slow", request=request)) == "timed out"
    assert ef._unreachable_reason(errors.ClientError(403, {"error": {"message": "no"}})) == "HTTP 403"
    assert ef._unreachable_reason(errors.ClientError(400, {"error": {"message": "no"}})) is None
    assert ef.api_host() == "generativelanguage.googleapis.com"


def test_error_survives_the_chromadb_worker_boundary():
    import pickle

    original = EmbeddingAPIUnreachable("openai", HOST, "connection failed")
    copy = pickle.loads(pickle.dumps(original))

    assert type(copy) is EmbeddingAPIUnreachable
    assert (copy.provider, copy.host, copy.reason) == ("openai", HOST, "connection failed")
    assert str(copy) == str(original)


# ---------------------------------------------------------------------------
# An update stops before embedding anything; a search points at keyword search
# ---------------------------------------------------------------------------


class FakeZotero:
    def __init__(self, keys, version=7):
        self.keys = keys
        self.version = version

    def last_modified_version(self):
        return self.version

    def item_versions(self, **kwargs):
        return {k: 1 for k in self.keys}


def _items(n):
    return [
        {
            "key": f"ITEM{i:04d}",
            "data": {"title": f"Title {i}", "itemType": "journalArticle",
                     "abstractNote": f"Abstract {i}", "creators": []},
        }
        for i in range(n)
    ]


def _build(tmp_path, monkeypatch, script, items):
    import json

    from zotero_mcp import semantic_search
    from zotero_mcp.chroma_client import ChromaClient

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"semantic_search": {"update_config": {
        "auto_update": True, "update_frequency": "startup", "last_update": "2026-01-01T00:00:00",
    }}}))
    chroma = ChromaClient(
        collection_name="unreachable_test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="openai",
        embedding_config={"api_key": API_KEY, "base_url": BASE_URL, "request_batch_size": 10},
    )
    ef = chroma.embedding_function
    ef.client = FakeOpenAIClient(script)
    ef.limiter = _no_wait_limiter()
    monkeypatch.setattr(
        semantic_search, "get_zotero_client", lambda: FakeZotero([i["key"] for i in items])
    )
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma, config_path=str(config))
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kw: list(items))
    return search, chroma, ef, config


def _saved(config):
    import json

    return json.loads(config.read_text())["semantic_search"]


@pytest.mark.parametrize("force_full_rebuild", [False, True])
def test_unreachable_api_stops_the_update_before_anything_changes(tmp_path, monkeypatch, force_full_rebuild):
    search, chroma, ef, config = _build(tmp_path, monkeypatch, always(connection_error), _items(30))
    reset = []
    monkeypatch.setattr(chroma, "reset_collection", lambda: reset.append(True))

    stats = search.update_database(force_full_rebuild=force_full_rebuild, allow_mass_deletion=True)

    assert HOST in stats["error"] and "again" in stats["error"]
    assert API_KEY not in stats["error"]
    assert ef.client.embeddings.calls == [["zotero-mcp connectivity check"]] * 2
    assert reset == [], "a rebuild must not empty the index before the API answers"
    assert _saved(config)["update_config"]["last_update"] == "2026-01-01T00:00:00"
    assert not _saved(config).get("last_sync_versions")


def test_nothing_to_embed_makes_no_provider_call(tmp_path, monkeypatch):
    search, chroma, ef, config = _build(tmp_path, monkeypatch, always(connection_error), [])

    stats = search.update_database()

    assert "error" not in stats
    assert ef.client.embeddings.calls == []


def test_semantic_search_suggests_keyword_search(tmp_path, monkeypatch):
    search, chroma, ef, config = _build(tmp_path, monkeypatch, always(connection_error), [])
    chroma.collection.add(ids=["A"], documents=["x"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"group_id": 0}])

    results = search.search("query", limit=3)

    assert results["embedding_api_unreachable"] is True
    assert HOST in results["error"]


@pytest.mark.parametrize("remote", [True, False])
def test_worker_proxy_checks_remote_apis_and_never_loads_a_local_model(monkeypatch, remote):
    from zotero_mcp import chroma_worker
    from zotero_mcp.embeddings import registry

    class Client:
        embedding_model, embedding_config = "openai", {}

        def _request(self, message):
            if not remote:
                raise AttributeError(message[1])
            return ("ok", 1)

    built = []

    class Function:
        def check_reachable(self):
            raise EmbeddingAPIUnreachable("openai", HOST, "connection failed")

    monkeypatch.setattr(registry, "create_embedding_function", lambda *a: built.append(a) or Function())
    proxy = chroma_worker._EmbeddingFunctionProxy(Client())

    if remote:
        with pytest.raises(EmbeddingAPIUnreachable):
            proxy.check_reachable()
    else:
        proxy.check_reachable()
        assert built == []
