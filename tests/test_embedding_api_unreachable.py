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
# Providers classify the failure
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
def test_openai_unreachable_errors_raise_named_error(factory, reason):
    ef = make_openai_ef(always(factory))

    with pytest.raises(EmbeddingAPIUnreachable) as info:
        ef(["some text"])

    err = info.value
    assert err.host == HOST
    assert err.reason == reason
    message = str(err)
    assert HOST in message
    assert "VPN" in message
    assert API_KEY not in message
    assert BASE_URL not in message


def test_openai_401_and_403_are_not_retried():
    ef = make_openai_ef(always(lambda: status_error(openai.PermissionDeniedError, 403)))

    with pytest.raises(EmbeddingAPIUnreachable):
        ef(["some text"])

    assert len(ef.client.embeddings.calls) == 1


def test_openai_connection_error_is_retried_before_giving_up():
    ef = make_openai_ef(always(connection_error), max_retries=3)

    with pytest.raises(EmbeddingAPIUnreachable):
        ef(["some text"])

    assert len(ef.client.embeddings.calls) == 4


def test_openai_other_errors_keep_their_type():
    ef = make_openai_ef(always(lambda: status_error(openai.BadRequestError, 400)))

    with pytest.raises(openai.BadRequestError):
        ef(["some text"])


def test_default_host_when_no_base_url():
    ef = OpenAIEmbeddingFunction(api_key=API_KEY)
    ef.client = FakeOpenAIClient(always(connection_error))
    ef.limiter = _no_wait_limiter()
    ef.max_retries = 0

    with pytest.raises(EmbeddingAPIUnreachable) as info:
        ef(["x"])
    assert info.value.host == "api.openai.com"


def test_host_never_carries_credentials():
    ef = make_openai_ef(always(connection_error))
    ef.base_url = "https://user:pa55word@embed.uni.example:8443/v1?key=abc"
    ef.max_retries = 0

    with pytest.raises(EmbeddingAPIUnreachable) as info:
        ef(["x"])
    assert info.value.host == HOST
    assert "pa55word" not in str(info.value) and "abc" not in str(info.value)


def test_gemini_classifies_transport_and_auth_errors():
    pytest.importorskip("google.genai")
    from google.genai import errors

    from zotero_mcp.embeddings.providers.gemini import GeminiEmbeddingFunction

    ef = GeminiEmbeddingFunction(api_key=API_KEY)
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/x")
    assert ef._unreachable_reason(httpx.ConnectError("down", request=request)) == "connection failed"
    assert ef._unreachable_reason(httpx.ConnectTimeout("slow", request=request)) == "timed out"
    assert ef._unreachable_reason(errors.ClientError(403, {"error": {"message": "no"}})) == "HTTP 403"
    assert ef._unreachable_reason(errors.ClientError(401, {"error": {"message": "no"}})) == "HTTP 401"
    assert ef._unreachable_reason(errors.ClientError(400, {"error": {"message": "no"}})) is None
    # A blip during a long run is retried like any other provider's.
    assert ef._classify_error(httpx.ConnectError("down", request=request))[0] is True
    assert ef.api_host() == "generativelanguage.googleapis.com"


def test_ollama_connection_error_names_the_host():
    ef = OllamaEmbeddingFunction(base_url="http://gpu.uni.example:11434", max_retries=1)
    ef.limiter = _no_wait_limiter()
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise requests.exceptions.ConnectionError("refused")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(requests, "post", refuse)
        with pytest.raises(EmbeddingAPIUnreachable) as info:
            ef(["x"])

    assert info.value.host == "gpu.uni.example"
    assert len(calls) == 2


def test_probe_uses_few_retries():
    ef = make_openai_ef(always(connection_error), max_retries=5)

    with pytest.raises(EmbeddingAPIUnreachable):
        ef.check_reachable()

    assert len(ef.client.embeddings.calls) == 2


def test_probe_ignores_errors_that_are_not_reachability():
    ef = make_openai_ef(always(lambda: status_error(openai.BadRequestError, 400)))

    ef.check_reachable()  # does not raise: other errors keep today's handling


def test_error_survives_the_chromadb_worker_boundary():
    """With ``idle_unload_minutes`` set, a search runs ChromaDB in a worker
    process and its exceptions come back pickled; the type and host must
    survive so the server still recognises an unreachable API."""
    import pickle

    original = EmbeddingAPIUnreachable("openai", HOST, "connection failed")
    copy = pickle.loads(pickle.dumps(original))

    assert type(copy) is EmbeddingAPIUnreachable
    assert (copy.provider, copy.host, copy.reason) == ("openai", HOST, "connection failed")
    assert str(copy) == str(original)
