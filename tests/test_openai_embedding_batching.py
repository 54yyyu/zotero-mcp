"""OpenAI embedding sub-batching + per-request rate limiting.

Ported from the engine work in #261 (credit @SipengXie2024), combined with the
encoding_format="float" fix from #348. Constructs the embedding function via
__new__ so the tests don't require the optional `openai` package (CI does not
install it).
"""

import base64
import struct
import threading

from zotero_mcp.chroma_client import OpenAIEmbeddingFunction


def _make(batch_size=64, rps=None):
    """Build an OpenAIEmbeddingFunction with a fake client, bypassing __init__."""
    ef = OpenAIEmbeddingFunction.__new__(OpenAIEmbeddingFunction)
    ef.model_name = "text-embedding-3-small"
    ef.base_url = None
    ef.request_batch_size = batch_size
    ef.rate_limit_rps = rps
    ef._rate_lock = threading.Lock()
    ef._last_request_ts = 0.0

    calls = []

    class _Resp:
        def __init__(self, items):
            # Echo each input back as a 1-d vector so order is verifiable.
            self.data = [type("D", (), {"embedding": [float(x)]}) for x in items]

    class _Embeddings:
        @staticmethod
        def create(model, input, encoding_format, extra_body=None):
            calls.append(
                {
                    "input": list(input),
                    "encoding_format": encoding_format,
                    "extra_body": extra_body,
                }
            )
            return _Resp(input)

    class _Client:
        embeddings = _Embeddings()

    ef.client = _Client()
    return ef, calls


def test_subbatches_large_input_preserving_order():
    ef, calls = _make(batch_size=2)
    out = ef([0, 1, 2, 3, 4])
    # 5 inputs at batch_size 2 -> three POSTs of sizes 2, 2, 1
    assert [len(c["input"]) for c in calls] == [2, 2, 1]
    # concatenated output matches input order
    assert out == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_single_request_when_under_cap():
    ef, calls = _make(batch_size=64)
    ef([0, 1, 2])
    assert len(calls) == 1
    assert len(calls[0]["input"]) == 3


def test_encoding_format_is_float_on_every_request():
    ef, calls = _make(batch_size=2)
    ef([0, 1, 2, 3])
    assert calls and all(c["encoding_format"] == "float" for c in calls)


def test_rate_limit_noop_when_unset():
    ef, _ = _make(rps=None)
    # Should return immediately and not raise.
    ef._wait_for_rate_limit()


def test_rate_limit_records_timestamp_when_set():
    ef, _ = _make(rps=1000.0)
    assert ef._last_request_ts == 0.0
    ef._wait_for_rate_limit()
    assert ef._last_request_ts > 0.0


def test_get_config_roundtrips_new_fields():
    ef, _ = _make(batch_size=128, rps=5.0)
    cfg = ef.get_config()
    assert cfg["request_batch_size"] == 128
    assert cfg["rate_limit_rps"] == 5.0
    assert cfg["model_name"] == "text-embedding-3-small"


def test_voyage_base_url_gets_base64_encoding_format():
    # Voyage AI (api.voyageai.com via base_url) rejects encoding_format="float"
    # with a 400; it only accepts "base64". Everything else keeps "float" (#348).
    ef, calls = _make(batch_size=2)
    ef.base_url = "https://api.voyageai.com/v1"
    ef([0, 1])
    assert calls and all(c["encoding_format"] == "base64" for c in calls)


def test_non_voyage_base_url_keeps_float_encoding_format():
    ef, calls = _make(batch_size=2)
    ef.base_url = "https://openrouter.ai/api/v1"
    ef([0, 1])
    assert calls and all(c["encoding_format"] == "float" for c in calls)


def test_voyage_base64_response_is_decoded():
    # Voyage returns each embedding as a base64 str; the provider decodes it to floats.
    ef, calls = _make()
    ef.base_url = "https://api.voyageai.com/v1"
    plain_create = ef.client.embeddings.create

    def create(model, input, encoding_format, extra_body=None):
        resp = plain_create(model, input, encoding_format)
        for d in resp.data:
            d.embedding = base64.b64encode(struct.pack(f"<{len(d.embedding)}f", *d.embedding)).decode()
        return resp

    ef.client.embeddings.create = create
    ef([0.5, 2.0])
    assert [c["encoding_format"] for c in calls] == ["base64"]


def test_voyage_document_call_sends_input_type_document():
    # Voyage optimizes the doc/query pair when input_type labels the role;
    # corpus ingestion (the __call__ path) must send "document" (#667 follow-up).
    ef, calls = _make(batch_size=2)
    ef.base_url = "https://api.voyageai.com/v1"
    ef([0, 1])
    assert calls and all(c["extra_body"] == {"input_type": "document"} for c in calls)


def test_voyage_query_call_sends_input_type_query():
    # The query path (embed_query -> _prepare_query -> is_query=True) must
    # label the request "query" so Voyage optimizes the retrieval direction.
    ef, calls = _make(batch_size=2)
    ef.base_url = "https://api.voyageai.com/v1"
    ef.embed_query([0.5, 2.0])
    assert calls and all(c["extra_body"] == {"input_type": "query"} for c in calls)


def test_non_voyage_requests_omit_input_type():
    # extra_body/input_type is Voyage-only; OpenAI and other OpenAI-compatible
    # backends must not receive unknown parameters.
    ef, calls = _make(batch_size=2)
    ef.base_url = "https://openrouter.ai/api/v1"
    ef([0, 1])
    ef.embed_query([0.5])
    assert calls and all(c["extra_body"] is None for c in calls)
