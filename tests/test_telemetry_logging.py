"""Tests for API load & latency telemetry logging in RemoteEmbeddingFunction."""

import logging
from typing import Any
from zotero_mcp.embeddings.base import RemoteEmbeddingFunction

class DummyTelemetryEmbeddingFunction(RemoteEmbeddingFunction):
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self._init_common(
            model_name="dummy-model",
            base_url=None,
            request_batch_size=16,
            rate_limit_rps=None,
            max_parallel_requests=1,
            max_retries=2,
            tokens_per_minute=None,
        )

    def name(self) -> str:
        return "dummy_telemetry"

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> tuple[list[list[float]], dict[str, str]]:
        vectors = [[0.1] * 8 for _ in texts]
        return vectors, self.headers


def test_telemetry_logging_parses_headers_and_calculates_load(caplog):
    headers = {
        "x-ratelimit-remaining-tokens": "200000",
        "x-ratelimit-limit-tokens": "300000",
        "x-ratelimit-remaining-requests": "4000",
        "x-ratelimit-limit-requests": "5000",
    }
    ef = DummyTelemetryEmbeddingFunction(headers=headers)

    with caplog.at_level(logging.INFO):
        res = ef(["doc1", "doc2"])

    assert len(res) == 2
    assert "dummy_telemetry API" in caplog.text
    assert "2 chunks" in caplog.text
    assert "Token Load: 33.3%" in caplog.text
    assert "200,000/300,000 left" in caplog.text
    assert "Req Load: 20.0%" in caplog.text
    assert "4,000/5,000 left" in caplog.text


def test_telemetry_logging_handles_missing_headers_gracefully(caplog):
    ef = DummyTelemetryEmbeddingFunction(headers={})

    with caplog.at_level(logging.INFO):
        res = ef(["doc1"])

    assert len(res) == 1
    assert "dummy_telemetry API" in caplog.text
    assert "1 chunks" in caplog.text
    assert "Token Load:" not in caplog.text
