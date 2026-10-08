"""``insecure_skip_verify`` on the OpenAI embedding function.

Some OpenAI-compatible internal gateways (an organization's own reverse
proxy in front of a self-hosted backend, for example) present a certificate
chain missing its intermediate, which no standards-compliant client can
verify. There is no environment variable that disables TLS verification for
httpx/openai -- it is only ever a constructor argument -- so this is an
explicit, narrowly-scoped opt-in rather than a blanket default, off unless
requested via the constructor or ``OPENAI_EMBEDDING_INSECURE_SKIP_VERIFY``.

Covers:
  * off by default, both via the constructor and the env var;
  * the env var accepts the same truthy spellings used elsewhere in this file
    (``1``/``true``/``yes``, case-insensitive);
  * an explicit constructor value always wins over the env var;
  * when on, the constructed ``openai.OpenAI`` client actually got a custom
    ``httpx.Client`` with ``verify=False`` -- not just that the flag is set;
  * it round-trips through ``get_config``/``build_from_config``, so a
    persisted collection rebuilt by name keeps behaving the same way.
"""

import pytest

pytest.importorskip("openai")

from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction  # noqa: E402


def test_off_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    monkeypatch.delenv("OPENAI_EMBEDDING_INSECURE_SKIP_VERIFY", raising=False)
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
    assert ef.insecure_skip_verify is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
def test_env_var_truthy_spellings_enable_it(monkeypatch, value):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    monkeypatch.setenv("OPENAI_EMBEDDING_INSECURE_SKIP_VERIFY", value)
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
    assert ef.insecure_skip_verify is True


@pytest.mark.parametrize("value", ["0", "false", "", "no"])
def test_env_var_falsy_spellings_leave_it_off(monkeypatch, value):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    monkeypatch.setenv("OPENAI_EMBEDDING_INSECURE_SKIP_VERIFY", value)
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
    assert ef.insecure_skip_verify is False


def test_explicit_constructor_arg_overrides_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    monkeypatch.setenv("OPENAI_EMBEDDING_INSECURE_SKIP_VERIFY", "true")
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", insecure_skip_verify=False)
    assert ef.insecure_skip_verify is False


def test_when_on_the_client_actually_has_verify_false(monkeypatch):
    """Not just the flag -- the real openai.OpenAI client's underlying httpx
    client must actually have verify=False, or the whole point is moot."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", insecure_skip_verify=True)
    http_client = ef.client._client  # openai.OpenAI wraps its httpx.Client here
    assert http_client._transport._pool._ssl_context.verify_mode.name == "CERT_NONE"


def test_when_off_the_client_keeps_verification_on(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", insecure_skip_verify=False)
    http_client = ef.client._client
    assert http_client._transport._pool._ssl_context.verify_mode.name != "CERT_NONE"


def test_get_config_roundtrips_the_flag(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small", insecure_skip_verify=True)
    assert ef.get_config()["insecure_skip_verify"] is True


def test_build_from_config_restores_the_flag(monkeypatch):
    """A persisted collection rebuilt by name must keep verify=False too --
    otherwise a query issued after a service restart would fail against the
    exact same endpoint the index was originally built against."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    monkeypatch.delenv("OPENAI_EMBEDDING_INSECURE_SKIP_VERIFY", raising=False)

    ef = OpenAIEmbeddingFunction.build_from_config(
        {"model_name": "text-embedding-3-small", "insecure_skip_verify": True}
    )
    assert ef.insecure_skip_verify is True


def test_build_from_config_defaults_to_env_when_absent(monkeypatch):
    """An older, pre-this-feature persisted config has no such key at all --
    build_from_config's .get() returns None, which must fall back to the env
    var rather than a hard False, so it can't silently disable a running
    deployment's intended insecure mode after a restart."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    monkeypatch.setenv("OPENAI_EMBEDDING_INSECURE_SKIP_VERIFY", "true")

    ef = OpenAIEmbeddingFunction.build_from_config({"model_name": "text-embedding-3-small"})
    assert ef.insecure_skip_verify is True
