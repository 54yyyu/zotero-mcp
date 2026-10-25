"""Regression test: the default ChromaDB EF must cap ONNX Runtime threads.

Issue #90: on a many-core host with a restricted CPU affinity mask (container,
Slurm job, ``taskset``) ONNX Runtime sizes its thread pool from the *total*
core count and then binds each thread to a core index. Cores outside the
process's allowed mask make ``pthread_setaffinity_np`` fail with
``Invalid argument``, flooding stderr once per thread.

ONNX Runtime does not read ``ONNXRUNTIME_INTRA_OP_NUM_THREADS`` itself — the
consuming application must set ``SessionOptions.intra_op_num_threads``. ChromaDB's
default EF builds its own ``SessionOptions`` and never exposes them, so
``zotero_mcp`` installs a one-time patch on ``onnxruntime.InferenceSession``
when it constructs the default embedding function.

These tests drive the public construction path (``ChromaClient`` ->
``_create_embedding_function``) and then observe the session options ONNX
Runtime actually receives. Nothing here downloads the ONNX model: the session
is built against a nonexistent path, which raises ``NoSuchFile`` *after*
``__init__`` has recorded the thread count, so the options object is still
inspectable.
"""

import os

import pytest

pytest.importorskip("chromadb")
ort = pytest.importorskip("onnxruntime")

from zotero_mcp import chroma_client  # noqa: E402

_ENV_VAR = "ONNXRUNTIME_INTRA_OP_NUM_THREADS"


@pytest.fixture(autouse=True)
def _restore_session_init():
    """Undo any patch so tests do not leak state into each other."""
    original = ort.InferenceSession.__init__
    yield
    ort.InferenceSession.__init__ = original


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)


def _default_ef():
    """Build the default embedding function exactly as production does."""
    client = chroma_client.ChromaClient.__new__(chroma_client.ChromaClient)
    client.embedding_model = "default"
    client.embedding_config = {}
    return client._create_embedding_function()


def _sess_options_after_init():
    """The SessionOptions ORT received when a session was constructed.

    The patched (or original) ``__init__`` mutates the options object before
    attempting to load the model, so a deliberately missing path lets us read
    the thread count back without a real model or a download.
    """
    options = ort.SessionOptions()
    with pytest.raises(Exception):
        ort.InferenceSession(
            "/nonexistent/zotero_mcp_test_model.onnx",
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return options


def test_default_ef_caps_intra_op_threads_at_allowed_cpu_count():
    """Constructing the default EF must pin intra-op threads to the CPU mask."""
    _default_ef()

    options = _sess_options_after_init()

    allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    assert options.intra_op_num_threads == allowed
    assert options.intra_op_num_threads > 0


def test_env_var_overrides_the_affinity_cap(monkeypatch):
    """The user's ``ONNXRUNTIME_INTRA_OP_NUM_THREADS`` must be honoured."""
    monkeypatch.setenv(_ENV_VAR, "8")
    _default_ef()

    options = _sess_options_after_init()

    assert options.intra_op_num_threads == 8


def test_invalid_env_var_falls_back_to_affinity_cap(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "not-a-number")
    _default_ef()

    options = _sess_options_after_init()

    assert options.intra_op_num_threads > 0


def test_explicit_thread_count_is_not_overridden():
    """A caller that chose its own count keeps it."""
    _default_ef()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    with pytest.raises(Exception):
        ort.InferenceSession(
            "/nonexistent/zotero_mcp_test_model.onnx",
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    assert options.intra_op_num_threads == 3


def test_patch_is_idempotent():
    """Constructing the default EF repeatedly must not stack patches."""
    first = _default_ef()
    after_first = ort.InferenceSession.__init__
    _default_ef()
    assert ort.InferenceSession.__init__ is after_first
    assert first.max_input_tokens == 256
