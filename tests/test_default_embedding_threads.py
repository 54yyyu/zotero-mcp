"""``ZOTERO_ONNX_THREADS`` reaches the ONNX session of the default model (#90).

Left to itself, ONNX Runtime sizes its thread pool to the physical cores and
pins each worker thread to one of them. In a container's cpuset, a Slurm job or
under taskset, pinning to the other cores fails and each new session logs a
``pthread_setaffinity_np failed`` error for every thread it could not pin. A
positive thread count set explicitly turns the pinning off.

Through a real ChromaDB collection, because the session that matters is the
one ChromaDB embeds with: it rebuilds the "default" embedding function by name
from the collection config rather than calling the one it was handed. Offline:
the model download, the tokenizer and the ONNX session are stand-ins.
"""

import importlib.util
import sys

import numpy as np
import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import chromadb  # noqa: E402
import onnxruntime  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from chromadb.utils.embedding_functions import (  # noqa: E402
    DefaultEmbeddingFunction,
    ONNXMiniLM_L6_V2,
    known_embedding_functions,
)

from zotero_mcp.embeddings.registry import create_embedding_function  # noqa: E402


class _Session:
    """Records the thread count each session is opened with."""

    threads: list[int] = []

    def __init__(self, path, providers=None, sess_options=None):
        _Session.threads.append(sess_options.intra_op_num_threads)

    def run(self, output_names, feed):
        batch, length = feed["input_ids"].shape
        return [np.ones((batch, length, 384), dtype=np.float32)]


class _Tokenizer:
    def encode(self, text):
        class Encoding:
            ids = [101, 102]
            attention_mask = [1, 1]

        return Encoding()


@pytest.mark.parametrize(
    "env, threads",
    [
        (None, 0),  # unset: left to ONNX Runtime
        ("2", 2),
        ("two", 0),  # not a number: left to ONNX Runtime
    ],
)
def test_default_embedding_takes_threads_from_zotero_onnx_threads(env, threads, monkeypatch, tmp_path):
    if env is None:
        monkeypatch.delenv("ZOTERO_ONNX_THREADS", raising=False)
    else:
        monkeypatch.setenv("ZOTERO_ONNX_THREADS", env)
    monkeypatch.setattr(ONNXMiniLM_L6_V2, "_download_model_if_not_exists", lambda self: None)
    monkeypatch.setattr(ONNXMiniLM_L6_V2, "tokenizer", _Tokenizer())
    monkeypatch.setattr(onnxruntime, "InferenceSession", _Session)
    monkeypatch.setattr(_Session, "threads", [])
    # ChromaDB registers the class of the function a collection is opened with,
    # so building the index below with its own class claims "default" again.
    monkeypatch.setitem(known_embedding_functions, "default", known_embedding_functions["default"])

    client = chromadb.PersistentClient(path=str(tmp_path), settings=Settings(anonymized_telemetry=False))
    # Built with ChromaDB's own class, as every default index from zotero-mcp 0.17 and earlier.
    client.create_collection("papers", embedding_function=DefaultEmbeddingFunction())
    collection = client.get_or_create_collection(
        "papers", embedding_function=create_embedding_function("default", {})
    )
    collection.add(documents=["Attention is all you need"], ids=["A"])
    collection.query(query_texts=["attention"], n_results=1)

    assert _Session.threads and set(_Session.threads) == {threads}
