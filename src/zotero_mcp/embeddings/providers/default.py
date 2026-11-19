"""ChromaDB's default embedding function, all-MiniLM-L6-v2 on ONNX Runtime."""

import logging
import os
from functools import cached_property, lru_cache
from typing import Any

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction as ChromaDefaultEmbeddingFunction,
)
from chromadb.utils.embedding_functions import (
    ONNXMiniLM_L6_V2,
    register_embedding_function,
)

logger = logging.getLogger(__name__)

# More threads than this is never useful for a 22M-parameter model, and a huge
# value makes ONNX Runtime raise on every embedding call.
_MAX_THREADS = 256
_warned_bad_value = False


def _configured_threads() -> int:
    """``ZOTERO_ONNX_THREADS`` as a thread count; 0 leaves the session to ONNX Runtime."""
    global _warned_bad_value
    raw = os.getenv("ZOTERO_ONNX_THREADS", "").strip()
    if not raw or raw == "0":
        return 0
    if not (raw.isascii() and raw.isdecimal()):
        if not _warned_bad_value:
            _warned_bad_value = True
            logger.warning(f"ZOTERO_ONNX_THREADS={raw!r} is not a positive whole number; ignoring it")
        return 0
    return min(int(raw), _MAX_THREADS)


class ONNXMiniLM(ONNXMiniLM_L6_V2):
    """ChromaDB's all-MiniLM-L6-v2, with the thread count from ``ZOTERO_ONNX_THREADS`` when set.

    Left to itself, ONNX Runtime sizes its thread pool to the physical cores
    and pins each worker thread to one of them. Where the process may run on
    only some cores (a container's cpuset, a Slurm job, taskset), pinning to
    the others fails and each new session logs an error for every thread it
    could not pin (issue #90). A positive thread count set explicitly turns
    the pinning off.
    """

    @cached_property
    def model(self) -> Any:
        threads = _configured_threads()
        if not threads:
            return super().model
        # ONNXMiniLM_L6_V2.model as of chromadb 1.5, plus intra_op_num_threads.
        options = self.ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        return self.ort.InferenceSession(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
            providers=[p for p in self.ort.get_available_providers() if p != "CoreMLExecutionProvider"],
            sess_options=options,
        )


@lru_cache(maxsize=1)
def _minilm(threads: int) -> ONNXMiniLM:
    """One model per process (per thread setting): the ONNX session is built
    on first use of an instance, and building it cost four fifths of every
    query when each call made a new one."""
    return ONNXMiniLM()


@register_embedding_function
class DefaultEmbeddingFunction(ChromaDefaultEmbeddingFunction):
    """ChromaDB's default embedding function, embedding through :class:`ONNXMiniLM`.

    Claims ChromaDB's name "default" because ChromaDB sets aside a
    DefaultEmbeddingFunction it is handed and embeds with the one it rebuilds
    by name from the collection config. The persisted config stays ChromaDB's,
    so indexes open the same with either class.
    """

    def __call__(self, input: Documents) -> Embeddings:
        return _minilm(_configured_threads())(input)

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "DefaultEmbeddingFunction":
        DefaultEmbeddingFunction.validate_config(config)
        return DefaultEmbeddingFunction()
