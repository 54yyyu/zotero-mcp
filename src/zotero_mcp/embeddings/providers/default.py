"""ChromaDB's default embedding function, all-MiniLM-L6-v2 on ONNX Runtime."""

import os
from functools import cached_property
from typing import Any

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction as ChromaDefaultEmbeddingFunction,
)
from chromadb.utils.embedding_functions import (
    ONNXMiniLM_L6_V2,
    register_embedding_function,
)


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
        threads = os.getenv("ZOTERO_ONNX_THREADS", "")
        if not threads.isdecimal():
            return super().model
        # ONNXMiniLM_L6_V2.model as of chromadb 1.5, plus intra_op_num_threads.
        options = self.ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = int(threads)
        return self.ort.InferenceSession(
            os.path.join(self.DOWNLOAD_PATH, self.EXTRACTED_FOLDER_NAME, "model.onnx"),
            providers=[p for p in self.ort.get_available_providers() if p != "CoreMLExecutionProvider"],
            sess_options=options,
        )


@register_embedding_function
class DefaultEmbeddingFunction(ChromaDefaultEmbeddingFunction):
    """ChromaDB's default embedding function, embedding through :class:`ONNXMiniLM`.

    Claims ChromaDB's name "default" because ChromaDB sets aside a
    DefaultEmbeddingFunction it is handed and embeds with the one it rebuilds
    by name from the collection config. The persisted config stays ChromaDB's,
    so indexes open the same with either class.
    """

    def __call__(self, input: Documents) -> Embeddings:
        return ONNXMiniLM()(input)

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "DefaultEmbeddingFunction":
        DefaultEmbeddingFunction.validate_config(config)
        return DefaultEmbeddingFunction()
