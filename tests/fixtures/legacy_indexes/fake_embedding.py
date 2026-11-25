"""The offline embedding function the legacy indexes were built with, and tests reopen them with.

It takes the place of ChromaDB's default embedding (the ONNX MiniLM download) and registers under
the same name, ``default``, so a collection built with it reopens without an "embedding function
conflict" and without a model. It deliberately does not subclass ChromaDB's own
``DefaultEmbeddingFunction``: ChromaDB sets such an instance aside and embeds with the persisted
configuration instead, which would be the real model. Vectors come from :func:`fake_zotero.embed`.
"""

from __future__ import annotations

from chromadb import EmbeddingFunction
from fake_zotero import embed


class FakeDefaultEmbedding(EmbeddingFunction):
    """Named ``default``, with the model replaced by a hashed bag of words."""

    def __init__(self, *args, **kwargs):  # no model, no download
        self.calls: list[list[str]] = []

    def __call__(self, input):
        self.calls.append(list(input))
        return [embed(text) for text in input]

    def truncate(self, text, max_tokens):
        return text

    @staticmethod
    def name():
        return "default"

    @staticmethod
    def build_from_config(config):
        return FakeDefaultEmbedding()

    def get_config(self):
        return {}
