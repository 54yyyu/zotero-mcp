"""Provider protocol and shared errors for MinerU parsing backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class MinerUProviderError(Exception):
    """Base provider error."""


class RecoverableMinerUProviderError(MinerUProviderError):
    """Provider error that should trigger fallback to the next provider."""


class MinerUProvider(Protocol):
    """Protocol for pluggable MinerU providers."""

    name: str

    def parse_pdf_to_markdown(self, file_path: Path, data_id: str) -> str:
        """Parse one PDF and return markdown text."""
