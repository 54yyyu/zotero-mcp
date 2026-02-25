"""
Compressed markdown storage for late materialization.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover
    zstd = None


class MarkdownStore:
    """Persist and load parsed markdown artifacts."""

    def __init__(self, base_dir: str | None = None):
        if base_dir is None:
            base_dir = str(Path.home() / ".config" / "zotero-mcp" / "md_store")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _validate_key(self, key: str, label: str) -> None:
        """Reject keys containing path separators or traversal sequences."""
        if not key or "/" in key or "\\" in key or ".." in key:
            raise ValueError(f"Invalid {label}: {key!r} contains unsafe characters")

    def _target_path(self, item_key: str, attachment_key: str, doc_hash: str) -> Path:
        self._validate_key(item_key, "item_key")
        self._validate_key(attachment_key, "attachment_key")
        dir_path = self.base_dir / item_key / attachment_key
        dir_path.mkdir(parents=True, exist_ok=True)
        if zstd is not None:
            return dir_path / f"{doc_hash}.md.zst"
        return dir_path / f"{doc_hash}.md"

    def write(self, item_key: str, attachment_key: str, markdown_text: str) -> tuple[str, str]:
        doc_hash = self.content_hash(markdown_text)
        path = self._target_path(item_key, attachment_key, doc_hash)

        # Check if already exists (fast path to avoid redundant work)
        if path.exists():
            return str(path), doc_hash

        raw = markdown_text.encode("utf-8", errors="ignore")
        if zstd is not None:
            data = zstd.ZstdCompressor(level=3).compress(raw)
        else:
            data = raw

        # Use exclusive creation to avoid race conditions with concurrent writers
        try:
            with path.open("xb") as f:
                f.write(data)
        except FileExistsError:
            # Another process created the same content concurrently - that's fine
            pass

        return str(path), doc_hash

    def close(self) -> None:
        """No-op close; MarkdownStore holds no persistent file handles."""
        pass

    def read(self, path: str) -> str:
        p = Path(path).resolve()
        base_resolved = self.base_dir.resolve()
        try:
            p.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path {path!r} is outside the md_store directory")
        data = p.read_bytes()
        if p.suffix == ".zst":
            if zstd is None:
                raise RuntimeError(
                    "Cannot read compressed markdown artifact: 'zstandard' is not installed "
                    f"but file has '.zst' suffix: {p}"
                )
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data).decode("utf-8", errors="ignore")
        return data.decode("utf-8", errors="ignore")

