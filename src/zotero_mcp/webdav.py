"""Helpers for accessing Zotero WebDAV attachment storage."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests

_PLACEHOLDER_PREFIX = "REPLACE_WITH_YOUR_"


class WebDAVNotConfiguredError(RuntimeError):
    """Raised when WebDAV environment variables are missing."""


def _get_env_value(name: str) -> str | None:
    """Return a non-placeholder environment value, if present."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith(_PLACEHOLDER_PREFIX):
        return None
    return value


def get_webdav_config() -> tuple[str, str, str] | None:
    """Return configured WebDAV credentials, or None when incomplete."""
    base_url = _get_env_value("ZOTERO_WEBDAV_URL")
    username = _get_env_value("ZOTERO_WEBDAV_USERNAME")
    password = _get_env_value("ZOTERO_WEBDAV_PASSWORD")

    if not all((base_url, username, password)):
        return None

    return (base_url.rstrip("/") + "/", username, password)


def is_webdav_configured() -> bool:
    """Return True when direct WebDAV access is configured."""
    return get_webdav_config() is not None


def _select_primary_member(
    members: list[zipfile.ZipInfo], expected_filename: str | None
) -> zipfile.ZipInfo:
    """Pick the most likely primary file from a Zotero WebDAV archive."""
    expected_basename = Path(expected_filename).name.lower() if expected_filename else ""
    expected_suffix = Path(expected_filename).suffix.lower() if expected_filename else ""

    def _score(info: zipfile.ZipInfo) -> tuple[bool, bool, int, int]:
        member_path = PurePosixPath(info.filename)
        basename = member_path.name.lower()
        suffix = member_path.suffix.lower()
        return (
            basename != expected_basename,
            bool(expected_suffix) and suffix != expected_suffix,
            len(member_path.parts),
            len(info.filename),
        )

    return min(members, key=_score)


def _extract_archive(
    archive_source: bytes | str | Path,
    destination_dir: str | Path,
    expected_filename: str | None,
) -> Path:
    """Extract a WebDAV attachment zip and return the primary file path."""
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    archive_file: io.BytesIO | str | Path
    if isinstance(archive_source, bytes):
        archive_file = io.BytesIO(archive_source)
    else:
        archive_file = archive_source

    with zipfile.ZipFile(archive_file) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if not members:
            raise ValueError("WebDAV archive contained no files")

        extracted_paths: dict[str, Path] = {}
        for info in members:
            relative = Path(PurePosixPath(info.filename))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path in WebDAV archive: {info.filename}")

            output_path = destination / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, open(output_path, "wb") as target:
                shutil.copyfileobj(source, target)
            extracted_paths[info.filename] = output_path

        selected = _select_primary_member(members, expected_filename)
        return extracted_paths[selected.filename]


def download_attachment_from_webdav(
    attachment_key: str,
    destination_dir: str | Path,
    expected_filename: str | None = None,
    timeout: float = 30.0,
) -> Path:
    """Download a WebDAV-backed Zotero attachment and return the primary file path."""
    config = get_webdav_config()
    if not config:
        raise WebDAVNotConfiguredError(
            "Missing ZOTERO_WEBDAV_URL / ZOTERO_WEBDAV_USERNAME / ZOTERO_WEBDAV_PASSWORD"
        )

    base_url, username, password = config
    url = f"{base_url}{quote(attachment_key)}.zip"

    session = requests.Session()
    session.auth = (username, password)
    session.trust_env = True

    temp_zip_path = None
    try:
        response = session.get(url, timeout=(10.0, timeout), stream=True)
        if response.status_code == 404:
            raise FileNotFoundError(f"Attachment {attachment_key} was not found in WebDAV storage")
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            temp_zip_path = temp_zip.name
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    temp_zip.write(chunk)
        return _extract_archive(temp_zip_path, destination_dir, expected_filename)
    finally:
        if temp_zip_path and os.path.exists(temp_zip_path):
            os.unlink(temp_zip_path)
        close = getattr(session, "close", None)
        if callable(close):
            close()
