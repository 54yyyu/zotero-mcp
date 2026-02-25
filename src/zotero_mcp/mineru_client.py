"""
MinerU batch-upload client with token rotation.
"""

from __future__ import annotations

import io
import ipaddress
from urllib.parse import urlparse
import logging
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _validate_external_https_url(url: str, label: str = "URL") -> None:
    """Raise MinerUError if url is not HTTPS or resolves to a private/loopback address."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise MinerUError(f"Invalid {label} scheme: {parsed.scheme!r}. Only HTTPS is allowed.")
    if not parsed.netloc or "@" in parsed.netloc:
        raise MinerUError(f"Invalid {label} netloc: {parsed.netloc!r}")
    hostname = parsed.hostname or ""
    if not hostname:
        raise MinerUError(f"Missing hostname in {label}: {url!r}")
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise MinerUError(f"{label} hostname resolves to a non-public address: {hostname!r}")
    except ValueError:
        # Not a bare IP — check well-known private hostnames
        if hostname.lower() in {"localhost", "localhost.localdomain"}:
            raise MinerUError(f"{label} hostname is not allowed: {hostname!r}")


class MinerUError(Exception):
    """Base MinerU error."""


class RecoverableMinerUError(MinerUError):
    """Error that should trigger token rotation / retry."""


@dataclass
class MinerUConfig:
    tokens: list[str]
    model_version: str = "vlm"
    batch_file_url: str = "https://mineru.net/api/v4/file-urls/batch"
    batch_result_url_template: str = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    poll_interval_seconds: int = 3
    poll_timeout_seconds: int = 300
    max_retries: int = 2
    token_cooldown_seconds: int = 120
    show_progress: bool = False


@dataclass
class MinerULocalApiConfig:
    base_url: str = "http://localhost:8000"
    submit_path: str = "/api/v1/tasks/submit"
    status_path_template: str = "/api/v1/tasks/{task_id}"
    data_path_template: str = "/api/v1/tasks/{task_id}/data"
    backend: str = "vlm"
    lang: str = "ch"
    method: str = "auto"
    formula_enable: bool = True
    table_enable: bool = True
    priority: int = 0
    include_fields: str = "md"
    poll_interval_seconds: int = 2
    poll_timeout_seconds: int = 300
    show_progress: bool = False


class MinerUTokenPool:
    """A tiny in-process token pool with cooldown."""

    def __init__(self, tokens: list[str], cooldown_seconds: int = 120):
        self.tokens = [t for t in tokens if t]
        self.cooldown_seconds = max(1, int(cooldown_seconds))
        self._cooling_until: dict[str, float] = {}

    def _is_available(self, token: str) -> bool:
        return time.time() >= self._cooling_until.get(token, 0)

    def mark_failed(self, token: str) -> None:
        self._cooling_until[token] = time.time() + self.cooldown_seconds

    def next_available(self) -> str | None:
        for token in self.tokens:
            if self._is_available(token):
                return token
        return None


class MinerUBatchClient:
    """Client for MinerU upload-batch parsing flow."""

    def __init__(self, config: MinerUConfig):
        if not config.tokens:
            raise ValueError("MinerU requires at least one token")
        self.config = config
        self.pool = MinerUTokenPool(config.tokens, config.token_cooldown_seconds)

    def _progress(self, message: str) -> None:
        if not self.config.show_progress:
            return
        try:
            sys.stderr.write(f"[MinerU] {message}\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _auth_headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _check_response(self, response: requests.Response) -> None:
        code = response.status_code
        if code < 400:
            return
        if code in (401, 403, 429) or code >= 500:
            raise RecoverableMinerUError(f"Recoverable MinerU error: HTTP {code}")
        # Avoid logging response body as it may contain sensitive information
        raise MinerUError(f"MinerU request failed: HTTP {code}")

    def _create_batch(self, token: str, file_name: str, data_id: str) -> tuple[str, str]:
        self._progress(f"Creating parse batch for '{file_name}'...")
        payload = {
            "files": [{"name": file_name, "data_id": data_id}],
            "model_version": self.config.model_version,
        }
        resp = requests.post(
            self.config.batch_file_url,
            headers=self._auth_headers(token),
            json=payload,
            timeout=30,
            allow_redirects=False,
        )
        self._check_response(resp)
        root = resp.json() or {}
        if int(root.get("code", 0)) != 0:
            raise MinerUError(f"MinerU create-batch returned non-zero code: {root}")
        data = root.get("data", {})
        batch_id = data.get("batch_id")
        urls = data.get("file_urls", [])
        upload_url = ""
        if isinstance(urls, list) and urls:
            first = urls[0]
            if isinstance(first, str):
                upload_url = first
            elif isinstance(first, dict):
                upload_url = first.get("url") or first.get("upload_url") or ""
        if not batch_id or not upload_url:
            raise MinerUError(f"Unexpected MinerU create-batch response: {resp.text[:400]}")
        self._progress(f"Batch created: {batch_id}")
        return str(batch_id), upload_url

    def _upload_file(self, upload_url: str, file_path: Path) -> None:
        _validate_external_https_url(upload_url, "upload_url")

        self._progress(f"Uploading PDF ({file_path.name})...")
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, data=f, timeout=120, allow_redirects=False)
        if resp.status_code >= 400:
            raise RecoverableMinerUError(f"MinerU upload failed: HTTP {resp.status_code}")
        self._progress("Upload complete.")

    def _poll_batch(self, token: str, batch_id: str) -> dict[str, Any]:
        start = time.time()
        poll_url = self.config.batch_result_url_template.format(batch_id=batch_id)
        next_status_log_at = 0.0
        self._progress("Polling parse status...")
        while True:
            resp = requests.get(
                poll_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
                allow_redirects=False,
            )
            self._check_response(resp)
            root = resp.json() or {}
            if int(root.get("code", 0)) not in (0,):
                raise RecoverableMinerUError(f"MinerU poll returned non-zero code: {root.get('code')}")
            data = root.get("data", {})
            results = data.get("extract_result") or data.get("results") or []
            if not results:
                if time.time() - start > self.config.poll_timeout_seconds:
                    raise RecoverableMinerUError("MinerU poll timeout: no result entries")
                elapsed = int(time.time() - start)
                if time.time() >= next_status_log_at:
                    self._progress(f"Still waiting for result... {elapsed}s elapsed")
                    next_status_log_at = time.time() + 15
                time.sleep(self.config.poll_interval_seconds)
                continue
            result = results[0] if isinstance(results, list) else results
            state = str(result.get("state", "")).lower()
            if state == "done":
                self._progress("Parse finished.")
                return result
            if state in {"failed", "error"}:
                raise RecoverableMinerUError(
                    f"MinerU task failed: {result.get('err_msg') or result.get('message') or 'unknown'}"
                )
            if time.time() - start > self.config.poll_timeout_seconds:
                raise RecoverableMinerUError("MinerU poll timeout")
            elapsed = int(time.time() - start)
            if time.time() >= next_status_log_at:
                self._progress(f"Current state: {state or 'processing'} ({elapsed}s elapsed)")
                next_status_log_at = time.time() + 15
            time.sleep(self.config.poll_interval_seconds)

    def _download_zip_bytes(self, zip_url: str) -> bytes:
        _validate_external_https_url(zip_url, "zip_url")

        self._progress("Downloading parsed result package...")
        last_exc: Exception | None = None
        for attempt in range(max(2, self.config.max_retries + 1)):
            try:
                resp = requests.get(zip_url, timeout=120, allow_redirects=False)
                if resp.status_code >= 500:
                    raise RecoverableMinerUError(
                        f"MinerU zip download failed: HTTP {resp.status_code}"
                    )
                if resp.status_code >= 400:
                    raise MinerUError(f"MinerU zip download failed: HTTP {resp.status_code}")
                self._progress("Result download complete.")
                return resp.content
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max(1, self.config.max_retries):
                    self._progress(f"Download retry {attempt + 1}...")
                    time.sleep(min(3, 1 + attempt))
                    continue
                # Fallback to curl for SSL EOF cases seen on some Python/OpenSSL builds.
                try:
                    self._progress("Network fallback: trying curl...")
                    proc = subprocess.run(
                        ["curl", "-fsSL", zip_url],
                        capture_output=True,
                        timeout=120,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout:
                        self._progress("Result download complete (curl fallback).")
                        return proc.stdout
                except Exception:
                    pass
                raise RecoverableMinerUError(f"MinerU zip download network error: {exc}") from exc
        raise RecoverableMinerUError(str(last_exc) if last_exc else "MinerU zip download failed")

    def _extract_best_markdown(self, zip_bytes: bytes) -> str:
        self._progress("Extracting markdown from result package...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            md_names = [n for n in zf.namelist() if n.lower().endswith(".md")]
            if not md_names:
                raise MinerUError("MinerU zip result does not contain markdown files")
            # Use the largest markdown file as the main body.
            md_names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
            with zf.open(md_names[0]) as fp:
                text = fp.read().decode("utf-8", errors="ignore")
            if not text.strip():
                raise MinerUError("MinerU markdown output is empty")
            self._progress(f"Markdown extracted: {len(text)} chars.")
            return text

    def parse_pdf_to_markdown(self, file_path: Path, data_id: str) -> str:
        last_error: Exception | None = None
        max_attempts = max(len(self.config.tokens), 1) * max(self.config.max_retries + 1, 1)
        self._progress(f"Starting MinerU parsing for '{file_path.name}'...")
        for _ in range(max_attempts):
            token = self.pool.next_available()
            if not token:
                # Single-token mode should still retry after transient failures.
                if len(self.config.tokens) == 1:
                    token = self.config.tokens[0]
                    time.sleep(1)
                else:
                    break
            try:
                batch_id, upload_url = self._create_batch(token, file_path.name, data_id)
                self._upload_file(upload_url, file_path)
                result = self._poll_batch(token, batch_id)
                zip_url = result.get("full_zip_url") or result.get("zip_url") or ""
                if not zip_url:
                    raise MinerUError("MinerU result missing full_zip_url")
                zip_bytes = self._download_zip_bytes(zip_url)
                text = self._extract_best_markdown(zip_bytes)
                self._progress("MinerU parsing completed successfully.")
                return text
            except RecoverableMinerUError as exc:
                last_error = exc
                # With a single token, keep retrying instead of cooling it out.
                if len(self.config.tokens) > 1:
                    self.pool.mark_failed(token)
                # Sanitize error message to avoid token leakage in logs
                error_type = exc.__class__.__name__
                self._progress(f"Recoverable error; rotating token/retrying (error type: {error_type})")
                logger.warning("MinerU token temporarily failed; rotating token (error type: %s)", error_type)
            except requests.RequestException as exc:
                last_error = exc
                if len(self.config.tokens) > 1:
                    self.pool.mark_failed(token)
                # Sanitize error message to avoid token leakage in logs
                error_type = exc.__class__.__name__
                self._progress(f"Network error; rotating token/retrying (error type: {error_type})")
                logger.warning("MinerU network error; rotating token (error type: %s)", error_type)
            except Exception as exc:
                # Non recoverable parser errors: stop fast and fall back locally.
                raise MinerUError(str(exc)) from exc
        if last_error:
            error_type = last_error.__class__.__name__
            raise RecoverableMinerUError(
                f"MinerU request failed after multiple attempts (last error type: {error_type})"
            )
        raise RecoverableMinerUError("No available MinerU token")


class MinerULocalApiClient:
    """Client for local MinerU docker API flow."""

    def __init__(self, config: MinerULocalApiConfig):
        self.config = config
        self.base_url = (config.base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise ValueError("MinerU local_api requires a non-empty base_url")

    def _progress(self, message: str) -> None:
        if not self.config.show_progress:
            return
        try:
            sys.stderr.write(f"[MinerU:local] {message}\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    @staticmethod
    def _extract_md(root: dict[str, Any]) -> str:
        data = root.get("data")
        if isinstance(data, dict):
            # Local API status endpoint usually returns markdown in data.content.
            for key in ("content", "md", "markdown", "text"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        # Rare responses may return markdown at top level.
        for key in ("md", "markdown", "text"):
            value = root.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _submit(self, file_path: Path) -> str:
        url = self._url(self.config.submit_path)
        self._progress(f"Submitting task for '{file_path.name}'...")
        with open(file_path, "rb") as f:
            resp = requests.post(
                url,
                files={"file": (file_path.name, f, "application/pdf")},
                data={
                    "backend": self.config.backend,
                    "lang": self.config.lang,
                    "method": self.config.method,
                    "formula_enable": str(bool(self.config.formula_enable)).lower(),
                    "table_enable": str(bool(self.config.table_enable)).lower(),
                    "priority": str(int(self.config.priority)),
                },
                timeout=60,
            )
        if resp.status_code >= 500:
            raise RecoverableMinerUError(f"Local MinerU submit failed: HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise MinerUError(f"Local MinerU submit failed: HTTP {resp.status_code}")
        root = resp.json() or {}
        task_id = root.get("task_id")
        if not task_id and isinstance(root.get("data"), dict):
            task_id = root["data"].get("task_id")
        if not task_id:
            raise MinerUError("Local MinerU submit response missing task_id")
        return str(task_id)

    def _fetch_data_md(self, task_id: str) -> str:
        path = self.config.data_path_template.format(task_id=task_id)
        url = self._url(path)
        resp = requests.get(
            url,
            params={"include_fields": self.config.include_fields, "include_metadata": "true"},
            timeout=30,
        )
        if resp.status_code >= 500:
            raise RecoverableMinerUError(f"Local MinerU data fetch failed: HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise MinerUError(f"Local MinerU data fetch failed: HTTP {resp.status_code}")
        return self._extract_md(resp.json() or {})

    def _poll(self, task_id: str) -> str:
        path = self.config.status_path_template.format(task_id=task_id)
        url = self._url(path)
        start = time.time()
        next_status_log_at = 0.0
        self._progress("Polling local task status...")
        while True:
            resp = requests.get(url, timeout=30)
            if resp.status_code >= 500:
                raise RecoverableMinerUError(f"Local MinerU status failed: HTTP {resp.status_code}")
            if resp.status_code >= 400:
                raise MinerUError(f"Local MinerU status failed: HTTP {resp.status_code}")
            root = resp.json() or {}
            status = str(root.get("status", "")).strip().lower()
            if status == "completed":
                text = self._extract_md(root)
                if text.strip():
                    return text
                text = self._fetch_data_md(task_id)
                if text.strip():
                    return text
                raise MinerUError("Local MinerU completed but markdown is empty")
            if status in {"failed", "error", "cancelled"}:
                raise RecoverableMinerUError(
                    f"Local MinerU task failed: {root.get('error_message') or root.get('message') or status}"
                )
            if time.time() - start > self.config.poll_timeout_seconds:
                raise RecoverableMinerUError("Local MinerU poll timeout")
            elapsed = int(time.time() - start)
            if time.time() >= next_status_log_at:
                self._progress(f"Current local status: {status or 'pending'} ({elapsed}s elapsed)")
                next_status_log_at = time.time() + 15
            time.sleep(max(1, int(self.config.poll_interval_seconds)))

    def parse_pdf_to_markdown(self, file_path: Path, data_id: str) -> str:
        del data_id  # local API does not use data_id
        self._progress(f"Starting local MinerU parsing for '{file_path.name}'...")
        try:
            task_id = self._submit(file_path)
            text = self._poll(task_id)
            self._progress("Local MinerU parsing completed successfully.")
            return text
        except requests.RequestException as exc:
            raise RecoverableMinerUError(f"Local MinerU network error: {exc.__class__.__name__}") from exc
