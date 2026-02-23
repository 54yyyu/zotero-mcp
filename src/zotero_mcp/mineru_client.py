"""
MinerU batch-upload client with token rotation.
"""

from __future__ import annotations

import io
import logging
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


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
        raise MinerUError(f"MinerU request failed: HTTP {code} {response.text[:300]}")

    def _create_batch(self, token: str, file_name: str, data_id: str) -> tuple[str, str]:
        payload = {
            "files": [{"name": file_name, "data_id": data_id}],
            "model_version": self.config.model_version,
        }
        resp = requests.post(
            self.config.batch_file_url,
            headers=self._auth_headers(token),
            json=payload,
            timeout=30,
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
        return str(batch_id), upload_url

    def _upload_file(self, upload_url: str, file_path: Path) -> None:
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, data=f, timeout=120)
        if resp.status_code >= 400:
            raise RecoverableMinerUError(f"MinerU upload failed: HTTP {resp.status_code}")

    def _poll_batch(self, token: str, batch_id: str) -> dict[str, Any]:
        start = time.time()
        poll_url = self.config.batch_result_url_template.format(batch_id=batch_id)
        while True:
            resp = requests.get(
                poll_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
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
                time.sleep(self.config.poll_interval_seconds)
                continue
            result = results[0] if isinstance(results, list) else results
            state = str(result.get("state", "")).lower()
            if state == "done":
                return result
            if state in {"failed", "error"}:
                raise RecoverableMinerUError(
                    f"MinerU task failed: {result.get('err_msg') or result.get('message') or 'unknown'}"
                )
            if time.time() - start > self.config.poll_timeout_seconds:
                raise RecoverableMinerUError("MinerU poll timeout")
            time.sleep(self.config.poll_interval_seconds)

    def _download_zip_bytes(self, zip_url: str) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(max(2, self.config.max_retries + 1)):
            try:
                resp = requests.get(zip_url, timeout=120)
                if resp.status_code >= 500:
                    raise RecoverableMinerUError(
                        f"MinerU zip download failed: HTTP {resp.status_code}"
                    )
                if resp.status_code >= 400:
                    raise MinerUError(f"MinerU zip download failed: HTTP {resp.status_code}")
                return resp.content
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max(1, self.config.max_retries):
                    time.sleep(min(3, 1 + attempt))
                    continue
                # Fallback to curl for SSL EOF cases seen on some Python/OpenSSL builds.
                try:
                    proc = subprocess.run(
                        ["curl", "-fsSL", zip_url],
                        capture_output=True,
                        timeout=120,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout:
                        return proc.stdout
                except Exception:
                    pass
                raise RecoverableMinerUError(f"MinerU zip download network error: {exc}") from exc
        raise RecoverableMinerUError(str(last_exc) if last_exc else "MinerU zip download failed")

    def _extract_best_markdown(self, zip_bytes: bytes) -> str:
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
            return text

    def parse_pdf_to_markdown(self, file_path: Path, data_id: str) -> str:
        last_error: Exception | None = None
        max_attempts = max(len(self.config.tokens), 1) * max(self.config.max_retries + 1, 1)
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
                return self._extract_best_markdown(zip_bytes)
            except RecoverableMinerUError as exc:
                last_error = exc
                # With a single token, keep retrying instead of cooling it out.
                if len(self.config.tokens) > 1:
                    self.pool.mark_failed(token)
                logger.warning("MinerU token temporarily failed; rotating token: %s", exc)
            except requests.RequestException as exc:
                last_error = exc
                if len(self.config.tokens) > 1:
                    self.pool.mark_failed(token)
                logger.warning("MinerU network error; rotating token: %s", exc)
            except Exception as exc:
                # Non recoverable parser errors: stop fast and fall back locally.
                raise MinerUError(str(exc)) from exc
        if last_error:
            raise RecoverableMinerUError(str(last_error))
        raise RecoverableMinerUError("No available MinerU token")
