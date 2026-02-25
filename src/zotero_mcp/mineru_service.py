"""Orchestrate MinerU providers with fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .mineru_client import (
    MinerUBatchClient,
    MinerUConfig,
    MinerULocalApiClient,
    MinerULocalApiConfig,
)
from .mineru_provider import MinerUProvider

logger = logging.getLogger(__name__)

OFFICIAL_PROVIDER = "official_upload_batch"
LOCAL_PROVIDER = "local_api"


class MinerUService:
    """Provider orchestrator for MinerU parsing."""

    def __init__(self, config: dict[str, Any] | None):
        self.config = dict(config or {})

    def _official_client(self) -> MinerUProvider | None:
        tokens = self.config.get("tokens") or []
        if not isinstance(tokens, list):
            tokens = []
        tokens = [str(t).strip() for t in tokens if str(t).strip()]
        if not tokens:
            return None
        cfg = MinerUConfig(
            tokens=tokens,
            model_version=self.config.get("model_version", "vlm"),
            batch_file_url=self.config.get(
                "batch_file_url", "https://mineru.net/api/v4/file-urls/batch"
            ),
            batch_result_url_template=self.config.get(
                "batch_result_url_template",
                "https://mineru.net/api/v4/extract-results/batch/{batch_id}",
            ),
            poll_interval_seconds=int(self.config.get("poll_interval_seconds", 3)),
            poll_timeout_seconds=int(self.config.get("poll_timeout_seconds", 300)),
            max_retries=int(self.config.get("max_retries", 2)),
            token_cooldown_seconds=int(self.config.get("token_cooldown_seconds", 120)),
        )
        return MinerUBatchClient(cfg)

    def _local_client(self) -> MinerUProvider | None:
        local_cfg_raw = self.config.get("local_api") or {}
        if not isinstance(local_cfg_raw, dict):
            local_cfg_raw = {}
        # Default backend is intentionally "vlm" as requested.
        cfg = MinerULocalApiConfig(
            base_url=str(local_cfg_raw.get("base_url", "http://localhost:8000")).strip(),
            submit_path=str(local_cfg_raw.get("submit_path", "/api/v1/tasks/submit")).strip(),
            status_path_template=str(
                local_cfg_raw.get("status_path_template", "/api/v1/tasks/{task_id}")
            ).strip(),
            data_path_template=str(
                local_cfg_raw.get("data_path_template", "/api/v1/tasks/{task_id}/data")
            ).strip(),
            backend=str(local_cfg_raw.get("backend", "vlm")).strip() or "vlm",
            lang=str(local_cfg_raw.get("lang", "ch")).strip() or "ch",
            method=str(local_cfg_raw.get("method", "auto")).strip() or "auto",
            formula_enable=bool(local_cfg_raw.get("formula_enable", True)),
            table_enable=bool(local_cfg_raw.get("table_enable", True)),
            priority=int(local_cfg_raw.get("priority", 0)),
            include_fields=str(local_cfg_raw.get("include_fields", "md")).strip() or "md",
            poll_interval_seconds=int(local_cfg_raw.get("poll_interval_seconds", 2)),
            poll_timeout_seconds=int(local_cfg_raw.get("poll_timeout_seconds", 300)),
        )
        if not cfg.base_url:
            return None
        return MinerULocalApiClient(cfg)

    def _providers(self) -> list[str]:
        configured = self.config.get("fallback_providers")
        if isinstance(configured, list) and configured:
            values = [str(v).strip().lower() for v in configured if str(v).strip()]
            return [v for v in values if v in {OFFICIAL_PROVIDER, LOCAL_PROVIDER}]
        provider = str(self.config.get("provider", OFFICIAL_PROVIDER)).strip().lower()
        if provider == LOCAL_PROVIDER:
            return [LOCAL_PROVIDER, OFFICIAL_PROVIDER]
        return [OFFICIAL_PROVIDER]

    def parse_pdf_to_markdown(self, file_path: Path, data_id: str) -> str:
        if not self.config.get("enabled"):
            return ""

        for provider in self._providers():
            try:
                if provider == LOCAL_PROVIDER:
                    client = self._local_client()
                elif provider == OFFICIAL_PROVIDER:
                    client = self._official_client()
                else:
                    continue
                if client is None:
                    continue
                text = client.parse_pdf_to_markdown(file_path=file_path, data_id=data_id)
                if text.strip():
                    return text
            except Exception as exc:
                logger.debug("MinerU provider %s failed for %s: %s", provider, file_path, exc)
                continue
        return ""
