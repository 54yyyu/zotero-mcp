#!/usr/bin/env python3
"""Developer utility to inspect OpenAI Embeddings API rate limits and throughput capacity.

Reads configuration directly from ~/.config/zotero-mcp/config.json or environment variables.
Useful for developers to check remaining capacity (RPM/TPM quotas and window reset times)
in real-time while a real embeddings generation run (`zotero-mcp update-db`) is active.

Mainly, this is how you find the value for
``semantic_search.embedding_config.tokens_per_minute``. Indexing paces on a
token budget rather than a request rate, because tokens are what bind: at a
64 x ~500-token payload each request costs ~32K tokens, so a 1,000,000 TPM
ceiling caps throughput near 31 requests/minute against a 3,000 RPM
allowance. Read ``x-ratelimit-limit-tokens`` from the output and set the
config key a little under it.

The limiter is seeded from that config value rather than from the headers
themselves, even though OpenAI does send them: headers arrive with a
response, so the first requests of a run would be unpaced, and a provider
that omits them would leave the limiter with no target at all.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"


def load_config(config_path: Path | str | None = None) -> tuple[str | None, str, str]:
    """Load OpenAI API key, model name, and base URL from config file and environment variables."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    config_data = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f) or {}
        except Exception as e:
            print(f"Warning: Could not read config file at {config_path}: {e}", file=sys.stderr)

    embedding_config = (
        config_data.get("semantic_search", {}).get("embedding_config", {})
        if isinstance(config_data.get("semantic_search"), dict)
        else {}
    )

    api_key = embedding_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    model_name = (
        embedding_config.get("model_name")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or os.getenv("ZOTERO_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )
    if model_name == "default":
        model_name = "text-embedding-3-small"

    base_url = (
        embedding_config.get("base_url")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )

    return api_key, model_name, base_url


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Developer script to analyze OpenAI Embeddings API rate limits and throughput. "
            "Useful for checking remaining RPM/TPM capacity in real time while an active embedding generation run is in progress."
        )
    )
    parser.add_argument(
        "--config-path",
        help="Path to config file (default: ~/.config/zotero-mcp/config.json)",
    )
    parser.add_argument(
        "--model",
        help="Override embedding model name (e.g. text-embedding-3-small, text-embedding-3-large)",
    )
    parser.add_argument(
        "--api-key",
        help="Override OpenAI API key",
    )

    args = parser.parse_args()

    api_key, model_name, base_url = load_config(args.config_path)
    if args.api_key:
        api_key = args.api_key
    if args.model:
        model_name = args.model

    if not api_key:
        print("❌ Error: OpenAI API key not found.", file=sys.stderr)
        print("Set 'api_key' in ~/.config/zotero-mcp/config.json or export OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    endpoint_url = base_url.rstrip("/")
    if not endpoint_url.endswith("/embeddings"):
        endpoint_url += "/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": "Developer throughput test string for OpenAI rate limits.",
        "model": model_name,
    }

    print("=" * 65)
    print("OPENAI EMBEDDINGS API RATE LIMIT & THROUGHPUT ANALYZER")
    print("Note: Useful for inspecting remaining capacity while an active embedding")
    print("      generation run (e.g. zotero-mcp update-db) is in progress.")
    print("=" * 65)
    print(f"Target Endpoint: {endpoint_url}")
    print(f"Embedding Model: {model_name}")
    print("Sending probe request...")

    start_time = time.perf_counter()
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Connection Error: {e}", file=sys.stderr)
        sys.exit(1)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"HTTP Status:     {response.status_code}")
    print(f"Latency:         {latency_ms:.1f} ms")

    if not response.ok:
        print(f"❌ API Error Response: {response.text}", file=sys.stderr)
        sys.exit(1)

    rate_limit_headers = {
        k.lower(): v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit-")
    }

    print("\n" + "-" * 65)
    print("RATE LIMIT HEADERS RETURNED BY OPENAI:")
    print("-" * 65)
    if not rate_limit_headers:
        print("No x-ratelimit-* headers found in response.")
    else:
        for k, v in sorted(rate_limit_headers.items()):
            print(f"{k:<32}: {v}")

    if "x-ratelimit-limit-tokens" not in rate_limit_headers:
        print(
            "\nNote: no x-ratelimit-limit-tokens header came back, so the token\n"
            "ceiling cannot be read from the API. Look up your tier's published\n"
            "limit and set semantic_search.embedding_config.tokens_per_minute\n"
            "a little under it (the default assumes the lowest tier)."
        )

    # Theoretical throughput calculations
    limit_req = rate_limit_headers.get("x-ratelimit-limit-requests")
    limit_tok = rate_limit_headers.get("x-ratelimit-limit-tokens")

    if limit_req or limit_tok:
        print("\n" + "-" * 65)
        print("THEORETICAL THROUGHPUT CAPACITY:")
        print("-" * 65)
        if limit_req:
            try:
                rpm = int(limit_req)
                rps = rpm / 60.0
                print(f"• Requests Per Minute (RPM): {rpm:,} ({rps:.1f} req/sec)")
            except ValueError:
                pass
        if limit_tok:
            try:
                tpm = int(limit_tok)
                tps = tpm / 60.0
                print(f"• Tokens Per Minute (TPM):   {tpm:,} ({tps:,.1f} tokens/sec)")
            except ValueError:
                pass
    print("=" * 65)


if __name__ == "__main__":
    main()
