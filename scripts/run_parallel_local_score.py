#!/usr/bin/env python3
"""Run parallel AWM scoring against an already-running local OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from awm_parallel_runner import add_common_args, check_required_data_files, config_from_args, run_parallel


def check_models_api(api_url: str, api_key: str) -> None:
    req = urllib.request.Request(
        api_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            json.loads(payload)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local model server is not ready at {api_url}/models: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    config = config_from_args(args, prefix="awm_local")
    check_required_data_files(config.data)
    check_models_api(config.api_url, config.api_key)
    summary = run_parallel(config)
    return 1 if summary.get("runner_failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())

