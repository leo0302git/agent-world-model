#!/usr/bin/env python3
"""Run parallel AWM scoring against a remote OpenAI-compatible API."""

from __future__ import annotations

import argparse

from awm_parallel_runner import add_common_args, check_required_data_files, config_from_args, run_parallel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    config = config_from_args(args, prefix="awm_api")
    check_required_data_files(config.data)
    summary = run_parallel(config)
    return 1 if summary.get("runner_failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
