#!/usr/bin/env python3
"""Read-only progress monitor for AWM run directories."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def load_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, f"expected object, got {type(data).__name__}"
    return data, None


def task_sort_key(path: Path) -> tuple[str, int, str]:
    scenario = path.parent.name
    task_name = path.name
    try:
        task_id = int(task_name.removeprefix("task_"))
    except ValueError:
        task_id = -1
    return scenario, task_id, str(path)


def scan_run(run_root: Path) -> dict:
    task_dirs = sorted(
        (p for p in run_root.glob("*/task_*") if p.is_dir()),
        key=task_sort_key,
    )
    counts: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    bad_verify: list[tuple[str, str]] = []
    verified = 0

    for task_dir in task_dirs:
        verify_path = task_dir / "verify.code.json"
        if not verify_path.exists():
            verify_path = task_dir / "verify.sql.json"
        if not verify_path.exists():
            continue

        data, error = load_json(verify_path)
        if error:
            bad_verify.append((str(verify_path.relative_to(run_root)), error))
            continue

        verified += 1
        counts[str(data.get("reward_type", "unknown"))] += 1
        modes[str(data.get("mode", "unknown"))] += 1

    return {
        "run_root": str(run_root),
        "task_dirs": len(task_dirs),
        "verified": verified,
        "pending": len(task_dirs) - verified,
        "counts": dict(sorted(counts.items())),
        "modes": dict(sorted(modes.items())),
        "bad_verify": bad_verify,
    }


def print_report(report: dict, limit_bad: int) -> None:
    total = report["verified"]
    complete = report["counts"].get("complete", 0)
    score = complete / total if total else 0.0

    print(f"run_root: {report['run_root']}")
    print(f"task_dirs: {report['task_dirs']}")
    print(f"verified: {report['verified']}")
    print(f"pending: {report['pending']}")
    print(f"score: {score:.4f}")
    print(f"counts: {json.dumps(report['counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"modes: {json.dumps(report['modes'], ensure_ascii=False, sort_keys=True)}")

    bad_verify = report["bad_verify"]
    print(f"bad_verify: {len(bad_verify)}")
    for rel_path, error in bad_verify[:limit_bad]:
        print(f"  {rel_path}: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="AWM run root, e.g. outputs/runs/<run_name>")
    parser.add_argument("--watch", action="store_true", help="Refresh periodically")
    parser.add_argument("--interval", type=float, default=30.0, help="Watch interval in seconds")
    parser.add_argument("--limit-bad", type=int, default=10, help="Max invalid verify files to print")
    args = parser.parse_args()

    if not args.run_root.exists():
        raise SystemExit(f"run root does not exist: {args.run_root}")
    if not args.run_root.is_dir():
        raise SystemExit(f"run root is not a directory: {args.run_root}")

    while True:
        report = scan_run(args.run_root)
        print_report(report, args.limit_bad)
        if not args.watch:
            break
        print("")
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
