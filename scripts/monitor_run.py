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
    code_counts: Counter[str] = Counter()
    sql_counts: Counter[str] = Counter()
    llm_judge_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    bad_verify: list[tuple[str, str]] = []
    bad_trajectory: list[tuple[str, str]] = []
    verified = 0
    code_verified = 0
    sql_verified = 0
    llm_judged = 0
    trajectory_seen = 0

    for task_dir in task_dirs:
        trajectory_path = task_dir / "trajectory.json"
        if trajectory_path.exists():
            trajectory, error = load_json(trajectory_path)
            if error:
                bad_trajectory.append((str(trajectory_path.relative_to(run_root)), error))
            else:
                trajectory_seen += 1
                if trajectory.get("skill_injected") is True:
                    skill_counts["injected"] += 1
                elif trajectory.get("skill_injected") is False:
                    skill_counts["not_injected"] += 1
                else:
                    skill_counts["unknown"] += 1

        task_has_verify = False
        for mode in ("code", "sql"):
            verify_path = task_dir / f"verify.{mode}.json"
            if not verify_path.exists():
                continue

            data, error = load_json(verify_path)
            if error:
                bad_verify.append((str(verify_path.relative_to(run_root)), error))
                continue

            task_has_verify = True
            reward_type = str(data.get("reward_type", "unknown"))
            counts[reward_type] += 1
            modes[str(data.get("mode", mode))] += 1

            if mode == "code":
                code_verified += 1
                code_counts[reward_type] += 1
            else:
                sql_verified += 1
                sql_counts[reward_type] += 1
                judge = data.get("llm_judge")
                if isinstance(judge, dict):
                    classification = str(judge.get("classification", "unknown"))
                    llm_judged += 1
                    llm_judge_counts[classification] += 1

        if task_has_verify:
            verified += 1

    return {
        "run_root": str(run_root),
        "task_dirs": len(task_dirs),
        "verified": verified,
        "pending": len(task_dirs) - verified,
        "counts": dict(sorted(counts.items())),
        "modes": dict(sorted(modes.items())),
        "code_verified": code_verified,
        "code_counts": dict(sorted(code_counts.items())),
        "sql_verified": sql_verified,
        "sql_counts": dict(sorted(sql_counts.items())),
        "llm_judged": llm_judged,
        "llm_judge_counts": dict(sorted(llm_judge_counts.items())),
        "trajectory_seen": trajectory_seen,
        "skill_counts": dict(sorted(skill_counts.items())),
        "bad_verify": bad_verify,
        "bad_trajectory": bad_trajectory,
    }


def print_report(report: dict, limit_bad: int) -> None:
    total = report["verified"]
    complete = report["counts"].get("complete", 0)
    score = complete / total if total else 0.0
    code_total = report["code_verified"]
    code_complete = report["code_counts"].get("complete", 0)
    code_score = code_complete / code_total if code_total else 0.0
    sql_total = report["sql_verified"]
    sql_complete = report["sql_counts"].get("complete", 0)
    sql_score = sql_complete / sql_total if sql_total else 0.0
    judge_total = report["llm_judged"]
    judge_complete = report["llm_judge_counts"].get("complete", 0)
    judge_score = judge_complete / judge_total if judge_total else 0.0
    trajectory_seen = report["trajectory_seen"]
    skill_injected = report["skill_counts"].get("injected", 0)
    skill_rate = skill_injected / trajectory_seen if trajectory_seen else 0.0

    print(f"run_root: {report['run_root']}")
    print(f"task_dirs: {report['task_dirs']}")
    print(f"verified: {report['verified']}")
    print(f"pending: {report['pending']}")
    print(f"score: {score:.4f}  # top-level reward_type across verify files")
    print(f"counts: {json.dumps(report['counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"modes: {json.dumps(report['modes'], ensure_ascii=False, sort_keys=True)}")
    print(f"code_verified: {code_total}")
    print(f"code_score: {code_score:.4f}")
    print(f"code_counts: {json.dumps(report['code_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"sql_verified: {sql_total}")
    print(f"sql_score: {sql_score:.4f}  # SQL verifier top-level reward_type")
    print(f"sql_counts: {json.dumps(report['sql_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"llm_judged: {judge_total}")
    print(f"llm_judge_score: {judge_score:.4f}")
    print(f"llm_judge_counts: {json.dumps(report['llm_judge_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"trajectory_seen: {trajectory_seen}")
    print(f"skill_injected_rate: {skill_rate:.4f}")
    print(f"skill_counts: {json.dumps(report['skill_counts'], ensure_ascii=False, sort_keys=True)}")

    bad_verify = report["bad_verify"]
    print(f"bad_verify: {len(bad_verify)}")
    for rel_path, error in bad_verify[:limit_bad]:
        print(f"  {rel_path}: {error}")

    bad_trajectory = report["bad_trajectory"]
    print(f"bad_trajectory: {len(bad_trajectory)}")
    for rel_path, error in bad_trajectory[:limit_bad]:
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
