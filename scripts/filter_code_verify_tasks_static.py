#!/usr/bin/env python3
"""Static filter for AWM code-verifiable tasks.

This does not run an agent. It checks dataset completeness, verifier syntax,
database reset, and a no-op verifier sanity check where initial_db == final_db.
The output allowlist is intended as the candidate set for later real-agent runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from awm.core.reset import reset_single_database
from awm.core.verify import execute_code_verifier
from awm.tools import normalize_scenario_name


DEFAULT_DATA = Path("/data1/jczhong/datasets/AgentWorldModel-1K")
DEFAULT_OUT = Path("outputs/task_allowlists/static_code_verify")
KNOWN_BAD_TASKS = {("q_a_knowledge_base_1", 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--task-allowlist-jsonl", type=Path,
                        help="Only filter scenario/task_id pairs listed in this JSONL file.")
    parser.add_argument("--task-manifest-json", type=Path,
                        help="Only filter tasks listed in an AWM run manifest.json.")
    parser.add_argument("--keep-complete-noop", action="store_true",
                        help="Do not reject verifiers that mark no-op DBs complete.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def scenario_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        scenario = normalize_scenario_name(str(row.get("scenario", "")))
        if scenario:
            out[scenario] = row
    return out


def verifier_map(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        scenario = normalize_scenario_name(str(row.get("scenario", "")))
        task_idx = row.get("task_idx")
        if scenario and isinstance(task_idx, int):
            out[(scenario, task_idx)] = row
    return out


def iter_tasks(tasks_by_scenario: dict[str, dict[str, Any]], limit_scenarios: int | None):
    count = 0
    for scenario, row in sorted(tasks_by_scenario.items()):
        if limit_scenarios is not None and count >= limit_scenarios:
            break
        tasks = row.get("tasks") or []
        for task_id, task_text in enumerate(tasks):
            yield scenario, task_id, str(task_text)
        count += 1


def load_selected_task_keys(args: argparse.Namespace) -> set[tuple[str, int]] | None:
    selected: set[tuple[str, int]] = set()
    if args.task_allowlist_jsonl:
        with args.task_allowlist_jsonl.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                scenario = normalize_scenario_name(str(row.get("scenario") or row.get("scenario_dir") or ""))
                if not scenario:
                    raise ValueError(f"{args.task_allowlist_jsonl}:{line_no}: missing scenario")
                selected.add((scenario, int(row["task_id"])))
    if args.task_manifest_json:
        manifest = json.loads(args.task_manifest_json.read_text(encoding="utf-8"))
        for idx, row in enumerate(manifest.get("tasks") or []):
            scenario = normalize_scenario_name(str(row.get("scenario") or row.get("scenario_dir") or ""))
            if not scenario:
                raise ValueError(f"{args.task_manifest_json}: tasks[{idx}] missing scenario")
            selected.add((scenario, int(row["task_id"])))
    return selected or None


def compile_verifier(code: str) -> tuple[bool, str | None]:
    if not isinstance(code, str) or len(code.strip()) < 10:
        return False, "missing_or_short_verifier_code"
    try:
        compile(code, "<verifier>", "exec")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as exc:
        return False, f"exec_{type(exc).__name__}: {exc}"
    if not any(name.startswith("verify_") and callable(value) for name, value in namespace.items()):
        return False, "verify_function_not_found"
    return True, None


def reject(
    rejected: list[dict[str, Any]],
    scenario: str,
    task_id: int,
    task: str,
    reason: str,
    details: str | dict[str, Any] | None = None,
) -> None:
    rejected.append({
        "scenario": scenario,
        "task_id": task_id,
        "task": task,
        "reason": reason,
        "details": details,
    })


def main() -> int:
    args = parse_args()
    data = args.data
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "tasks": data / "gen_tasks.jsonl",
        "envs": data / "gen_envs.jsonl",
        "db": data / "gen_db.jsonl",
        "sample": data / "gen_sample.jsonl",
        "verifier": data / "gen_verifier.pure_code.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing dataset files: " + ", ".join(missing))

    tasks_by_scenario = scenario_map(read_jsonl(paths["tasks"]))
    envs_by_scenario = scenario_map(read_jsonl(paths["envs"]))
    db_by_scenario = scenario_map(read_jsonl(paths["db"]))
    sample_by_scenario = scenario_map(read_jsonl(paths["sample"]))
    verifiers = verifier_map(read_jsonl(paths["verifier"]))

    allowlist: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    scenario_db: dict[str, str] = {}
    scenario_db_error: dict[str, str] = {}
    counters: Counter[str] = Counter()
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)

    selected_task_keys = load_selected_task_keys(args)
    tasks = [
        row for row in iter_tasks(tasks_by_scenario, args.limit_scenarios)
        if selected_task_keys is None or (row[0], row[1]) in selected_task_keys
    ]
    with tempfile.TemporaryDirectory(prefix="awm_static_filter_") as tmp:
        db_dir = Path(tmp) / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        for idx, (scenario, task_id, task_text) in enumerate(tasks, 1):
            if args.verbose and (idx == 1 or idx % 100 == 0):
                print(f"[{idx}/{len(tasks)}] {scenario}/task_{task_id}")

            base = {
                "scenario": scenario,
                "scenario_dir": scenario,
                "task_id": task_id,
                "task": task_text,
            }

            if (scenario, task_id) in KNOWN_BAD_TASKS:
                counters["known_bad_task"] += 1
                by_scenario[scenario]["known_bad_task"] += 1
                reject(rejected, scenario, task_id, task_text, "known_bad_task")
                continue

            required_scenario_sources = {
                "missing_env": scenario not in envs_by_scenario,
                "missing_db_schema": scenario not in db_by_scenario,
                "missing_sample": scenario not in sample_by_scenario,
            }
            missing_reasons = [name for name, is_missing in required_scenario_sources.items() if is_missing]
            if missing_reasons:
                reason = ",".join(missing_reasons)
                counters[reason] += 1
                by_scenario[scenario][reason] += 1
                reject(rejected, scenario, task_id, task_text, reason)
                continue

            verifier = verifiers.get((scenario, task_id))
            if verifier is None:
                counters["missing_verifier"] += 1
                by_scenario[scenario]["missing_verifier"] += 1
                reject(rejected, scenario, task_id, task_text, "missing_verifier")
                continue

            code = (verifier.get("verification") or {}).get("code", "")
            ok, error = compile_verifier(code)
            if not ok:
                counters["bad_verifier_compile"] += 1
                by_scenario[scenario]["bad_verifier_compile"] += 1
                reject(rejected, scenario, task_id, task_text, "bad_verifier_compile", error)
                continue

            if scenario not in scenario_db and scenario not in scenario_db_error:
                try:
                    db_path = reset_single_database(
                        input_db=str(paths["db"]),
                        input_sample=str(paths["sample"]),
                        scenario=scenario,
                        database_dir=str(db_dir),
                    )
                    scenario_db[scenario] = db_path
                except Exception as exc:
                    scenario_db_error[scenario] = f"{type(exc).__name__}: {exc}"

            db_error = scenario_db_error.get(scenario)
            if db_error:
                counters["db_reset_error"] += 1
                by_scenario[scenario]["db_reset_error"] += 1
                reject(rejected, scenario, task_id, task_text, "db_reset_error", db_error)
                continue

            db_path = scenario_db[scenario]
            noop_final = Path(tmp) / "noop_final" / f"{scenario}_{task_id}.db"
            noop_final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(db_path, noop_final)
            result = execute_code_verifier(
                code,
                "verify_task_completion",
                db_path,
                str(noop_final),
                final_answer="",
            )
            if result.get("execution_status") != "success":
                counters["noop_verifier_error"] += 1
                by_scenario[scenario]["noop_verifier_error"] += 1
                reject(rejected, scenario, task_id, task_text, "noop_verifier_error", result)
                continue
            if result.get("result") == "complete" and not args.keep_complete_noop:
                counters["noop_complete"] += 1
                by_scenario[scenario]["noop_complete"] += 1
                reject(rejected, scenario, task_id, task_text, "noop_complete", result)
                continue

            counters["accepted"] += 1
            by_scenario[scenario]["accepted"] += 1
            allowlist.append({
                **base,
                "checks": {
                    "has_env": True,
                    "has_db_schema": True,
                    "has_sample": True,
                    "has_verifier": True,
                    "verifier_compile": True,
                    "db_reset": True,
                    "noop_execution_status": result.get("execution_status"),
                    "noop_result": result.get("result"),
                },
            })

    stats = {
        "data": str(data),
        "task_allowlist_jsonl": str(args.task_allowlist_jsonl) if args.task_allowlist_jsonl else None,
        "task_manifest_json": str(args.task_manifest_json) if args.task_manifest_json else None,
        "total_tasks_seen": len(tasks),
        "accepted": len(allowlist),
        "rejected": len(rejected),
        "counts": dict(sorted(counters.items())),
        "scenarios_seen": len({scenario for scenario, _, _ in tasks}),
        "accepted_scenarios": len({row["scenario"] for row in allowlist}),
        "by_scenario": {k: dict(v) for k, v in sorted(by_scenario.items())},
        "known_bad_tasks": [
            {"scenario": scenario, "task_id": task_id}
            for scenario, task_id in sorted(KNOWN_BAD_TASKS)
        ],
    }
    write_jsonl(out_dir / "allowlist.jsonl", allowlist)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    write_json(out_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
