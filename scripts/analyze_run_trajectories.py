#!/usr/bin/env python3
"""Analyze AWM run trajectories for runtime-tool and task-operability filtering."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ERROR_PATTERNS = {
    "http_500": re.compile(r"\b(500|internal server error)\b", re.I),
    "timeout": re.compile(r"\b(timeout|timed out)\b", re.I),
    "validation": re.compile(r"(input validation error|required property|validationerror|invalid)", re.I),
    "route_or_schema": re.compile(r"(not found|404|no route|unknown tool|unexpected keyword|missing .*argument)", re.I),
}
EMPTY_PAT = re.compile(r"\b(no .*found|not found|empty|does not exist|not available|unavailable)\b", re.I)
MUTATING_TOOL_PAT = re.compile(
    r"(^|_)(create|update|delete|cancel|book|reserve|add|remove|assign|mark|set|submit|approve|reject|complete|close)_",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--allowlist-jsonl", type=Path,
                        help="Only analyze scenario/task_id pairs from this JSONL file.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", default="code")
    parser.add_argument("--error-rate-threshold", type=float, default=0.30)
    parser.add_argument("--min-repeated-error", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true",
                        help="Reject tasks that never completed across the supplied runs.")
    parser.add_argument("--strict-all-hard-reject", action="store_true",
                        help="Reject only when every supplied run has a hard runtime/task issue.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def task_key(scenario: str, task_id: int) -> str:
    return f"{scenario}/task_{task_id}"


def load_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            allowed.add(task_key(str(row["scenario"]), int(row["task_id"])))
    return allowed


def iter_task_dirs(run_root: Path) -> list[Path]:
    return sorted(path.parent for path in run_root.glob("*/task_*/trajectory.json"))


def json_shape(value: Any) -> tuple[bool, bool]:
    """Return (is_json_like, has_non_empty_payload)."""
    if isinstance(value, list):
        return True, bool(value)
    if not isinstance(value, dict):
        return False, False
    if not value:
        return True, False
    saw_container = False
    for child in value.values():
        if isinstance(child, list):
            saw_container = True
            if child:
                return True, True
        elif isinstance(child, dict):
            saw_container = True
            child_is_json, child_has_payload = json_shape(child)
            if child_is_json and child_has_payload:
                return True, True
        elif child not in (None, "", []):
            return True, True
    return True, not saw_container


def response_payload_flags(text: str) -> dict[str, bool]:
    stripped = text.strip()
    if not stripped or stripped.startswith("Error:"):
        return {"json_like": False, "non_empty_payload": False, "empty_signal": bool(EMPTY_PAT.search(stripped))}
    try:
        parsed = json.loads(stripped)
    except Exception:
        return {
            "json_like": False,
            "non_empty_payload": bool(stripped),
            "empty_signal": bool(EMPTY_PAT.search(stripped)),
        }
    json_like, non_empty = json_shape(parsed)
    return {
        "json_like": json_like,
        "non_empty_payload": non_empty,
        "empty_signal": json_like and not non_empty,
    }


def classify_error(text: str) -> str | None:
    if not text.startswith("Error:"):
        return None
    for label, pattern in ERROR_PATTERNS.items():
        if pattern.search(text):
            return label
    return "tool_error"


def extract_call_tool_name(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return ""
    if not isinstance(arguments, dict):
        return ""
    name = str(arguments.get("tool_name") or "")
    if name.startswith("mcp_tool_"):
        name = name[len("mcp_tool_"):]
    return name


def has_malformed_inner_arguments(arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return False
    inner = arguments.get("arguments")
    if not isinstance(inner, str):
        return False
    # The common 7B failure is an unescaped JSON string split into top-level keys.
    return inner.strip().startswith("{") and len(arguments) > 2


def analyze_one(task_dir: Path, run_root: Path, mode: str) -> dict[str, Any]:
    trajectory_path = task_dir / "trajectory.json"
    verify_path = task_dir / f"verify.{mode}.json"
    trajectory = load_json(trajectory_path)
    verify = load_json(verify_path)
    rel = str(task_dir.relative_to(run_root))

    if trajectory is None:
        return {
            "run": run_root.name,
            "run_dir": rel,
            "scenario": task_dir.parent.name,
            "task_id": int(task_dir.name.removeprefix("task_")),
            "reward_type": verify.get("reward_type") if verify else None,
            "analysis": {"missing_or_bad_trajectory": True},
        }

    scenario = str(trajectory.get("scenario") or task_dir.parent.name)
    task_id = int(trajectory.get("task_id", task_dir.name.removeprefix("task_")))
    steps = trajectory.get("trajectory") or []
    tool_calls = 0
    call_tools = 0
    list_tools = 0
    final_seen = False
    mutating_calls = 0
    non_empty_tool_responses = 0
    empty_tool_responses = 0
    error_counts: Counter[str] = Counter()
    tool_error_counts: Counter[str] = Counter()
    repeated_error_signatures: Counter[str] = Counter()
    evidence: list[dict[str, Any]] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        final_seen = final_seen or bool(step.get("is_final"))
        calls = step.get("tool_calls") or []
        response = step.get("tool_response") or {}
        response_text = str(response.get("content") or "")
        error_label = classify_error(response_text)
        flags = response_payload_flags(response_text)

        for call in calls:
            if not isinstance(call, dict):
                continue
            tool_calls += 1
            name = str(call.get("name") or "")
            if name == "list_tools":
                list_tools += 1
            elif name == "call_tool":
                call_tools += 1
                tool_name = extract_call_tool_name(call.get("arguments"))
                if MUTATING_TOOL_PAT.search(tool_name):
                    mutating_calls += 1
                if error_label and error_label == "validation" and has_malformed_inner_arguments(call.get("arguments")):
                    error_label = "agent_argument_parse_error"

                if error_label:
                    error_counts[error_label] += 1
                    tool_error_counts[tool_name or "<unknown>"] += 1
                    signature = f"{tool_name or '<unknown>'}:{error_label}:{response_text[:160]}"
                    repeated_error_signatures[signature] += 1
                    evidence.append({
                        "iteration": step.get("iteration"),
                        "tool": tool_name,
                        "kind": error_label,
                        "response": response_text[:500],
                    })
                elif flags["non_empty_payload"]:
                    non_empty_tool_responses += 1
                elif flags["empty_signal"]:
                    empty_tool_responses += 1
                    evidence.append({
                        "iteration": step.get("iteration"),
                        "tool": tool_name,
                        "kind": "empty_or_not_found",
                        "response": response_text[:500],
                    })

    return {
        "run": run_root.name,
        "run_dir": rel,
        "scenario": scenario,
        "task_id": task_id,
        "task": str(trajectory.get("task") or ""),
        "reward_type": verify.get("reward_type") if verify else None,
        "verify_execution_status": ((verify.get("verify_result") or {}).get("execution_status") if verify else None),
        "analysis": {
            "missing_or_bad_trajectory": False,
            "tool_calls": tool_calls,
            "call_tools": call_tools,
            "list_tools": list_tools,
            "final_seen": final_seen,
            "error_counts": dict(error_counts),
            "tool_error_counts": dict(tool_error_counts.most_common(10)),
            "repeated_error_signatures": dict(repeated_error_signatures.most_common(10)),
            "mutating_calls": mutating_calls,
            "non_empty_tool_responses": non_empty_tool_responses,
            "empty_tool_responses": empty_tool_responses,
            "evidence": evidence[:12],
        },
    }


def decide_task(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, list[str]]:
    reasons: list[str] = []
    completed = any(row.get("reward_type") == "complete" for row in rows)
    analyzed_runs = len(rows)
    total_call_tools = 0
    total_errors = 0
    total_agent_argument_errors = 0
    error_counts: Counter[str] = Counter()
    repeated_signatures: Counter[str] = Counter()
    empty_runs = 0
    non_empty_runs = 0
    mutating_runs = 0

    for row in rows:
        a = row.get("analysis") or {}
        if a.get("missing_or_bad_trajectory"):
            reasons.append("missing_or_bad_trajectory")
            continue
        total_call_tools += int(a.get("call_tools") or 0)
        counts = Counter(a.get("error_counts") or {})
        error_counts.update(counts)
        total_agent_argument_errors += counts.get("agent_argument_parse_error", 0)
        total_errors += sum(
            value for label, value in counts.items()
            if label != "agent_argument_parse_error"
        )
        repeated_signatures.update(a.get("repeated_error_signatures") or {})
        if int(a.get("empty_tool_responses") or 0) > 0:
            empty_runs += 1
        if int(a.get("non_empty_tool_responses") or 0) > 0:
            non_empty_runs += 1
        if int(a.get("mutating_calls") or 0) > 0:
            mutating_runs += 1

    error_rate = (total_errors / total_call_tools) if total_call_tools else 0.0
    if total_call_tools == 0:
        reasons.append("no_call_tool_path")
    if total_agent_argument_errors:
        reasons.append(f"agent_argument_parse_error:{total_agent_argument_errors}")
    if error_rate >= args.error_rate_threshold and total_errors:
        reasons.append(f"high_tool_error_rate:{error_rate:.2f}")
    if error_counts.get("http_500", 0):
        reasons.append(f"http_500:{error_counts['http_500']}")
    if error_counts.get("timeout", 0):
        reasons.append(f"timeout:{error_counts['timeout']}")
    repeated_bad = [
        sig for sig, count in repeated_signatures.items()
        if count >= args.min_repeated_error and any(label in sig for label in (":validation:", ":route_or_schema:"))
    ]
    if repeated_bad:
        reasons.append(f"repeated_schema_or_route_error:{len(repeated_bad)}")
    if empty_runs == analyzed_runs and non_empty_runs == 0:
        reasons.append("only_empty_or_not_found_tool_responses")
    if not completed and mutating_runs == 0 and total_call_tools > 0:
        reasons.append("no_successful_mutating_path_observed")
    if args.require_complete and not completed:
        reasons.append("never_completed")

    hard_reasons = [
        reason for reason in reasons
        if reason.startswith(("high_tool_error_rate", "http_500", "timeout", "repeated_schema_or_route_error"))
        or reason in {"missing_or_bad_trajectory", "no_call_tool_path", "only_empty_or_not_found_tool_responses", "never_completed"}
    ]
    if hard_reasons:
        return "reject", reasons
    if completed:
        return "keep", reasons
    return "review", reasons


def hard_reasons_for_run(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if row.get("reward_type") == "complete":
        return reasons

    a = row.get("analysis") or {}
    if a.get("missing_or_bad_trajectory"):
        return ["missing_or_bad_trajectory"]

    call_tools = int(a.get("call_tools") or 0)
    counts = Counter(a.get("error_counts") or {})
    non_agent_errors = sum(
        value for label, value in counts.items()
        if label != "agent_argument_parse_error"
    )
    error_rate = (non_agent_errors / call_tools) if call_tools else 0.0

    if call_tools == 0:
        reasons.append("no_call_tool_path")
    if error_rate >= args.error_rate_threshold and non_agent_errors:
        reasons.append(f"high_tool_error_rate:{error_rate:.2f}")
    if counts.get("http_500", 0):
        reasons.append(f"http_500:{counts['http_500']}")
    if counts.get("timeout", 0):
        reasons.append(f"timeout:{counts['timeout']}")

    repeated = Counter(a.get("repeated_error_signatures") or {})
    repeated_bad = [
        sig for sig, count in repeated.items()
        if count >= args.min_repeated_error and any(label in sig for label in (":validation:", ":route_or_schema:"))
    ]
    if repeated_bad:
        reasons.append(f"repeated_schema_or_route_error:{len(repeated_bad)}")

    if call_tools > 0 and int(a.get("empty_tool_responses") or 0) > 0 and int(a.get("non_empty_tool_responses") or 0) == 0:
        reasons.append("only_empty_or_not_found_tool_responses")
    if args.require_complete:
        reasons.append("never_completed")
    return reasons


def decide_task_strict(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    expected_runs: int,
) -> tuple[str, list[str]]:
    if any(row.get("reward_type") == "complete" for row in rows):
        return "keep", ["complete_in_at_least_one_run"]

    if len(rows) < expected_runs:
        return "review", [f"missing_run_records:{expected_runs - len(rows)}"]

    per_run_hard = [hard_reasons_for_run(row, args) for row in rows]
    if all(reasons for reasons in per_run_hard):
        flat = [reason for reasons in per_run_hard for reason in reasons]
        counts = Counter(reason.split(":", 1)[0] for reason in flat)
        return "reject", [f"{name}:{count}" for name, count in sorted(counts.items())]

    soft_reasons: list[str] = []
    for row, hard in zip(rows, per_run_hard):
        if hard:
            soft_reasons.append(f"{row.get('run')}:hard")
        else:
            soft_reasons.append(f"{row.get('run')}:not_hard")
    return "review", soft_reasons


def main() -> int:
    args = parse_args()
    allowed = load_allowlist(args.allowlist_jsonl)
    per_run_rows: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run_root in args.run_roots:
        for task_dir in iter_task_dirs(run_root):
            row = analyze_one(task_dir, run_root, args.mode)
            key = task_key(str(row["scenario"]), int(row["task_id"]))
            if allowed is not None and key not in allowed:
                continue
            per_run_rows.append(row)
            by_task[key].append(row)

    task_rows: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {"keep": [], "review": [], "reject": []}
    for key, rows in sorted(by_task.items()):
        if args.strict_all_hard_reject:
            decision, reasons = decide_task_strict(rows, args, expected_runs=len(args.run_roots))
        else:
            decision, reasons = decide_task(rows, args)
        first = rows[0]
        merged = {
            "scenario": first["scenario"],
            "task_id": first["task_id"],
            "task": first.get("task", ""),
            "decision": decision,
            "reasons": reasons,
            "runs": rows,
        }
        task_rows.append(merged)
        buckets[decision].append({
            "scenario": first["scenario"],
            "task_id": first["task_id"],
            "task": first.get("task", ""),
            "reasons": reasons,
        })

    stats = {
        "run_roots": [str(path) for path in args.run_roots],
        "allowlist_jsonl": str(args.allowlist_jsonl) if args.allowlist_jsonl else None,
        "tasks_analyzed": len(task_rows),
        "run_task_records": len(per_run_rows),
        "decisions": {name: len(rows) for name, rows in buckets.items()},
        "reason_counts": dict(Counter(reason.split(":", 1)[0] for row in task_rows for reason in row["reasons"])),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "task_report.jsonl", task_rows)
    write_jsonl(args.out_dir / "per_run_report.jsonl", per_run_rows)
    write_jsonl(args.out_dir / "allowlist.jsonl", buckets["keep"] + buckets["review"])
    write_jsonl(args.out_dir / "keep.jsonl", buckets["keep"])
    write_jsonl(args.out_dir / "review.jsonl", buckets["review"])
    write_jsonl(args.out_dir / "rejected.jsonl", buckets["reject"])
    write_json(args.out_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
