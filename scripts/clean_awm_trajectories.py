#!/usr/bin/env python3
"""Clean AWM trajectory.json files into compact traces for skill distillation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ERROR_PATTERNS = (
    "traceback",
    "timeout",
    "500 internal server error",
    "server error",
    "internal server error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean AWM run trajectories into skill-distillation JSONL."
    )
    parser.add_argument("--input-root", required=True, help="AWM run output root.")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned outputs.")
    parser.add_argument("--max-items-per-list", type=int, default=10)
    parser.add_argument("--max-string-chars", type=int, default=300)
    parser.add_argument("--max-json-chars", type=int, default=2000)
    parser.add_argument("--max-text-chars", type=int, default=800)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sorted trajectory files.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Keep only trajectories whose verify.*.json has reward_type=complete.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars >= 0 and len(text) > max_chars:
        return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]", True
    return text, False


def maybe_json_loads(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, True
    try:
        return json.loads(value), True
    except json.JSONDecodeError:
        return value, False


def compress_json_value(
    value: Any,
    *,
    max_items_per_list: int,
    max_string_chars: int,
    truncations: list[dict[str, Any]],
    path: str = "$",
) -> Any:
    if isinstance(value, dict):
        return {
            key: compress_json_value(
                subvalue,
                max_items_per_list=max_items_per_list,
                max_string_chars=max_string_chars,
                truncations=truncations,
                path=f"{path}.{key}",
            )
            for key, subvalue in value.items()
        }
    if isinstance(value, list):
        kept = value[:max_items_per_list]
        omitted = max(0, len(value) - len(kept))
        if omitted:
            truncations.append({"path": path, "type": "list", "omitted_count": omitted})
        return [
            compress_json_value(
                item,
                max_items_per_list=max_items_per_list,
                max_string_chars=max_string_chars,
                truncations=truncations,
                path=f"{path}[{idx}]",
            )
            for idx, item in enumerate(kept)
        ]
    if isinstance(value, str):
        truncated, did_truncate = truncate_text(value, max_string_chars)
        if did_truncate:
            truncations.append(
                {
                    "path": path,
                    "type": "string",
                    "original_chars": len(value),
                    "kept_chars": max_string_chars,
                }
            )
        return truncated
    return value


def summarize_observation(
    content: Any,
    *,
    max_items_per_list: int,
    max_string_chars: int,
    max_json_chars: int,
    max_text_chars: int,
) -> tuple[dict[str, Any], bool]:
    if content is None:
        return {"type": "none", "content": None, "truncated": False}, False

    if isinstance(content, (dict, list)):
        parsed = content
        is_json = True
        parse_ok = True
        raw_text = json.dumps(content, ensure_ascii=False)
    elif isinstance(content, str):
        raw_text = content
        parsed, parse_ok = maybe_json_loads(content)
        is_json = parse_ok and isinstance(parsed, (dict, list))
    else:
        raw_text = str(content)
        parsed = raw_text
        is_json = False
        parse_ok = False

    obs_error = any(pattern in raw_text.lower() for pattern in ERROR_PATTERNS)

    if is_json:
        truncations: list[dict[str, Any]] = []
        compressed = compress_json_value(
            parsed,
            max_items_per_list=max_items_per_list,
            max_string_chars=max_string_chars,
            truncations=truncations,
        )
        compact = json.dumps(compressed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) > max_json_chars:
            truncated_text, _ = truncate_text(compact, max_json_chars)
            return (
                {
                    "type": "json",
                    "content": truncated_text,
                    "truncated": True,
                    "truncations": truncations
                    + [
                        {
                            "path": "$",
                            "type": "json_chars",
                            "original_chars": len(compact),
                            "kept_chars": max_json_chars,
                        }
                    ],
                },
                obs_error,
            )
        return (
            {
                "type": "json",
                "content": compressed,
                "truncated": bool(truncations),
                "truncations": truncations,
            },
            obs_error,
        )

    text = compact_whitespace(raw_text)
    truncated, did_truncate = truncate_text(text, max_text_chars)
    return (
        {
            "type": "text",
            "content": truncated,
            "truncated": did_truncate,
            "parse_ok": parse_ok,
        },
        obs_error,
    )


def extract_tool_call_from_text(content: str) -> dict[str, Any] | None:
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not match:
        return None
    payload = match.group(1)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def normalize_call(raw_call: dict[str, Any]) -> dict[str, Any] | None:
    name = raw_call.get("name")
    arguments = raw_call.get("arguments")
    if name == "list_tools":
        return None
    if name != "call_tool":
        return {
            "tool": name,
            "args": arguments,
            "args_parse_ok": not isinstance(arguments, str),
        }

    if not isinstance(arguments, dict):
        return {
            "tool": None,
            "args": arguments,
            "args_parse_ok": False,
        }

    tool_name = arguments.get("tool_name")
    tool_args = arguments.get("arguments")
    parsed_args, parse_ok = maybe_json_loads(tool_args)
    return {
        "tool": tool_name,
        "args": parsed_args,
        "args_parse_ok": parse_ok,
    }


def get_entry_tool_calls(entry: dict[str, Any]) -> list[dict[str, Any]]:
    calls = entry.get("tool_calls")
    if isinstance(calls, list) and calls:
        return [call for call in calls if isinstance(call, dict)]
    content = entry.get("content", "")
    if isinstance(content, str):
        parsed = extract_tool_call_from_text(content)
        if parsed:
            return [parsed]
    return []


def get_tool_response_content(
    trajectory: list[dict[str, Any]], entry_index: int, entry: dict[str, Any]
) -> Any:
    response = entry.get("tool_response")
    if isinstance(response, dict) and "content" in response:
        return response.get("content")
    for next_entry in trajectory[entry_index + 1 :]:
        if next_entry.get("role") == "tool":
            return next_entry.get("content")
        if next_entry.get("role") == "assistant" and get_entry_tool_calls(next_entry):
            break
    return None


def extract_final_answer(trajectory: list[dict[str, Any]]) -> str | None:
    for entry in reversed(trajectory):
        if entry.get("role") != "assistant":
            continue
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if get_entry_tool_calls(entry):
            continue
        return content.strip()
    return None


def load_verify(task_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for name in ("verify.code.json", "verify.sql.json"):
        path = task_dir / name
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            return None, path
        verify_result = data.get("verify_result")
        execution_status = None
        if isinstance(verify_result, dict):
            execution_status = verify_result.get("execution_status")
        return (
            {
                "mode": data.get("mode"),
                "reward_type": data.get("reward_type"),
                "execution_status": execution_status,
            },
            path,
        )
    return None, None


def find_verify_path(task_dir: Path) -> Path | None:
    for name in ("verify.code.json", "verify.sql.json"):
        path = task_dir / name
        if path.exists():
            return path
    return None


def build_source(input_root: Path, trajectory_path: Path, verify_path: Path | None) -> dict[str, Any]:
    task_dir = trajectory_path.parent
    return {
        "run_name": input_root.name,
        "input_root": str(input_root.resolve()),
        "task_dir": str(task_dir.resolve()),
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_relpath": str(trajectory_path.relative_to(input_root)),
        "verify_path": str(verify_path.resolve()) if verify_path else None,
    }


def reject_row(
    input_root: Path,
    trajectory_path: Path,
    reason: str,
    details: str,
    verify_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "source": build_source(input_root, trajectory_path, verify_path),
        "reason": reason,
        "details": details,
    }


def clean_one(
    trajectory_path: Path,
    *,
    input_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int, int]:
    raw_chars = 0
    try:
        raw_text = trajectory_path.read_text(encoding="utf-8")
        raw_chars = len(raw_text)
        data = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        return (
            None,
            reject_row(
                input_root,
                trajectory_path,
                "parse_error",
                str(exc),
                find_verify_path(trajectory_path.parent),
            ),
            raw_chars,
            0,
        )

    if not isinstance(data, dict):
        return (
            None,
            reject_row(
                input_root,
                trajectory_path,
                "parse_error",
                "Top-level JSON is not an object",
                find_verify_path(trajectory_path.parent),
            ),
            raw_chars,
            0,
        )

    task_dir = trajectory_path.parent
    verify, verify_path = load_verify(task_dir)
    if args.require_complete and (not verify or verify.get("reward_type") != "complete"):
        return (
            None,
            reject_row(
                input_root, trajectory_path, "not_complete", f"verify={verify}", verify_path
            ),
            raw_chars,
            0,
        )

    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list):
        return (
            None,
            reject_row(
                input_root, trajectory_path, "parse_error", "Missing trajectory list", verify_path
            ),
            raw_chars,
            0,
        )

    tool_sequence: list[dict[str, Any]] = []
    tool_parse_failures = 0
    has_obs_error = False
    for idx, entry in enumerate(trajectory):
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "assistant":
            continue
        for raw_call in get_entry_tool_calls(entry):
            normalized = normalize_call(raw_call)
            if normalized is None:
                continue
            if not normalized.get("tool"):
                tool_parse_failures += 1
                continue
            response_content = get_tool_response_content(trajectory, idx, entry)
            obs, obs_error = summarize_observation(
                response_content,
                max_items_per_list=args.max_items_per_list,
                max_string_chars=args.max_string_chars,
                max_json_chars=args.max_json_chars,
                max_text_chars=args.max_text_chars,
            )
            has_obs_error = has_obs_error or obs_error
            tool_sequence.append(
                {
                    "step": len(tool_sequence) + 1,
                    "tool": normalized["tool"],
                    "args": normalized.get("args"),
                    "args_parse_ok": normalized.get("args_parse_ok", False),
                    "obs": obs,
                    "obs_error": obs_error,
                }
            )

    if not tool_sequence:
        reason = "all_tool_calls_unparseable" if tool_parse_failures else "no_call_tool"
        return (
            None,
            reject_row(
                input_root,
                trajectory_path,
                reason,
                "No non-list_tools call_tool entries found",
                verify_path,
            ),
            raw_chars,
            0,
        )
    if has_obs_error:
        return (
            None,
            reject_row(
                input_root,
                trajectory_path,
                "tool_error",
                "Observation contains server/environment error text",
                verify_path,
            ),
            raw_chars,
            0,
        )

    final_answer = extract_final_answer(trajectory)
    if not final_answer:
        return (
            None,
            reject_row(
                input_root,
                trajectory_path,
                "no_final_answer",
                "No final assistant answer found",
                verify_path,
            ),
            raw_chars,
            0,
        )

    source = build_source(input_root, trajectory_path, verify_path)
    scenario = data.get("scenario")
    task_id = data.get("task_id")
    traj_id = f"{scenario}/task_{task_id}"
    skill_trace = build_skill_trace(data.get("task", ""), tool_sequence, final_answer)
    row = {
        "traj_id": traj_id,
        "source": source,
        "scenario": scenario,
        "task_id": task_id,
        "task": data.get("task", ""),
        "model": data.get("model"),
        "verify": verify,
        "num_tool_calls": len(tool_sequence),
        "tools_used": [step["tool"] for step in tool_sequence],
        "tool_sequence": tool_sequence,
        "final_answer": final_answer,
        "skill_trace": skill_trace,
    }
    clean_chars = len(json.dumps(row, ensure_ascii=False))
    return row, None, raw_chars, clean_chars


def build_skill_trace(task: str, tool_sequence: list[dict[str, Any]], final_answer: str) -> str:
    lines = [f"Task: {task}", "Steps:"]
    for step in tool_sequence:
        obs = step.get("obs", {})
        obs_content = obs.get("content") if isinstance(obs, dict) else obs
        obs_text = json.dumps(obs_content, ensure_ascii=False, separators=(",", ":"))
        lines.append(
            f"{step['step']}. Call {step['tool']} with "
            f"{json.dumps(step.get('args'), ensure_ascii=False, separators=(',', ':'))}; "
            f"observe {obs_text}."
        )
    lines.append(f"Final: {final_answer}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(input_root.glob("*/task_*/trajectory.json"))
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    paths = all_paths[: args.limit] if args.limit is not None else all_paths
    cleaned: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    raw_chars_total = 0
    clean_chars_total = 0
    tool_counts: Counter[str] = Counter()

    for path in paths:
        row, reject, raw_chars, clean_chars = clean_one(path, input_root=input_root, args=args)
        raw_chars_total += raw_chars
        clean_chars_total += clean_chars
        if row is not None:
            cleaned.append(row)
            tool_counts.update(row["tools_used"])
        elif reject is not None:
            rejected.append(reject)

    write_jsonl(output_dir / "clean_trajs.jsonl", cleaned)
    write_jsonl(output_dir / "rejected.jsonl", rejected)

    reject_reasons = Counter(row["reason"] for row in rejected)
    avg_raw = raw_chars_total / len(paths) if paths else 0.0
    avg_clean = clean_chars_total / len(cleaned) if cleaned else 0.0
    stats = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "cleaning_args": {
            "max_items_per_list": args.max_items_per_list,
            "max_string_chars": args.max_string_chars,
            "max_json_chars": args.max_json_chars,
            "max_text_chars": args.max_text_chars,
            "require_complete": args.require_complete,
            "limit": args.limit,
        },
        "matched_files": len(all_paths),
        "raw_files": len(paths),
        "kept": len(cleaned),
        "rejected": len(rejected),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "avg_tool_calls": (
            sum(row["num_tool_calls"] for row in cleaned) / len(cleaned) if cleaned else 0.0
        ),
        "avg_raw_chars": avg_raw,
        "avg_clean_chars": avg_clean,
        "avg_compression_ratio": avg_clean / avg_raw if avg_raw else 0.0,
        "top_tools": tool_counts.most_common(50),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(
        f"Cleaned {len(cleaned)} / {len(paths)} trajectories; "
        f"rejected {len(rejected)}. Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
