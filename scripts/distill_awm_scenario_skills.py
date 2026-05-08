#!/usr/bin/env python3
"""Distill one simple skill.md per AWM scenario from cleaned trajectories."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You distill reusable agent skills from successful AWM tool-use trajectories.
Write concise operational guidance for another agent. Generalize IDs and literal values into lookup rules.
Do not mention training, datasets, trajectories, or that you are reading examples.
Do not invent tools or fields that are not supported by the examples.
Return the final skill directly. Do not include analysis, reasoning, or a thinking process.
Output only Markdown in the requested template."""


USER_TEMPLATE = """Distill one scenario-level skill for the AWM scenario `{scenario}`.

Use successful examples as the main evidence. Use unsuccessful examples only to identify avoidable mistakes.

Required Markdown template:

# {scenario}

## Goal
One short paragraph describing what this scenario is about and what tasks usually require.

## Tool Strategy
1. First action.
2. Middle actions.
3. Final confirmation.

## Success Checks
- Check 1.
- Check 2.
- Check 3.

## Common Failures
- Failure 1 and how to avoid it.
- Failure 2 and how to avoid it.

## Evidence
- Positive task ids: ...
- Negative task ids: ...
- Confidence: high | medium | low

Rules:
- Keep the skill under {max_words} words.
- Prefer concrete tool names when they are stable across examples.
- Never hard-code example-specific IDs, names, addresses, emails, dates, or prices.
- If there are no positive examples, write a low-confidence skill focused on safe exploration and failure avoidance.
- The final line in Evidence must contain exactly one confidence value.

Positive examples:
{positive_examples}

Negative or auxiliary examples:
{negative_examples}
"""


@dataclass(frozen=True)
class Row:
    scenario: str
    task_id: int
    task: str
    reward_type: str
    num_tool_calls: int
    tools_used: list[str]
    skill_trace: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-complete",
        type=Path,
        required=True,
        help="Path to clean_complete/clean_trajs.jsonl.",
    )
    parser.add_argument(
        "--clean-all",
        type=Path,
        required=True,
        help="Path to clean_all/clean_trajs.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-397b-a17b")
    parser.add_argument("--scenario", action="append", help="Only distill selected scenario(s).")
    parser.add_argument("--limit-scenarios", type=int, help="Only process first N sorted scenarios.")
    parser.add_argument("--max-positive", type=int, default=4)
    parser.add_argument("--max-negative", type=int, default=2)
    parser.add_argument("--max-trace-chars", type=int, default=3500)
    parser.add_argument("--max-words", type=int, default=650)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-reasoning-fallback", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep after each API call.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            verify = item.get("verify") if isinstance(item.get("verify"), dict) else {}
            rows.append(
                Row(
                    scenario=str(item.get("scenario", "")),
                    task_id=int(item.get("task_id", -1)),
                    task=str(item.get("task", "")),
                    reward_type=str(verify.get("reward_type", "unknown")),
                    num_tool_calls=int(item.get("num_tool_calls", 0)),
                    tools_used=[str(x) for x in item.get("tools_used", [])],
                    skill_trace=str(item.get("skill_trace", "")),
                )
            )
    return rows


class EmptyCompletionError(RuntimeError):
    def __init__(self, message: str, response: dict[str, Any]) -> None:
        super().__init__(message)
        self.response = response


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def format_examples(rows: list[Row], max_trace_chars: int) -> str:
    if not rows:
        return "(none)"
    blocks: list[str] = []
    for row in rows:
        tools = ", ".join(row.tools_used)
        blocks.append(
            "\n".join(
                [
                    f"Task id: {row.task_id}",
                    f"Verdict: {row.reward_type}",
                    f"Task: {row.task}",
                    f"Tools: {tools}",
                    "Trace:",
                    truncate(row.skill_trace, max_trace_chars),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def choose_examples(
    positives: list[Row],
    auxiliaries: list[Row],
    *,
    max_positive: int,
    max_negative: int,
) -> tuple[list[Row], list[Row]]:
    # Prefer shorter successful traces: they are usually cleaner and easier to generalize.
    pos = sorted(positives, key=lambda r: (r.num_tool_calls, r.task_id))[:max_positive]
    pos_ids = {r.task_id for r in pos}
    neg_pool = [r for r in auxiliaries if r.task_id not in pos_ids]
    # Prefer failed/other examples first, then any remaining auxiliary traces.
    neg_pool = sorted(
        neg_pool,
        key=lambda r: (r.reward_type == "complete", r.num_tool_calls, r.task_id),
    )
    neg = neg_pool[:max_negative]
    return pos, neg


def call_chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    allow_reasoning_fallback: bool,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    try:
        choice = obj["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected chat completion response: {obj}") from exc

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip(), obj

    reasoning_content = message.get("reasoning_content")
    if allow_reasoning_fallback and isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip(), obj

    finish_reason = choice.get("finish_reason")
    raise EmptyCompletionError(
        "empty chat completion content "
        f"(finish_reason={finish_reason!r}, message_keys={sorted(message.keys())})",
        obj,
    )


def safe_name(scenario: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", scenario.strip())
    return name.strip("_") or "unknown_scenario"


def build_prompt(
    scenario: str,
    positives: list[Row],
    negatives: list[Row],
    *,
    max_trace_chars: int,
    max_words: int,
) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        scenario=scenario,
        max_words=max_words,
        positive_examples=format_examples(positives, max_trace_chars),
        negative_examples=format_examples(negatives, max_trace_chars),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def distill_one(
    scenario: str,
    positives_by_scenario: dict[str, list[Row]],
    all_by_scenario: dict[str, list[Row]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    scenario_dir = args.output_dir / "by_scenario" / safe_name(scenario)
    skill_path = scenario_dir / "skill.md"
    prompt_path = scenario_dir / "prompt.json"
    meta_path = scenario_dir / "meta.json"
    response_path = scenario_dir / "response.json"
    error_path = scenario_dir / "error.json"
    if skill_path.exists() and not args.overwrite:
        return {"scenario": scenario, "status": "skipped", "skill_path": str(skill_path)}

    positives, negatives = choose_examples(
        positives_by_scenario.get(scenario, []),
        all_by_scenario.get(scenario, []),
        max_positive=args.max_positive,
        max_negative=args.max_negative,
    )
    messages = build_prompt(
        scenario,
        positives,
        negatives,
        max_trace_chars=args.max_trace_chars,
        max_words=args.max_words,
    )
    scenario_dir.mkdir(parents=True, exist_ok=True)
    with prompt_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    meta = {
        "scenario": scenario,
        "num_positive_available": len(positives_by_scenario.get(scenario, [])),
        "num_aux_available": len(all_by_scenario.get(scenario, [])),
        "positive_task_ids": [r.task_id for r in positives],
        "negative_task_ids": [r.task_id for r in negatives],
        "model": args.model,
    }
    if args.dry_run:
        content = "# DRY RUN\n\nPrompt written to prompt.json.\n"
    else:
        try:
            content, response = call_chat_completion(
                api_url=args.api_url,
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
                allow_reasoning_fallback=args.allow_reasoning_fallback,
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            if isinstance(response, dict):
                with response_path.open("w", encoding="utf-8") as f:
                    json.dump(response, f, ensure_ascii=False, indent=2, sort_keys=True)
            with error_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "scenario": scenario,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            raise
        with response_path.open("w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2, sort_keys=True)
        if args.sleep:
            time.sleep(args.sleep)

    with skill_path.open("w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    return {"scenario": scenario, "status": "done", "skill_path": str(skill_path), **meta}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    complete_rows = load_rows(args.clean_complete)
    all_rows = load_rows(args.clean_all)
    positives_by_scenario: dict[str, list[Row]] = defaultdict(list)
    all_by_scenario: dict[str, list[Row]] = defaultdict(list)

    for row in complete_rows:
        positives_by_scenario[row.scenario].append(row)
    for row in all_rows:
        all_by_scenario[row.scenario].append(row)

    scenarios = sorted(set(all_by_scenario) | set(positives_by_scenario))
    if args.scenario:
        selected = set(args.scenario)
        scenarios = [s for s in scenarios if s in selected]
        missing = sorted(selected - set(scenarios))
        if missing:
            raise SystemExit(f"requested scenario(s) not found in cleaned data: {missing}")
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    manifest_path = args.output_dir / "skills_manifest.jsonl"
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(distill_one, scenario, positives_by_scenario, all_by_scenario, args)
            for scenario in scenarios
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']} {result['scenario']} -> {result.get('skill_path')}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda x: x["scenario"]):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    stats = {
        "output_dir": str(args.output_dir.resolve()),
        "scenarios_requested": len(scenarios),
        "done": sum(1 for r in results if r["status"] == "done"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "clean_complete_rows": len(complete_rows),
        "clean_complete_scenarios": len(positives_by_scenario),
        "clean_all_rows": len(all_rows),
        "clean_all_scenarios": len(all_by_scenario),
    }
    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
