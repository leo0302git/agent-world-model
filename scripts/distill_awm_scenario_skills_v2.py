#!/usr/bin/env python3
"""Distill verifier-sensitive AWM scenario skills from cleaned trajectories.

This v2 distiller is optimized for prompt injection during AWM scoring. It asks
the teacher model to produce compact planning hints, state checks, and guardrails
instead of broad step-by-step recipes. The target failure mode is strong agents
that already know how to use tools but miss exact database-state requirements.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BANNED_TEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdatasets?\b",
        r"\btrajector(?:y|ies)\b",
        r"\btraining\b",
        r"\bthinking process\b",
        r"\banalyze the request\b",
        r"\bdrafting the content\b",
    )
]

WILDCARD_TOOL_RE = re.compile(
    r"mcp_tool_[a-zA-Z0-9_]+\*|mcp_tool_[a-zA-Z0-9_]+_\b|\b(?:list|get|create|update|delete)_\*"
)

SYSTEM_PROMPT = """You distill AWM scenario skills for a strong tool-using agent.
The skill will be shown after the live list_tools response, so it must not replace live tool schemas.
Write verifier-sensitive guidance: exact state checks, tool-selection boundaries, and common failure avoidance.
Do not mention training, datasets, trajectories, examples, or that you are reading logs.
Do not include reasoning, analysis, or a thinking process.
Do not invent tools, IDs, field names, or business facts.
Output only Markdown in the required template."""


USER_TEMPLATE = """Distill a high-signal scenario skill for `{scenario}`.

The target agent already receives live tool names and schemas from list_tools. Your skill should therefore:
- steer planning without overriding live schemas;
- prevent near-miss database-state failures;
- say when a common workflow does not apply;
- avoid generic advice the agent already knows.

Required Markdown template:

# {scenario}

## Use When
- One or two bullets naming the task patterns this skill covers.

## Planning Hints
- 2-4 bullets describing high-level workflows only when supported by evidence.
- Prefer action patterns over rigid first/middle/final instructions.

## Tool Guardrails
- 2-5 bullets about exact tool-selection boundaries.
- Mention concrete tool names only if they appear in the evidence and are stable.
- Never use wildcard or placeholder tool names such as `mcp_tool_*`, `list_*`, or `create_*`.
- If a tool family is not stable, describe the operation without naming a tool.

## State Checks
- 3-6 bullets that must be true before the final answer.
- Focus on verifier-sensitive state: exact target entity, linked child rows, totals/balances/statuses, primary/default uniqueness, no duplicate action records, and required preserved fields.

## Avoid
- 2-5 bullets naming mistakes that cause wrong final database state.
- Include "do not blindly follow this skill if the live tool list or user task differs" if the evidence is thin.

## Evidence
- Positive task ids: ...
- Auxiliary/failure task ids: ...
- Confidence: high | medium | low

Rules:
- Keep the skill under {max_words} words.
- Do not say "call list_tools"; the agent already must do that.
- Do not tell the agent to ignore live tool schemas.
- Do not hard-code example-specific IDs, names, emails, addresses, dates, prices, titles, or product names.
- If there are zero positive examples, produce a low-confidence failure-boundary skill, not a fake recipe.
- The final line in Evidence must contain exactly one confidence value.

Positive evidence:
{positive_examples}

Auxiliary or failure evidence:
{negative_examples}

Optional paired eval evidence:
{eval_examples}
"""


@dataclass(frozen=True)
class CleanRow:
    scenario: str
    task_id: int
    task: str
    reward_type: str
    num_tool_calls: int
    tools_used: list[str]
    skill_trace: str


@dataclass(frozen=True)
class EvalRow:
    scenario: str
    task_id: int
    baseline_reward: str
    skill_reward: str
    task: str
    baseline_tools: list[str]
    skill_tools: list[str]
    baseline_iters: int | None
    skill_iters: int | None


class EmptyCompletionError(RuntimeError):
    def __init__(self, message: str, response: dict[str, Any]) -> None:
        super().__init__(message)
        self.response = response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-complete", type=Path, required=True)
    parser.add_argument("--clean-all", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-397b-a17b")
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--max-positive", type=int, default=5)
    parser.add_argument("--max-negative", type=int, default=4)
    parser.add_argument("--max-eval-examples", type=int, default=4)
    parser.add_argument("--max-trace-chars", type=int, default=4200)
    parser.add_argument("--max-words", type=int, default=520)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=1700)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-reasoning-fallback", action="store_true")
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-run",
        type=Path,
        help="Optional no-skill or baseline run root. Used with --skill-run to add win/loss evidence.",
    )
    parser.add_argument(
        "--skill-run",
        type=Path,
        help="Optional prior skill run root. Used with --baseline-run to add paired eval evidence.",
    )
    return parser.parse_args()


def load_clean_rows(path: Path) -> list[CleanRow]:
    rows: list[CleanRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            verify = item.get("verify") if isinstance(item.get("verify"), dict) else {}
            rows.append(
                CleanRow(
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


def normalize_scenario_name(scenario: str) -> str:
    s = scenario.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_").strip()


def safe_name(scenario: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", scenario.strip())
    return name.strip("_") or "unknown_scenario"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def extract_tool_sequence(trajectory: dict[str, Any]) -> list[str]:
    sequence: list[str] = []
    for step in trajectory.get("trajectory", []):
        if not isinstance(step, dict):
            continue
        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", ""))
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            if name == "call_tool":
                tool_name = str(args.get("tool_name", ""))
                if tool_name:
                    sequence.append(tool_name)
            elif name:
                sequence.append(name)
    return sequence


def load_run_rows(run_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for verify_path in run_root.glob("*/task_*/verify.code.json"):
        try:
            verify = load_json(verify_path)
        except Exception:
            continue
        scenario = str(verify.get("scenario", ""))
        task_id = int(verify.get("task_id", verify_path.parent.name.removeprefix("task_")))
        trajectory_path = verify_path.parent / "trajectory.json"
        trajectory: dict[str, Any] = {}
        if trajectory_path.exists():
            try:
                trajectory = load_json(trajectory_path)
            except Exception:
                trajectory = {}
        rows[(scenario, task_id)] = {
            "reward": str(verify.get("reward_type", "unknown")),
            "task": str(verify.get("task") or trajectory.get("task") or ""),
            "tools": extract_tool_sequence(trajectory),
            "iters": trajectory.get("total_iterations"),
        }
    return rows


def load_eval_rows(baseline_run: Path | None, skill_run: Path | None) -> dict[str, list[EvalRow]]:
    if not baseline_run or not skill_run:
        return {}
    baseline = load_run_rows(baseline_run)
    skill = load_run_rows(skill_run)
    by_scenario: dict[str, list[EvalRow]] = defaultdict(list)
    for key in sorted(set(baseline) & set(skill)):
        base = baseline[key]
        cur = skill[key]
        if base["reward"] == cur["reward"]:
            continue
        scenario, task_id = key
        by_scenario[scenario].append(
            EvalRow(
                scenario=scenario,
                task_id=task_id,
                baseline_reward=base["reward"],
                skill_reward=cur["reward"],
                task=str(cur["task"] or base["task"]),
                baseline_tools=[str(x) for x in base["tools"]],
                skill_tools=[str(x) for x in cur["tools"]],
                baseline_iters=base["iters"] if isinstance(base["iters"], int) else None,
                skill_iters=cur["iters"] if isinstance(cur["iters"], int) else None,
            )
        )
    return by_scenario


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def format_clean_examples(rows: list[CleanRow], max_trace_chars: int) -> str:
    if not rows:
        return "(none)"
    blocks: list[str] = []
    for row in rows:
        blocks.append(
            "\n".join(
                [
                    f"Task id: {row.task_id}",
                    f"Verdict: {row.reward_type}",
                    f"Task: {row.task}",
                    f"Tools used: {', '.join(row.tools_used)}",
                    "Clean trace:",
                    truncate(row.skill_trace, max_trace_chars),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def format_eval_examples(rows: list[EvalRow], max_examples: int) -> str:
    if not rows:
        return "(none)"
    wins = [r for r in rows if r.baseline_reward != "complete" and r.skill_reward == "complete"]
    losses = [r for r in rows if r.baseline_reward == "complete" and r.skill_reward != "complete"]
    selected = (wins[: max_examples // 2 + max_examples % 2] + losses[: max_examples // 2])[:max_examples]
    blocks: list[str] = []
    for row in selected:
        blocks.append(
            "\n".join(
                [
                    f"Task id: {row.task_id}",
                    f"Baseline reward: {row.baseline_reward}",
                    f"Prior skill reward: {row.skill_reward}",
                    f"Task: {row.task}",
                    f"Baseline tools: {', '.join(row.baseline_tools[:12])}",
                    f"Prior skill tools: {', '.join(row.skill_tools[:12])}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def choose_examples(
    positives: list[CleanRow],
    auxiliaries: list[CleanRow],
    *,
    max_positive: int,
    max_negative: int,
) -> tuple[list[CleanRow], list[CleanRow]]:
    # Prefer concise successes, but keep tool diversity so the skill does not
    # overfit to one task path.
    selected: list[CleanRow] = []
    covered_tools: set[str] = set()
    for row in sorted(positives, key=lambda r: (r.num_tool_calls, r.task_id)):
        if len(selected) >= max_positive:
            break
        tool_gain = len(set(row.tools_used) - covered_tools)
        if not selected or tool_gain > 0 or len(selected) < min(2, max_positive):
            selected.append(row)
            covered_tools.update(row.tools_used)

    for row in sorted(positives, key=lambda r: (r.task_id, r.num_tool_calls)):
        if len(selected) >= max_positive:
            break
        if row not in selected:
            selected.append(row)

    selected_ids = {r.task_id for r in selected}
    neg_pool = [r for r in auxiliaries if r.task_id not in selected_ids]
    neg_pool = sorted(
        neg_pool,
        key=lambda r: (
            r.reward_type == "complete",
            -len(set(r.tools_used) & covered_tools),
            r.num_tool_calls,
            r.task_id,
        ),
    )
    return selected, neg_pool[:max_negative]


def build_prompt(
    scenario: str,
    positives: list[CleanRow],
    negatives: list[CleanRow],
    eval_rows: list[EvalRow],
    *,
    max_trace_chars: int,
    max_words: int,
    max_eval_examples: int,
) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        scenario=scenario,
        max_words=max_words,
        positive_examples=format_clean_examples(positives, max_trace_chars),
        negative_examples=format_clean_examples(negatives, max_trace_chars),
        eval_examples=format_eval_examples(eval_rows, max_eval_examples),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


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
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        api_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
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

    reasoning = message.get("reasoning_content")
    if allow_reasoning_fallback and isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip(), obj

    raise EmptyCompletionError(
        "empty chat completion content "
        f"(finish_reason={choice.get('finish_reason')!r}, message_keys={sorted(message.keys())})",
        obj,
    )


def validate_skill(text: str, max_words: int) -> dict[str, Any]:
    required_sections = [
        "## Use When",
        "## Planning Hints",
        "## Tool Guardrails",
        "## State Checks",
        "## Avoid",
        "## Evidence",
    ]
    warnings: list[str] = []
    missing = [section for section in required_sections if section not in text]
    if missing:
        warnings.append("missing_sections:" + ",".join(missing))
    if len(text.split()) > max_words:
        warnings.append(f"too_long:{len(text.split())}>{max_words}")
    if WILDCARD_TOOL_RE.search(text):
        warnings.append("wildcard_or_placeholder_tool_name")
    banned_hits = [pat.pattern for pat in BANNED_TEXT_PATTERNS if pat.search(text)]
    if banned_hits:
        warnings.append("banned_terms:" + ",".join(banned_hits))
    if "```" in text:
        warnings.append("code_fence")
    confidence_hits = re.findall(r"Confidence:\s*(high|medium|low)\b", text, flags=re.IGNORECASE)
    if len(confidence_hits) != 1:
        warnings.append(f"confidence_count:{len(confidence_hits)}")
    return {
        "word_count": len(text.split()),
        "warnings": warnings,
        "ok": not warnings,
    }


def build_repair_prompt(skill: str, validation: dict[str, Any], max_words: int) -> list[dict[str, str]]:
    user = f"""Repair this Markdown skill so it passes the quality constraints.

Constraints:
- Keep the same scenario and evidence task ids.
- Keep it under {max_words} words.
- Include all required sections: Use When, Planning Hints, Tool Guardrails, State Checks, Avoid, Evidence.
- Remove wildcard or placeholder tool names.
- Remove mentions of datasets, trajectories, training, or thinking process.
- Output only the repaired Markdown skill.

Validation warnings:
{json.dumps(validation.get("warnings", []), ensure_ascii=False)}

Skill to repair:
{skill}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def distill_one(
    scenario: str,
    positives_by_scenario: dict[str, list[CleanRow]],
    all_by_scenario: dict[str, list[CleanRow]],
    eval_by_scenario: dict[str, list[EvalRow]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    scenario_dir = args.output_dir / "by_scenario" / safe_name(scenario)
    skill_path = scenario_dir / "skill.md"
    prompt_path = scenario_dir / "prompt.json"
    meta_path = scenario_dir / "meta.json"
    response_path = scenario_dir / "response.json"
    quality_path = scenario_dir / "quality.json"
    error_path = scenario_dir / "error.json"

    if skill_path.exists() and not args.overwrite:
        return {"scenario": scenario, "status": "skipped", "skill_path": str(skill_path)}

    positives, negatives = choose_examples(
        positives_by_scenario.get(scenario, []),
        all_by_scenario.get(scenario, []),
        max_positive=args.max_positive,
        max_negative=args.max_negative,
    )
    eval_rows = eval_by_scenario.get(scenario, [])
    messages = build_prompt(
        scenario,
        positives,
        negatives,
        eval_rows,
        max_trace_chars=args.max_trace_chars,
        max_words=args.max_words,
        max_eval_examples=args.max_eval_examples,
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    with prompt_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    meta = {
        "scenario": scenario,
        "num_positive_available": len(positives_by_scenario.get(scenario, [])),
        "num_aux_available": len(all_by_scenario.get(scenario, [])),
        "num_eval_available": len(eval_rows),
        "positive_task_ids": [r.task_id for r in positives],
        "auxiliary_task_ids": [r.task_id for r in negatives],
        "eval_task_ids": [r.task_id for r in eval_rows[: args.max_eval_examples]],
        "model": args.model,
        "template_version": "v2_verifier_sensitive",
    }

    if args.dry_run:
        content = "# DRY RUN\n\nPrompt written to prompt.json.\n"
        response: dict[str, Any] = {}
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
            validation = validate_skill(content, args.max_words)
            repair_responses: list[dict[str, Any]] = []
            for _ in range(max(0, args.repair_attempts)):
                if validation["ok"]:
                    break
                repair_messages = build_repair_prompt(content, validation, args.max_words)
                content, repair_response = call_chat_completion(
                    api_url=args.api_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=repair_messages,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                    allow_reasoning_fallback=args.allow_reasoning_fallback,
                )
                repair_responses.append(repair_response)
                validation = validate_skill(content, args.max_words)
            if repair_responses:
                response = {"initial_response": response, "repair_responses": repair_responses}
        except Exception as exc:
            response_obj = getattr(exc, "response", None)
            if isinstance(response_obj, dict):
                with response_path.open("w", encoding="utf-8") as f:
                    json.dump(response_obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            with error_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"scenario": scenario, "error_type": type(exc).__name__, "error": str(exc)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            raise
        if args.sleep:
            time.sleep(args.sleep)

    quality = validate_skill(content, args.max_words)
    with skill_path.open("w", encoding="utf-8") as f:
        f.write(content.rstrip() + "\n")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2, sort_keys=True)
    with response_path.open("w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2, sort_keys=True)

    return {
        "scenario": scenario,
        "status": "done",
        "skill_path": str(skill_path),
        "quality_ok": quality["ok"],
        "quality_warnings": quality["warnings"],
        **meta,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    complete_rows = load_clean_rows(args.clean_complete)
    all_rows = load_clean_rows(args.clean_all)
    eval_by_scenario = load_eval_rows(args.baseline_run, args.skill_run)

    positives_by_scenario: dict[str, list[CleanRow]] = defaultdict(list)
    all_by_scenario: dict[str, list[CleanRow]] = defaultdict(list)
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

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(distill_one, scenario, positives_by_scenario, all_by_scenario, eval_by_scenario, args)
            for scenario in scenarios
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            warning_count = len(result.get("quality_warnings", []))
            print(f"{result['status']} {result['scenario']} warnings={warning_count} -> {result.get('skill_path')}")

    with (args.output_dir / "skills_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda x: x["scenario"]):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    warning_counts = Counter()
    for row in results:
        for warning in row.get("quality_warnings", []):
            warning_counts[str(warning).split(":", 1)[0]] += 1

    stats = {
        "output_dir": str(args.output_dir.resolve()),
        "template_version": "v2_verifier_sensitive",
        "scenarios_requested": len(scenarios),
        "done": sum(1 for r in results if r["status"] == "done"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "quality_ok": sum(1 for r in results if r.get("quality_ok") is True),
        "quality_warning_counts": dict(sorted(warning_counts.items())),
        "clean_complete_rows": len(complete_rows),
        "clean_complete_scenarios": len(positives_by_scenario),
        "clean_all_rows": len(all_rows),
        "clean_all_scenarios": len(all_by_scenario),
        "eval_scenarios": len(eval_by_scenario),
    }
    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
