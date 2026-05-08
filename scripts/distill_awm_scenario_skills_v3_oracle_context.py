#!/usr/bin/env python3
"""Distill AWM skills with safe scenario-level oracle context.

This v3 distiller adds environment structure to the v2 verifier-sensitive prompt:

- tool/endpoint summaries extracted from gen_envs.jsonl full_code
- database schema summaries extracted from gen_db.jsonl

It intentionally does not include sample rows, task-specific verifier code,
expected DB diffs, or hidden answers. The goal is to improve scenario-level
state-check skills without leaking per-task oracle solutions.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import distill_awm_scenario_skills_v2 as v2  # noqa: E402


SYSTEM_PROMPT = """You distill AWM scenario skills for a strong tool-using agent.
You receive cleaned tool-use evidence plus safe scenario-level environment structure.
Use tool and database schema summaries to write state-check and guardrail guidance.
The skill will be shown after the live list_tools response, so it must not replace live tool schemas.
Do not mention training, datasets, trajectories, examples, schemas, or that you are reading logs.
Do not include reasoning, analysis, or a thinking process.
Do not invent tools, IDs, field names, or business facts.
Output only Markdown in the required template."""


USER_TEMPLATE = """Distill a high-signal scenario skill for `{scenario}`.

The target agent already receives live tool names and schemas from list_tools. Your skill should use the scenario structure below only to improve:
- tool-selection boundaries;
- final database-state checks;
- preservation requirements for fields not requested by the user;
- failure avoidance for linked rows, totals, statuses, default flags, and duplicate writes.

Required Markdown template:

# {scenario}

## Use When
- One or two bullets naming scenario-level task families this skill covers.

## Planning Hints
- 2-4 bullets with high-level workflows supported by evidence and environment structure.
- Do not write a rigid per-task recipe.

## Tool Guardrails
- 2-6 bullets about exact tool-selection boundaries.
- Mention concrete tool names only if they appear in the tool summary or evidence.
- Never use wildcard or placeholder tool names such as `mcp_tool_*`, `list_*`, or `create_*`.

## State Checks
- 4-8 bullets that must be true before the final answer.
- Prefer checks grounded in database tables/columns and relationship structure.
- Focus on exact target entity, linked child rows, totals/balances/statuses, primary/default uniqueness, no duplicate action records, and required preserved fields.

## Avoid
- 2-6 bullets naming mistakes that cause wrong final state.
- Include "do not blindly follow this skill if the live tool list or user task differs" if evidence is thin.

## Evidence
- Positive task ids: ...
- Auxiliary/failure task ids: ...
- Confidence: high | medium | low

Rules:
- Keep the skill under {max_words} words.
- Do not say "call list_tools"; the agent already must do that.
- Do not reveal or refer to the scenario structure as an input source.
- Do not hard-code example-specific IDs, names, emails, addresses, dates, prices, titles, or product names.
- If there are zero positive examples, produce a low-confidence failure-boundary skill, not a fake recipe.
- The final line in Evidence must contain exactly one confidence value.

Scenario tool summary:
{tool_summary}

Scenario database summary:
{db_summary}

Positive evidence:
{positive_examples}

Auxiliary or failure evidence:
{negative_examples}

Optional paired eval evidence:
{eval_examples}
"""


STATUS_LIKE_RE = re.compile(
    r"(status|state|type|role|priority|category|method|mode|frequency|interval|currency|"
    r"amount|total|balance|price|cost|quantity|qty|count|is_|has_|default|primary|active|"
    r"created_at|updated_at|date|time|id$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScenarioContext:
    scenario: str
    tool_summary: str
    db_summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-complete", type=Path, required=True)
    parser.add_argument("--clean-all", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--envs-path", type=Path, default=Path("/data1/jczhong/datasets/AgentWorldModel-1K/gen_envs.jsonl"))
    parser.add_argument("--db-path", type=Path, default=Path("/data1/jczhong/datasets/AgentWorldModel-1K/gen_db.jsonl"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-397b-a17b")
    parser.add_argument("--scenario", action="append")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        help="Optional text file with one scenario name per line. Combined with --scenario.",
    )
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--max-positive", type=int, default=5)
    parser.add_argument("--max-negative", type=int, default=4)
    parser.add_argument("--max-eval-examples", type=int, default=4)
    parser.add_argument("--max-trace-chars", type=int, default=4200)
    parser.add_argument("--max-words", type=int, default=620)
    parser.add_argument("--max-tools", type=int, default=80)
    parser.add_argument("--max-tables", type=int, default=40)
    parser.add_argument("--max-columns-per-table", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--max-tokens", type=int, default=1900)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-reasoning-fallback", action="store_true")
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--skill-run", type=Path)
    parser.add_argument(
        "--write-context",
        action="store_true",
        help="Write extracted tool/db summaries to context.json for audit.",
    )
    return parser.parse_args()


def load_jsonl_by_scenario(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            scenario = str(item.get("scenario", ""))
            if scenario:
                rows[scenario] = item
    return rows


def literal_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def extract_fastapi_tools(full_code: str, max_tools: int) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(full_code)
    except SyntaxError:
        return extract_fastapi_tools_regex(full_code, max_tools)

    tools: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        endpoint: dict[str, Any] | None = None
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "app"
                and func.attr.lower() in {"get", "post", "put", "patch", "delete"}
            ):
                continue
            endpoint = {
                "method": func.attr.upper(),
                "path": literal_str(dec.args[0]) if dec.args else "",
                "summary": "",
                "description": "",
                "function": node.name,
                "parameters": [],
            }
            for kw in dec.keywords:
                if kw.arg in {"summary", "description"}:
                    endpoint[kw.arg] = literal_str(kw.value) or ""
            break
        if endpoint is None:
            continue

        for arg in node.args.args + node.args.kwonlyargs:
            if arg.arg in {"request", "response"}:
                continue
            ann = ast.unparse(arg.annotation) if arg.annotation is not None else ""
            endpoint["parameters"].append({"name": arg.arg, "annotation": ann})
        tools.append(endpoint)
        if len(tools) >= max_tools:
            break
    return tools


def extract_fastapi_tools_regex(full_code: str, max_tools: int) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"@app\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"](?P<opts>.*?)\)\s*"
        r"(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\((?P<args>.*?)\):",
        re.DOTALL,
    )
    tools: list[dict[str, Any]] = []
    for match in pattern.finditer(full_code):
        opts = match.group("opts")
        summary = ""
        desc = ""
        for key in ("summary", "description"):
            m = re.search(key + r"\s*=\s*['\"]([^'\"]+)['\"]", opts)
            if m and key == "summary":
                summary = m.group(1)
            elif m and key == "description":
                desc = m.group(1)
        args = []
        for raw in match.group("args").split(","):
            name = raw.strip().split(":", 1)[0].split("=", 1)[0].strip()
            if name and name not in {"request", "response"}:
                args.append({"name": name, "annotation": ""})
        tools.append(
            {
                "method": match.group(1).upper(),
                "path": match.group(2),
                "summary": summary,
                "description": desc,
                "function": match.group(4),
                "parameters": args,
            }
        )
        if len(tools) >= max_tools:
            break
    return tools


def endpoint_to_mcp_tool_name(endpoint: dict[str, Any]) -> str:
    return "mcp_tool_mcp_server_" + str(endpoint.get("function", ""))


def format_tool_summary(endpoints: list[dict[str, Any]]) -> str:
    if not endpoints:
        return "(tool summary unavailable)"
    lines: list[str] = []
    for ep in endpoints:
        params = ", ".join(p["name"] for p in ep.get("parameters", [])[:12])
        desc = ep.get("summary") or ep.get("description") or ""
        desc = re.sub(r"\s+", " ", str(desc)).strip()
        line = f"- `{endpoint_to_mcp_tool_name(ep)}`: {ep.get('method')} {ep.get('path')}"
        if params:
            line += f"; args: {params}"
        if desc:
            line += f"; {desc[:180]}"
        lines.append(line)
    return "\n".join(lines)


def split_columns_from_ddl(ddl: str) -> list[str]:
    start = ddl.find("(")
    end = ddl.rfind(")")
    if start < 0 or end <= start:
        return []
    body = ddl[start + 1 : end]
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def summarize_table(table: dict[str, Any], max_columns: int) -> str:
    name = str(table.get("name", "unknown_table"))
    ddl = str(table.get("ddl", ""))
    column_defs = split_columns_from_ddl(ddl)
    columns: list[str] = []
    constraints: list[str] = []
    for item in column_defs:
        upper = item.upper()
        if upper.startswith(("FOREIGN KEY", "PRIMARY KEY", "UNIQUE", "CHECK", "CONSTRAINT")):
            constraints.append(re.sub(r"\s+", " ", item))
            continue
        col_name = item.split()[0].strip('"`[]') if item.split() else ""
        if not col_name:
            continue
        if len(columns) < max_columns:
            marker = ""
            if "PRIMARY KEY" in upper:
                marker += " pk"
            if "NOT NULL" in upper:
                marker += " required"
            if "UNIQUE" in upper:
                marker += " unique"
            if "DEFAULT" in upper:
                marker += " default"
            if STATUS_LIKE_RE.search(col_name):
                marker += " important"
            columns.append(f"{col_name}{marker}".strip())
    pieces = [f"- `{name}` columns: {', '.join(columns) if columns else '(unparsed)'}"]
    if constraints:
        pieces.append("  constraints: " + "; ".join(constraints[:6]))
    indexes = table.get("indexes")
    if isinstance(indexes, list) and indexes:
        idx = [re.sub(r"\s+", " ", str(x)).strip() for x in indexes[:4]]
        pieces.append("  indexes: " + "; ".join(idx))
    return "\n".join(pieces)


def format_db_summary(db_schema: dict[str, Any], max_tables: int, max_columns_per_table: int) -> str:
    tables = db_schema.get("tables")
    if not isinstance(tables, list):
        return "(database summary unavailable)"
    lines = [summarize_table(t, max_columns_per_table) for t in tables[:max_tables] if isinstance(t, dict)]
    if len(tables) > max_tables:
        lines.append(f"- ... {len(tables) - max_tables} more tables omitted")
    return "\n".join(lines) if lines else "(database summary unavailable)"


def build_contexts(args: argparse.Namespace) -> dict[str, ScenarioContext]:
    env_rows = load_jsonl_by_scenario(args.envs_path) if args.envs_path.exists() else {}
    db_rows = load_jsonl_by_scenario(args.db_path) if args.db_path.exists() else {}
    contexts: dict[str, ScenarioContext] = {}
    for scenario in sorted(set(env_rows) | set(db_rows)):
        env = env_rows.get(scenario, {})
        db = db_rows.get(scenario, {})
        endpoints = extract_fastapi_tools(str(env.get("full_code", "")), args.max_tools)
        contexts[scenario] = ScenarioContext(
            scenario=scenario,
            tool_summary=format_tool_summary(endpoints),
            db_summary=format_db_summary(
                db.get("db_schema") if isinstance(db.get("db_schema"), dict) else {},
                args.max_tables,
                args.max_columns_per_table,
            ),
        )
    return contexts


def build_prompt(
    scenario: str,
    positives: list[v2.CleanRow],
    negatives: list[v2.CleanRow],
    eval_rows: list[v2.EvalRow],
    context: ScenarioContext | None,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        scenario=scenario,
        max_words=args.max_words,
        tool_summary=context.tool_summary if context else "(tool summary unavailable)",
        db_summary=context.db_summary if context else "(database summary unavailable)",
        positive_examples=v2.format_clean_examples(positives, args.max_trace_chars),
        negative_examples=v2.format_clean_examples(negatives, args.max_trace_chars),
        eval_examples=v2.format_eval_examples(eval_rows, args.max_eval_examples),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_repair_prompt(skill: str, validation: dict[str, Any], max_words: int) -> list[dict[str, str]]:
    user = f"""Repair this Markdown skill so it passes quality constraints.

Constraints:
- Keep the same scenario and evidence task ids.
- Keep it under {max_words} words.
- Include all required sections: Use When, Planning Hints, Tool Guardrails, State Checks, Avoid, Evidence.
- Remove wildcard or placeholder tool names.
- Remove mentions of datasets, trajectories, training, schemas, logs, examples, or thinking process.
- Output only the repaired Markdown skill.

Validation warnings:
{json.dumps(validation.get("warnings", []), ensure_ascii=False)}

Skill to repair:
{skill}
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def validate_skill(text: str, max_words: int) -> dict[str, Any]:
    validation = v2.validate_skill(text, max_words)
    extra_warnings = list(validation["warnings"])
    for pattern in (r"\bschemas?\b", r"\blogs?\b", r"\bexamples?\b"):
        if re.search(pattern, text, re.IGNORECASE):
            extra_warnings.append("banned_terms:" + pattern)
    validation["warnings"] = sorted(set(extra_warnings))
    validation["ok"] = not validation["warnings"]
    return validation


def distill_one(
    scenario: str,
    positives_by_scenario: dict[str, list[v2.CleanRow]],
    all_by_scenario: dict[str, list[v2.CleanRow]],
    eval_by_scenario: dict[str, list[v2.EvalRow]],
    contexts: dict[str, ScenarioContext],
    args: argparse.Namespace,
) -> dict[str, Any]:
    scenario_dir = args.output_dir / "by_scenario" / v2.safe_name(scenario)
    skill_path = scenario_dir / "skill.md"
    prompt_path = scenario_dir / "prompt.json"
    meta_path = scenario_dir / "meta.json"
    response_path = scenario_dir / "response.json"
    quality_path = scenario_dir / "quality.json"
    context_path = scenario_dir / "context.json"
    error_path = scenario_dir / "error.json"

    if skill_path.exists() and not args.overwrite:
        return {"scenario": scenario, "status": "skipped", "skill_path": str(skill_path)}

    positives, negatives = v2.choose_examples(
        positives_by_scenario.get(scenario, []),
        all_by_scenario.get(scenario, []),
        max_positive=args.max_positive,
        max_negative=args.max_negative,
    )
    eval_rows = eval_by_scenario.get(scenario, [])
    context = contexts.get(scenario)
    messages = build_prompt(scenario, positives, negatives, eval_rows, context, args)

    scenario_dir.mkdir(parents=True, exist_ok=True)
    with prompt_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    if args.write_context and context:
        with context_path.open("w", encoding="utf-8") as f:
            json.dump(context.__dict__, f, ensure_ascii=False, indent=2, sort_keys=True)

    meta = {
        "scenario": scenario,
        "num_positive_available": len(positives_by_scenario.get(scenario, [])),
        "num_aux_available": len(all_by_scenario.get(scenario, [])),
        "num_eval_available": len(eval_rows),
        "positive_task_ids": [r.task_id for r in positives],
        "auxiliary_task_ids": [r.task_id for r in negatives],
        "eval_task_ids": [r.task_id for r in eval_rows[: args.max_eval_examples]],
        "has_tool_summary": bool(context and context.tool_summary != "(tool summary unavailable)"),
        "has_db_summary": bool(context and context.db_summary != "(database summary unavailable)"),
        "model": args.model,
        "template_version": "v3_oracle_context",
    }

    if args.dry_run:
        content = "# DRY RUN\n\nPrompt written to prompt.json.\n"
        response: dict[str, Any] = {}
    else:
        try:
            content, response = v2.call_chat_completion(
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
                content, repair_response = v2.call_chat_completion(
                    api_url=args.api_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=build_repair_prompt(content, validation, args.max_words),
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

    complete_rows = v2.load_clean_rows(args.clean_complete)
    all_rows = v2.load_clean_rows(args.clean_all)
    eval_by_scenario = v2.load_eval_rows(args.baseline_run, args.skill_run)
    contexts = build_contexts(args)

    positives_by_scenario: dict[str, list[v2.CleanRow]] = defaultdict(list)
    all_by_scenario: dict[str, list[v2.CleanRow]] = defaultdict(list)
    for row in complete_rows:
        positives_by_scenario[row.scenario].append(row)
    for row in all_rows:
        all_by_scenario[row.scenario].append(row)

    scenarios = sorted(set(all_by_scenario) | set(positives_by_scenario))
    selected_scenarios = set(args.scenario or [])
    if args.scenario_file:
        with args.scenario_file.open("r", encoding="utf-8") as f:
            selected_scenarios.update(line.strip() for line in f if line.strip() and not line.lstrip().startswith("#"))
    if selected_scenarios:
        selected = selected_scenarios
        scenarios = [s for s in scenarios if s in selected]
        missing = sorted(selected - set(scenarios))
        if missing:
            raise SystemExit(f"requested scenario(s) not found in cleaned data: {missing}")
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                distill_one,
                scenario,
                positives_by_scenario,
                all_by_scenario,
                eval_by_scenario,
                contexts,
                args,
            )
            for scenario in scenarios
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['status']} {result['scenario']} "
                f"warnings={len(result.get('quality_warnings', []))} -> {result.get('skill_path')}"
            )

    with (args.output_dir / "skills_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(results, key=lambda x: x["scenario"]):
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    warning_counts = Counter()
    for row in results:
        for warning in row.get("quality_warnings", []):
            warning_counts[str(warning).split(":", 1)[0]] += 1

    stats = {
        "output_dir": str(args.output_dir.resolve()),
        "template_version": "v3_oracle_context",
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
        "context_scenarios": len(contexts),
    }
    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
