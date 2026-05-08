#!/usr/bin/env python3
"""Shared parallel scoring utilities for AWM scripts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VALID_REWARD_TYPES = {
    "complete",
    "others",
    "judge_error",
    "incomplete",
    "server_error",
    "agent_error",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PROCS: set[subprocess.Popen] = set()
ACTIVE_PROCS_LOCK = threading.Lock()


@dataclass(frozen=True)
class TaskItem:
    global_idx: int
    worker_id: int
    scenario: str
    scenario_dir: str
    task_id: int
    output_dir: str
    preferred_port: int


@dataclass
class RunnerConfig:
    data: Path
    api_url: str
    api_urls: list[str]
    api_key: str
    model: str
    run_name: str
    workers: int
    base_port: int
    port_stride: int
    scenario_limit: int | None
    task_ids: list[int]
    verify_mode: str
    max_iterations: int
    max_tokens: int
    temperature: float
    judge_api_url: str | None
    judge_api_key: str | None
    judge_model: str | None
    judge_provider: str | None
    skill_dir: Path | None
    resume: bool
    verbose: bool
    run_root: Path


def normalize_scenario_name(scenario: str) -> str:
    import re

    s = scenario.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_").strip()


def parse_task_ids(value: str) -> list[int]:
    task_ids: set[int] = set()
    for part in value.replace(",", " ").split():
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid task id range: {part}")
            task_ids.update(range(start, end + 1))
        else:
            task_ids.add(int(part))
    if not task_ids:
        raise argparse.ArgumentTypeError("no task ids provided")
    return sorted(task_ids)


def default_run_name(prefix: str, model: str) -> str:
    safe_model = model.split("/")[-1].replace(" ", "_").replace(":", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{safe_model}_{timestamp}"


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_scenarios(tasks_path: Path, scenario_limit: int | None) -> list[tuple[str, int]]:
    scenarios: list[tuple[str, int]] = []
    with tasks_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if scenario_limit is not None and len(scenarios) >= scenario_limit:
                break
            item = json.loads(line)
            scenarios.append((item["scenario"], len(item.get("tasks", []))))
    if not scenarios:
        raise ValueError(f"no scenarios loaded from {tasks_path}")
    return scenarios


def build_manifest(config: RunnerConfig) -> list[TaskItem]:
    scenarios = load_scenarios(config.data / "gen_tasks.jsonl", config.scenario_limit)
    tasks: list[TaskItem] = []
    global_idx = 0
    for scenario, task_count in scenarios:
        scenario_dir = normalize_scenario_name(scenario)
        for task_id in config.task_ids:
            if task_id < 0 or task_id >= task_count:
                continue
            worker_id = global_idx % config.workers
            worker_port_start = config.base_port + worker_id * config.port_stride
            preferred_port = worker_port_start + (global_idx // config.workers) % config.port_stride
            output_dir = config.run_root / scenario_dir / f"task_{task_id}"
            tasks.append(
                TaskItem(
                    global_idx=global_idx,
                    worker_id=worker_id,
                    scenario=scenario,
                    scenario_dir=scenario_dir,
                    task_id=task_id,
                    output_dir=str(output_dir),
                    preferred_port=preferred_port,
                )
            )
            global_idx += 1
    return tasks


def write_manifest(config: RunnerConfig, tasks: list[TaskItem]) -> None:
    config.run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": config.run_name,
        "run_root": str(config.run_root),
        "data": str(config.data),
        "api_url": config.api_url,
        "api_urls": config.api_urls,
        "model": config.model,
        "workers": config.workers,
        "base_port": config.base_port,
        "port_stride": config.port_stride,
        "scenario_limit": config.scenario_limit,
        "task_ids": config.task_ids,
        "verify_mode": config.verify_mode,
        "judge_api_url": config.judge_api_url,
        "judge_model": config.judge_model,
        "judge_provider": config.judge_provider,
        "skill_dir": str(config.skill_dir) if config.skill_dir else None,
        "tasks": [asdict(task) for task in tasks],
    }
    with (config.run_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_port(task: TaskItem, config: RunnerConfig) -> int:
    start = config.base_port + task.worker_id * config.port_stride
    end = start + config.port_stride
    preferred_offset = max(0, min(task.preferred_port - start, config.port_stride - 1))
    for step in range(config.port_stride):
        port = start + (preferred_offset + step) % config.port_stride
        if port >= end:
            continue
        if is_port_available(port):
            return port
    raise RuntimeError(f"no free MCP port in worker {task.worker_id} range {start}-{end - 1}")


def verify_path_for(task: TaskItem, mode: str) -> Path:
    return Path(task.output_dir) / f"verify.{mode}.json"


def is_task_complete(task: TaskItem, mode: str) -> bool:
    verify_path = verify_path_for(task, mode)
    if not verify_path.exists():
        return False
    try:
        with verify_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return (
        isinstance(data, dict)
        and data.get("mode") == mode
        and data.get("reward_type") in VALID_REWARD_TYPES
        and "verify_result" in data
    )


def run_command(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=REPO_ROOT,
            text=True,
        )
        with ACTIVE_PROCS_LOCK:
            ACTIVE_PROCS.add(proc)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        finally:
            with ACTIVE_PROCS_LOCK:
                ACTIVE_PROCS.discard(proc)


def terminate_active_processes(timeout: float = 10.0) -> None:
    with ACTIVE_PROCS_LOCK:
        procs = [proc for proc in ACTIVE_PROCS if proc.poll() is None]
    for proc in procs:
        proc.terminate()
    deadline = time.time() + timeout
    for proc in procs:
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_task(task: TaskItem, config: RunnerConfig) -> dict:
    if config.resume and is_task_complete(task, config.verify_mode):
        return {"status": "skipped", "task": asdict(task)}

    output_dir = Path(task.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_url = config.api_urls[task.worker_id % len(config.api_urls)]
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = api_url
    env["OPENAI_API_KEY"] = config.api_key
    env["AWM_RUN_NAME"] = config.run_name
    env["AWM_SYN_OVERRIDE_MODEL"] = config.model

    port = choose_port(task, config)
    agent_cmd = [
        "uv",
        "run",
        "awm",
        "agent",
        "--scenario",
        task.scenario,
        "--task_id",
        str(task.task_id),
        "--envs_path",
        str(config.data / "gen_envs.jsonl"),
        "--tasks_path",
        str(config.data / "gen_tasks.jsonl"),
        "--db_path",
        str(config.data / "gen_db.jsonl"),
        "--sample_path",
        str(config.data / "gen_sample.jsonl"),
        "--api_url",
        api_url,
        "--model",
        config.model,
        "--run_root",
        str(config.run_root),
        "--mcp_port",
        str(port),
        "--max_iterations",
        str(config.max_iterations),
        "--temperature",
        str(config.temperature),
        "--max_tokens",
        str(config.max_tokens),
        "--verbose",
        str(config.verbose),
    ]
    if config.skill_dir:
        agent_cmd.extend(["--skill_dir", str(config.skill_dir)])

    agent_rc = run_command(agent_cmd, env, output_dir / "runner_agent.log")
    if agent_rc != 0:
        return {"status": "agent_failed", "returncode": agent_rc, "task": asdict(task), "port": port}

    verify_cmd = [
        "uv",
        "run",
        "awm",
        "verify",
        "--input",
        str(output_dir),
        "--mode",
        config.verify_mode,
    ]
    if config.verify_mode == "code":
        verify_cmd.extend(["--verifier_code_path", str(config.data / "gen_verifier.pure_code.jsonl")])
    else:
        verify_cmd.extend(["--verifier_path", str(config.data / "gen_verifier.jsonl")])

    verify_env = env.copy()
    if config.verify_mode == "sql":
        if config.judge_api_url:
            verify_env["OPENAI_BASE_URL"] = config.judge_api_url
        if config.judge_api_key:
            verify_env["OPENAI_API_KEY"] = config.judge_api_key
        if config.judge_model:
            verify_env["AWM_SYN_OVERRIDE_MODEL"] = config.judge_model
        if config.judge_provider:
            verify_env["AWM_SYN_LLM_PROVIDER"] = config.judge_provider

    verify_rc = run_command(verify_cmd, verify_env, output_dir / "runner_verify.log")
    if verify_rc != 0:
        return {"status": "verify_failed", "returncode": verify_rc, "task": asdict(task), "port": port}

    return {"status": "done", "task": asdict(task), "port": port}


def collect_results(run_root: Path, mode: str) -> tuple[list[dict], dict]:
    results: list[dict] = []
    bad_files: list[dict] = []
    for verify_path in sorted(run_root.glob(f"*/task_*/verify.{mode}.json")):
        try:
            with verify_path.open("r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception as exc:
            bad_files.append({"path": str(verify_path.relative_to(run_root)), "error": str(exc)})
            continue
        if not isinstance(item, dict):
            bad_files.append({"path": str(verify_path.relative_to(run_root)), "error": "verify JSON is not an object"})
            continue
        item["run_dir"] = str(verify_path.parent.relative_to(run_root))
        results.append(item)

    results.sort(key=lambda x: (str(x.get("scenario", "")), int(x.get("task_id", -1))))
    counts = Counter(str(item.get("reward_type", "unknown")) for item in results)
    complete = counts.get("complete", 0)
    summary = {
        "run_root": str(run_root),
        "mode": mode,
        "total": len(results),
        "complete": complete,
        "others": counts.get("others", 0),
        "judge_error": counts.get("judge_error", 0),
        "counts": dict(sorted(counts.items())),
        "score": complete / len(results) if results else 0.0,
        "bad_verify_files": bad_files,
    }
    return results, summary


def write_summary(run_root: Path, mode: str) -> dict:
    results, summary = collect_results(run_root, mode)
    with (run_root / "results.jsonl").open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    with (run_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def run_parallel(config: RunnerConfig) -> dict:
    start_monotonic = time.monotonic()
    started_at = utc_now_iso()
    if config.workers < 1:
        raise ValueError("--workers must be >= 1")
    if config.port_stride < 1:
        raise ValueError("--port-stride must be >= 1")
    if config.verify_mode not in ("code", "sql"):
        raise ValueError("--verify-mode must be code or sql")

    tasks = build_manifest(config)
    write_manifest(config, tasks)
    print(f"run_root={config.run_root}")
    print(
        f"tasks={len(tasks)} workers={config.workers} "
        f"endpoints={len(config.api_urls)} verify_mode={config.verify_mode}"
    )

    statuses: Counter[str] = Counter()
    failures: list[dict] = []
    interrupted = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.workers)
    future_to_task = {executor.submit(run_task, task, config): task for task in tasks}
    try:
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "exception", "error": str(exc), "task": asdict(task)}
            status = str(result.get("status", "unknown"))
            statuses[status] += 1
            if status not in {"done", "skipped"}:
                failures.append(result)
            print(
                f"[{sum(statuses.values())}/{len(tasks)}] {status} "
                f"{task.scenario_dir}/task_{task.task_id}"
            )
    except KeyboardInterrupt:
        interrupted = True
        for future in future_to_task:
            future.cancel()
        terminate_active_processes()
        executor.shutdown(wait=False, cancel_futures=True)
        print("Interrupted; writing summary for completed tasks.", file=sys.stderr)
    finally:
        if not interrupted:
            executor.shutdown(wait=True)
        summary = write_summary(config.run_root, config.verify_mode)
        elapsed_seconds = time.monotonic() - start_monotonic
        summary["started_at"] = started_at
        summary["finished_at"] = utc_now_iso()
        summary["elapsed_seconds"] = round(elapsed_seconds, 3)
        summary["elapsed"] = format_duration(elapsed_seconds)
        summary["runner_statuses"] = dict(sorted(statuses.items()))
        summary["runner_failures"] = failures
        summary["interrupted"] = interrupted
        with (config.run_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=Path("/data1/jczhong/datasets/AgentWorldModel-1K"))
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=9100)
    parser.add_argument("--port-stride", type=int, default=100)
    parser.add_argument("--scenario-limit", type=int)
    parser.add_argument("--task-ids", type=parse_task_ids, default=parse_task_ids("0-9"))
    parser.add_argument("--verify-mode", choices=["code", "sql"], default="code")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--judge-api-url")
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY"))
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-provider")
    parser.add_argument("--skill-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)


def config_from_args(args: argparse.Namespace, prefix: str) -> RunnerConfig:
    run_name = args.run_name or default_run_name(prefix, args.model)
    run_root = REPO_ROOT / "outputs" / "runs" / run_name
    api_urls = [url.strip().rstrip("/") for url in args.api_url.split(",") if url.strip()]
    if not api_urls:
        raise ValueError("--api-url must contain at least one endpoint")
    return RunnerConfig(
        data=args.data,
        api_url=api_urls[0],
        api_urls=api_urls,
        api_key=args.api_key,
        model=args.model,
        run_name=run_name,
        workers=args.workers,
        base_port=args.base_port,
        port_stride=args.port_stride,
        scenario_limit=args.scenario_limit,
        task_ids=args.task_ids,
        verify_mode=args.verify_mode,
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        judge_api_url=args.judge_api_url,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        skill_dir=args.skill_dir,
        resume=args.resume,
        verbose=args.verbose,
        run_root=run_root,
    )


def check_required_data_files(data: Path) -> None:
    required = [
        "gen_tasks.jsonl",
        "gen_envs.jsonl",
        "gen_db.jsonl",
        "gen_sample.jsonl",
        "gen_verifier.pure_code.jsonl",
    ]
    missing = [name for name in required if not (data / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset files under {data}: {', '.join(missing)}")
