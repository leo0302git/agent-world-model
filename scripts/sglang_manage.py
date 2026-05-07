#!/usr/bin/env python3
"""Manage a local SGLang OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_service_paths(served_model_name: str) -> tuple[Path, Path]:
    safe = served_model_name.replace("/", "_").replace(":", "_")
    service_dir = REPO_ROOT / "outputs" / "services"
    return service_dir / f"{safe}.pid", service_dir / f"{safe}.log"


def models_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/models"


def check_models(api_url: str, api_key: str = "EMPTY", timeout: float = 5.0) -> tuple[bool, str]:
    req = urllib.request.Request(
        api_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            json.loads(body)
        return True, "ok"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, str(exc)


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_server(api_url: str, api_key: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, _ = check_models(api_url, api_key=api_key, timeout=5)
        if ok:
            return True
        time.sleep(5)
    return False


def cmd_start(args: argparse.Namespace) -> int:
    pid_file, log_file = default_service_paths(args.served_model_name)
    pid_file = args.pid_file or pid_file
    log_file = args.log_file or log_file
    api_url = f"http://{args.host}:{args.port}/v1"

    ok, detail = check_models(api_url, api_key=args.api_key)
    if ok:
        print(f"SGLang already reachable: {api_url}")
        return 0

    existing_pid = read_pid(pid_file)
    if existing_pid and pid_alive(existing_pid) and not args.force:
        print(f"PID file exists and process is alive, but API is not ready: pid={existing_pid}")
        print(f"Use --force to start a new server anyway. Last API error: {detail}")
        return 1

    if not args.model_path.exists():
        print(f"model path does not exist: {args.model_path}")
        return 1

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model_path),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--context-length",
        str(args.context_length),
    ]
    if args.tp is not None:
        cmd.extend(["--tp", str(args.tp)])
    if args.mem_fraction_static is not None:
        cmd.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    if args.extra_args:
        cmd.extend(args.extra_args)

    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    with log_file.open("a", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    pid_file.write_text(str(proc.pid), encoding="utf-8")
    print(f"Started SGLang pid={proc.pid}")
    print(f"API: {api_url}")
    print(f"PID: {pid_file}")
    print(f"LOG: {log_file}")

    if args.no_wait:
        return 0
    if wait_for_server(api_url, args.api_key, args.wait_timeout):
        print("SGLang is ready.")
        return 0
    print(f"SGLang did not become ready within {args.wait_timeout}s; check {log_file}")
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    pid = read_pid(args.pid_file) if args.pid_file else None
    if pid is not None:
        print(f"pid: {pid}")
        print(f"pid_alive: {pid_alive(pid)}")
    else:
        print("pid: unknown")

    ok, detail = check_models(args.api_url, api_key=args.api_key)
    print(f"api_url: {args.api_url}")
    print(f"models_api: {ok}")
    if not ok:
        print(f"error: {detail}")
    return 0 if ok else 1


def cmd_stop(args: argparse.Namespace) -> int:
    pid = read_pid(args.pid_file)
    if pid is None:
        print(f"cannot read pid file: {args.pid_file}")
        return 1
    if not pid_alive(pid):
        print(f"pid is not alive: {pid}")
        if args.remove_stale_pid:
            args.pid_file.unlink(missing_ok=True)
            print(f"removed stale pid file: {args.pid_file}")
        return 0

    print(f"Stopping pid={pid}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            args.pid_file.unlink(missing_ok=True)
            print("stopped")
            return 0
        time.sleep(1)

    if args.kill:
        print(f"SIGTERM timed out after {args.timeout}s; sending SIGKILL")
        os.kill(pid, signal.SIGKILL)
        args.pid_file.unlink(missing_ok=True)
        return 0

    print(f"still alive after {args.timeout}s; rerun with --kill to force")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="Start or reuse an SGLang server")
    start.add_argument("--model-path", type=Path, required=True)
    start.add_argument("--served-model-name", required=True)
    start.add_argument("--gpu", default="")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--api-key", default="EMPTY")
    start.add_argument("--context-length", type=int, default=65536)
    start.add_argument("--tp", type=int)
    start.add_argument("--mem-fraction-static", type=float)
    start.add_argument("--pid-file", type=Path)
    start.add_argument("--log-file", type=Path)
    start.add_argument("--wait-timeout", type=int, default=900)
    start.add_argument("--no-wait", action="store_true")
    start.add_argument("--force", action="store_true")
    start.add_argument("extra_args", nargs=argparse.REMAINDER)
    start.set_defaults(func=cmd_start)

    status = sub.add_parser("status", help="Check an SGLang server")
    status.add_argument("--api-url", required=True)
    status.add_argument("--api-key", default="EMPTY")
    status.add_argument("--pid-file", type=Path)
    status.set_defaults(func=cmd_status)

    stop = sub.add_parser("stop", help="Stop an SGLang server by pid file")
    stop.add_argument("--pid-file", type=Path, required=True)
    stop.add_argument("--timeout", type=int, default=30)
    stop.add_argument("--kill", action="store_true")
    stop.add_argument("--remove-stale-pid", action="store_true", default=True)
    stop.set_defaults(func=cmd_stop)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
