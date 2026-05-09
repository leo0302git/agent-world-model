#!/usr/bin/env bash
# Unattended pipeline:
# 1. Finish/adopt 5x 7B runtime runs on the static code-verify allowlist.
# 2. Build a conservative v2 runtime allowlist.
# 3. Stop 7B, start 397B, and run one 397B pass on the v2 allowlist.

set -uo pipefail

REPO=${REPO:-/data1/jczhong/repos/agent-world-model}
cd "$REPO" || exit 1

PY=${PY:-"uv run python"}
ALLOWLIST=${ALLOWLIST:-outputs/task_allowlists/static_code_verify/allowlist.jsonl}
V2_OUT=${V2_OUT:-outputs/task_allowlists/runtime_7b_static8314_w256_5runs_v2}

SEVEN_B_MODEL=${SEVEN_B_MODEL:-qwen2.5-7b-instruct}
SEVEN_B_MODEL_PATH=${SEVEN_B_MODEL_PATH:-/data1/jczhong/models/Qwen2.5-7B-Instruct}
SEVEN_B_API_URLS=${SEVEN_B_API_URLS:-http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1,http://127.0.0.1:8103/v1,http://127.0.0.1:8104/v1,http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1,http://127.0.0.1:8107/v1}
SEVEN_B_WORKERS=${SEVEN_B_WORKERS:-256}
SEVEN_B_PORT_STRIDE=${SEVEN_B_PORT_STRIDE:-64}
SEVEN_B_RUN_PREFIX=${SEVEN_B_RUN_PREFIX:-awm_qwen25_7b_static8314_runtime_w256p64}
SEVEN_B_MAX_ITERATIONS=${SEVEN_B_MAX_ITERATIONS:-30}
SEVEN_B_TEMPERATURE=${SEVEN_B_TEMPERATURE:-0.6}

Q397_MODEL=${Q397_MODEL:-qwen3.5-397b-a17b}
Q397_MODEL_PATH=${Q397_MODEL_PATH:-/data1/models/Qwen/Qwen3.5-397B-A17B}
Q397_API_URL=${Q397_API_URL:-http://127.0.0.1:8000/v1}
Q397_WORKERS=${Q397_WORKERS:-48}
Q397_BASE_PORT=${Q397_BASE_PORT:-31000}
Q397_PORT_STRIDE=${Q397_PORT_STRIDE:-100}
Q397_RUN_NAME=${Q397_RUN_NAME:-awm_qwen397b_v2allowlist_runtime_w48_v1}
Q397_MAX_ITERATIONS=${Q397_MAX_ITERATIONS:-30}
Q397_TEMPERATURE=${Q397_TEMPERATURE:-0.6}

RUN_397B=${RUN_397B:-1}
STALL_TIMEOUT_SECONDS=${STALL_TIMEOUT_SECONDS:-3600}
POLL_SECONDS=${POLL_SECONDS:-300}
LOG_DIR=${LOG_DIR:-outputs/logs/unattended_allowlist_v2}
mkdir -p "$LOG_DIR" outputs/services
LOG_FILE="$LOG_DIR/pipeline_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export SGLANG_DISABLE_CUDNN_CHECK=1

SEVEN_B_BASE_PORTS=(15000 32000 48000 15000 32000)

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

run_root() {
  printf 'outputs/runs/%s\n' "$1"
}

allowlist_count() {
  wc -l < "$ALLOWLIST" | tr -d ' '
}

json_field() {
  local path=$1
  local expr=$2
  python - "$path" "$expr" <<'PY'
import json
import sys
path, expr = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
cur = data
for part in expr.split("."):
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break
print("" if cur is None else cur)
PY
}

verified_count() {
  local root=$1
  if [[ ! -d "$root" ]]; then
    echo 0
    return
  fi
  find "$root" -name verify.code.json | wc -l | tr -d ' '
}

run_parent_alive() {
  local run_name=$1
  pgrep -f "[r]un_parallel_local_score.py .*--run-name ${run_name}" >/dev/null 2>&1
}

summary_complete_enough() {
  local run_name=$1
  local expected=$2
  local summary
  summary="$(run_root "$run_name")/summary.json"
  [[ -f "$summary" ]] || return 1
  python - "$summary" "$expected" <<'PY'
import json
import sys
summary_path, expected_s = sys.argv[1], sys.argv[2]
expected = int(expected_s)
try:
    data = json.load(open(summary_path, "r", encoding="utf-8"))
except Exception:
    raise SystemExit(1)
statuses = data.get("runner_statuses") or {}
attempted = sum(int(v or 0) for v in statuses.values())
interrupted = bool(data.get("interrupted"))
verified = int(data.get("total") or 0)
# runner_failures make run_parallel_local_score exit 1; that is acceptable if
# almost every allowlisted task reached a terminal runner status.
ok = (not interrupted) and attempted >= int(expected * 0.995) and verified > 0
print(f"attempted={attempted} verified={verified} interrupted={interrupted}")
raise SystemExit(0 if ok else 1)
PY
}

cleanup_run_processes() {
  local run_name=$1
  log "Cleaning residual processes for ${run_name}"
  local pids
  pids="$(pgrep -f "$run_name" 2>/dev/null | grep -v "^$$\$" || true)"
  if [[ -n "$pids" ]]; then
    kill -TERM $pids 2>/dev/null || true
    sleep 10
  fi
  pids="$(pgrep -f "$run_name" 2>/dev/null | grep -v "^$$\$" || true)"
  if [[ -n "$pids" ]]; then
    kill -KILL $pids 2>/dev/null || true
    sleep 3
  fi
}

wait_for_existing_run() {
  local run_name=$1
  local expected=$2
  local root
  root="$(run_root "$run_name")"

  if summary_complete_enough "$run_name" "$expected"; then
    log "${run_name} already complete enough."
    cleanup_run_processes "$run_name"
    return 0
  fi

  if ! run_parent_alive "$run_name"; then
    log "${run_name} is not running and is not complete."
    return 1
  fi

  log "${run_name} is running; waiting before starting the remaining long jobs."
  local last_count
  local last_progress_ts
  last_count="$(verified_count "$root")"
  last_progress_ts="$(date +%s)"

  while run_parent_alive "$run_name"; do
    sleep "$POLL_SECONDS"
    local now count idle
    now="$(date +%s)"
    count="$(verified_count "$root")"
    if [[ "$count" != "$last_count" ]]; then
      last_count="$count"
      last_progress_ts="$now"
    fi
    idle=$((now - last_progress_ts))
    log "${run_name}: verify_files=${count}/${expected}, idle_seconds=${idle}"
    if (( idle >= STALL_TIMEOUT_SECONDS )); then
      log "${run_name} appears stalled; terminating and will resume it."
      cleanup_run_processes "$run_name"
      return 1
    fi
  done

  if summary_complete_enough "$run_name" "$expected"; then
    log "${run_name} finished while waiting."
    cleanup_run_processes "$run_name"
    return 0
  fi
  log "${run_name} exited without a complete summary; it will be resumed."
  cleanup_run_processes "$run_name"
  return 1
}

run_command_with_progress_watch() {
  local run_name=$1
  local expected=$2
  shift 2
  local root
  root="$(run_root "$run_name")"

  "$@" &
  local runner_pid=$!
  log "${run_name}: started pid=${runner_pid}"

  local last_count
  local last_progress_ts
  last_count="$(verified_count "$root")"
  last_progress_ts="$(date +%s)"

  while kill -0 "$runner_pid" >/dev/null 2>&1; do
    sleep "$POLL_SECONDS"
    local now count idle
    now="$(date +%s)"
    count="$(verified_count "$root")"
    if [[ "$count" != "$last_count" ]]; then
      last_count="$count"
      last_progress_ts="$now"
    fi
    idle=$((now - last_progress_ts))
    log "${run_name}: verify_files=${count}/${expected}, idle_seconds=${idle}"
    if (( idle >= STALL_TIMEOUT_SECONDS )); then
      log "${run_name}: no verify progress for ${idle}s; terminating stuck runner and residual children."
      cleanup_run_processes "$run_name"
      wait "$runner_pid" >/dev/null 2>&1 || true
      return 124
    fi
  done

  wait "$runner_pid"
  return $?
}

check_models() {
  local api_url=$1
  python - "$api_url" <<'PY'
import json
import sys
import urllib.request
url = sys.argv[1].rstrip("/") + "/models"
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        json.loads(resp.read().decode("utf-8", errors="replace"))
except Exception as exc:
    print(exc)
    raise SystemExit(1)
raise SystemExit(0)
PY
}

wait_for_api() {
  local api_url=$1
  local timeout=$2
  local deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    if check_models "$api_url" >/dev/null 2>&1; then
      log "API ready: ${api_url}"
      return 0
    fi
    sleep 10
  done
  log "API did not become ready: ${api_url}"
  return 1
}

ensure_7b_endpoints() {
  log "Checking 7B endpoints."
  local all_ok=1
  for gpu in 0 1 2 3 4 5 6 7; do
    local port=$((8100 + gpu))
    local url="http://127.0.0.1:${port}/v1"
    if check_models "$url" >/dev/null 2>&1; then
      log "7B endpoint already ready: ${url}"
      continue
    fi
    all_ok=0
    log "Starting 7B endpoint gpu=${gpu} port=${port}"
    tmux kill-session -t "sglang_qwen25_7b_gpu${gpu}" >/dev/null 2>&1 || true
    tmux new-session -d -s "sglang_qwen25_7b_gpu${gpu}" \
      "cd '$REPO'; CUDA_VISIBLE_DEVICES=${gpu} SGLANG_DISABLE_CUDNN_CHECK=1 /usr/bin/python -m sglang.launch_server --model-path '$SEVEN_B_MODEL_PATH' --served-model-name '$SEVEN_B_MODEL' --host 127.0.0.1 --port ${port} --context-length 32768 --mem-fraction-static 0.85 > outputs/services/qwen2.5-7b-gpu${gpu}.log 2>&1"
  done

  for gpu in 0 1 2 3 4 5 6 7; do
    local port=$((8100 + gpu))
    wait_for_api "http://127.0.0.1:${port}/v1" 1200 || return 1
  done
  if [[ "$all_ok" -eq 0 ]]; then
    log "One or more 7B endpoints were started/restarted."
  fi
}

run_7b_round() {
  local idx=$1
  local base_port=$2
  local expected=$3
  local run_name="${SEVEN_B_RUN_PREFIX}_v${idx}"

  if wait_for_existing_run "$run_name" "$expected"; then
    return 0
  fi

  ensure_7b_endpoints || return 1
  cleanup_run_processes "$run_name"
  log "Starting/resuming 7B round ${idx}: ${run_name}"
  run_command_with_progress_watch "$run_name" "$expected" $PY scripts/run_parallel_local_score.py \
    --api-url "$SEVEN_B_API_URLS" \
    --model "$SEVEN_B_MODEL" \
    --run-name "$run_name" \
    --workers "$SEVEN_B_WORKERS" \
    --base-port "$base_port" \
    --port-stride "$SEVEN_B_PORT_STRIDE" \
    --verify-mode code \
    --task-allowlist-jsonl "$ALLOWLIST" \
    --skill-reminder-interval 0 \
    --max-iterations "$SEVEN_B_MAX_ITERATIONS" \
    --temperature "$SEVEN_B_TEMPERATURE"
  local rc=$?
  log "7B round ${idx} command exited rc=${rc}; checking summary."

  if summary_complete_enough "$run_name" "$expected"; then
    cleanup_run_processes "$run_name"
    return 0
  fi

  log "7B round ${idx} incomplete; retrying once with resume."
  ensure_7b_endpoints || true
  run_command_with_progress_watch "$run_name" "$expected" $PY scripts/run_parallel_local_score.py \
    --api-url "$SEVEN_B_API_URLS" \
    --model "$SEVEN_B_MODEL" \
    --run-name "$run_name" \
    --workers "$SEVEN_B_WORKERS" \
    --base-port "$base_port" \
    --port-stride "$SEVEN_B_PORT_STRIDE" \
    --verify-mode code \
    --task-allowlist-jsonl "$ALLOWLIST" \
    --skill-reminder-interval 0 \
    --max-iterations "$SEVEN_B_MAX_ITERATIONS" \
    --temperature "$SEVEN_B_TEMPERATURE"
  rc=$?
  log "7B round ${idx} retry exited rc=${rc}; checking summary."
  cleanup_run_processes "$run_name"
  summary_complete_enough "$run_name" "$expected"
}

build_v2_allowlist() {
  log "Building v2 allowlist at ${V2_OUT}"
  mkdir -p "$V2_OUT"
  local roots=()
  for idx in 1 2 3 4 5; do
    roots+=("outputs/runs/${SEVEN_B_RUN_PREFIX}_v${idx}")
  done
  $PY scripts/analyze_run_trajectories.py "${roots[@]}" \
    --allowlist-jsonl "$ALLOWLIST" \
    --out-dir "$V2_OUT" \
    --mode code \
    --strict-all-hard-reject
  local rc=$?
  if [[ "$rc" -ne 0 || ! -s "$V2_OUT/allowlist.jsonl" ]]; then
    log "v2 allowlist analysis failed or produced an empty allowlist."
    return 1
  fi
  safeguard_v2_allowlist
  log "v2 allowlist ready: $(wc -l < "$V2_OUT/allowlist.jsonl" | tr -d ' ') tasks"
}

safeguard_v2_allowlist() {
  log "Applying v2 missing-evidence safeguard."
  python - "$ALLOWLIST" "$V2_OUT" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])

def key(row):
    return f"{row.get('scenario')}/task_{int(row.get('task_id'))}"

original = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            original.append(json.loads(line))

analyzed = set()
task_report = out / "task_report.jsonl"
if task_report.exists():
    with task_report.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                analyzed.add(key(json.loads(line)))

allowlist_path = out / "allowlist.jsonl"
existing_rows = []
existing = set()
if allowlist_path.exists():
    with allowlist_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                existing_rows.append(row)
                existing.add(key(row))

missing = [row for row in original if key(row) not in analyzed]
append_rows = [row for row in missing if key(row) not in existing]

if append_rows:
    with allowlist_path.open("a", encoding="utf-8") as f:
        for row in append_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "missing_from_analysis.jsonl").open("w", encoding="utf-8") as f:
        for row in missing:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

stats_path = out / "stats.json"
try:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
except Exception:
    stats = {}
stats["source_allowlist_tasks"] = len(original)
stats["missing_from_analysis"] = len(missing)
stats["missing_appended_to_allowlist"] = len(append_rows)
stats["final_allowlist_tasks"] = sum(1 for line in allowlist_path.open("r", encoding="utf-8") if line.strip())
stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(json.dumps({
    "source_allowlist_tasks": len(original),
    "analyzed_tasks": len(analyzed),
    "missing_from_analysis": len(missing),
    "missing_appended_to_allowlist": len(append_rows),
    "final_allowlist_tasks": stats["final_allowlist_tasks"],
}, ensure_ascii=False, sort_keys=True))
PY
}

stop_7b_endpoints() {
  log "Stopping 7B endpoints to release GPU memory."
  for gpu in 0 1 2 3 4 5 6 7; do
    tmux kill-session -t "sglang_qwen25_7b_gpu${gpu}" >/dev/null 2>&1 || true
  done
  local pids
  pids="$(pgrep -f "[s]glang.launch_server.*qwen2.5-7b-instruct|[Q]wen2.5-7B-Instruct" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -TERM $pids 2>/dev/null || true
    sleep 30
  fi
  pids="$(pgrep -f "[s]glang.launch_server.*qwen2.5-7b-instruct|[Q]wen2.5-7B-Instruct" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -KILL $pids 2>/dev/null || true
    sleep 10
  fi
}

start_397b() {
  if check_models "$Q397_API_URL" >/dev/null 2>&1; then
    log "397B endpoint already ready: ${Q397_API_URL}"
    return 0
  fi
  log "Starting 397B endpoint; this can take 30+ minutes."
  $PY scripts/sglang_manage.py start \
    --model-path "$Q397_MODEL_PATH" \
    --served-model-name "$Q397_MODEL" \
    --gpu 0,1,2,3,4,5,6,7 \
    --host 127.0.0.1 \
    --port 8000 \
    --context-length 32768 \
    --tp 8 \
    --mem-fraction-static 0.88 \
    --wait-timeout 3600
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    log "397B start command returned rc=${rc}; checking API anyway."
  fi
  wait_for_api "$Q397_API_URL" 1800
}

run_397b() {
  [[ "$RUN_397B" == "1" ]] || {
    log "RUN_397B=${RUN_397B}; skipping 397B stage."
    return 0
  }
  local v2_allowlist="$V2_OUT/allowlist.jsonl"
  [[ -s "$v2_allowlist" ]] || {
    log "Missing v2 allowlist: ${v2_allowlist}"
    return 1
  }

  stop_7b_endpoints
  start_397b || return 1
  cleanup_run_processes "$Q397_RUN_NAME"
  log "Starting/resuming 397B run: ${Q397_RUN_NAME}"
  local expected
  expected="$(wc -l < "$v2_allowlist" | tr -d ' ')"
  run_command_with_progress_watch "$Q397_RUN_NAME" "$expected" $PY scripts/run_parallel_local_score.py \
    --api-url "$Q397_API_URL" \
    --model "$Q397_MODEL" \
    --run-name "$Q397_RUN_NAME" \
    --workers "$Q397_WORKERS" \
    --base-port "$Q397_BASE_PORT" \
    --port-stride "$Q397_PORT_STRIDE" \
    --verify-mode code \
    --task-allowlist-jsonl "$v2_allowlist" \
    --skill-reminder-interval 0 \
    --max-iterations "$Q397_MAX_ITERATIONS" \
    --temperature "$Q397_TEMPERATURE"
  local rc=$?
  log "397B command exited rc=${rc}; summary follows if available."
  local summary="outputs/runs/${Q397_RUN_NAME}/summary.json"
  if [[ -f "$summary" ]]; then
    python -m json.tool "$summary" | sed -n '1,120p'
    return 0
  fi
  return "$rc"
}

main() {
  log "Log file: ${LOG_FILE}"
  log "Static allowlist: ${ALLOWLIST}"
  local expected
  expected="$(allowlist_count)"
  log "Expected static allowlist tasks: ${expected}"

  for idx in 1 2 3 4 5; do
    local base_port=${SEVEN_B_BASE_PORTS[$((idx - 1))]}
    if ! run_7b_round "$idx" "$base_port" "$expected"; then
      log "7B round ${idx} did not complete cleanly after retry; continuing to next round to avoid wasting unattended time."
    fi
  done

  if ! build_v2_allowlist; then
    log "Cannot continue to 397B without v2 allowlist."
    return 1
  fi

  run_397b
}

main "$@"
