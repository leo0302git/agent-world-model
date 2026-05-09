#!/usr/bin/env bash
# External watchdog for the unattended allowlist v2 pipeline.
# It is intentionally independent from run_allowlist_v2_unattended.sh so it can
# supervise a pipeline process that was launched before the latest script edits.

set -uo pipefail

REPO=${REPO:-/data1/jczhong/repos/agent-world-model}
cd "$REPO" || exit 1

SESSION=${SESSION:-allowlist_v2_pipeline}
SCRIPT=${SCRIPT:-scripts/run_allowlist_v2_unattended.sh}
RUN_PREFIX=${RUN_PREFIX:-awm_qwen25_7b_static8314_runtime_w256p64}
Q397_RUN_NAME=${Q397_RUN_NAME:-awm_qwen397b_v2allowlist_runtime_w48_v1}
POLL_SECONDS=${POLL_SECONDS:-300}
STALL_TIMEOUT_SECONDS=${STALL_TIMEOUT_SECONDS:-3600}
LOG_DIR=${LOG_DIR:-outputs/logs/unattended_allowlist_v2}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/watchdog_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

run_root() {
  printf 'outputs/runs/%s\n' "$1"
}

verified_count() {
  local root=$1
  if [[ ! -d "$root" ]]; then
    echo 0
    return
  fi
  find "$root" -name verify.code.json | wc -l | tr -d ' '
}

active_run_name() {
  local line
  line="$(pgrep -af "[r]un_parallel_local_score.py .*--run-name" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo ""
    return
  fi
  sed -n 's/.*--run-name \([^ ]*\).*/\1/p' <<<"$line"
}

session_alive() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
}

restart_pipeline() {
  log "Restarting pipeline session ${SESSION}"
  tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
  sleep 2
  tmux new-session -d -s "$SESSION" "cd '$REPO' && '$SCRIPT'"
}

cleanup_run() {
  local run_name=$1
  log "Cleaning stuck run processes: ${run_name}"
  local pids
  pids="$(pgrep -f "$run_name" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -TERM $pids 2>/dev/null || true
    sleep 15
  fi
  pids="$(pgrep -f "$run_name" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -KILL $pids 2>/dev/null || true
    sleep 5
  fi
}

main() {
  log "Watchdog log: ${LOG_FILE}"
  local last_run=""
  local last_count=0
  local last_progress_ts
  last_progress_ts="$(date +%s)"

  while true; do
    if ! session_alive; then
      log "Pipeline session is missing."
      restart_pipeline
      last_run=""
      last_count=0
      last_progress_ts="$(date +%s)"
      sleep "$POLL_SECONDS"
      continue
    fi

    local run_name
    run_name="$(active_run_name)"
    if [[ -z "$run_name" ]]; then
      log "No active scoring run; pipeline session is alive."
      sleep "$POLL_SECONDS"
      continue
    fi

    local root count now idle
    root="$(run_root "$run_name")"
    count="$(verified_count "$root")"
    now="$(date +%s)"

    if [[ "$run_name" != "$last_run" || "$count" != "$last_count" ]]; then
      last_run="$run_name"
      last_count="$count"
      last_progress_ts="$now"
      idle=0
    else
      idle=$((now - last_progress_ts))
    fi

    log "active_run=${run_name} verify_files=${count} idle_seconds=${idle}"
    if (( idle >= STALL_TIMEOUT_SECONDS )); then
      log "Detected stalled active run."
      cleanup_run "$run_name"
      restart_pipeline
      last_run=""
      last_count=0
      last_progress_ts="$(date +%s)"
    fi
    sleep "$POLL_SECONDS"
  done
}

main "$@"
