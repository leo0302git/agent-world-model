#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run Qwen397B on the AWM 552-env / 3312-task arena with system-prompt skill injection and code verifier.

Defaults:
  AWM_RUN_NAME=awm_qwen397b_552env_3312tasks_code_skill_v0
  SCENARIO_LIMIT=552
  TASK_IDS=0-5
  WORKERS=32
  BASE_PORT=35000
  PORT_STRIDE=200
  VERIFY_MODE=code
  SKILL_DIR=outputs/arena3312_skill_v0/skills_qwen397b_v0/by_scenario

Usage:
  scripts/run_qwen397b_skill_code_arena3312.sh

Override any setting with an environment variable, for example:
  WORKERS=32 BASE_PORT=45000 scripts/run_qwen397b_skill_code_arena3312.sh
EOF
  exit 0
fi

# Code-verifier run aligned to:
# outputs/runs/awm_qwen397b_552env_3312tasks_sqljudge
#
# Assumes the Qwen397B OpenAI-compatible server is already resident at
# OPENAI_BASE_URL. Set START_SERVER=1 only if you explicitly want the wrapper
# to launch SGLang first.

export START_SERVER="${START_SERVER:-0}"
export AWM_RUN_NAME="${AWM_RUN_NAME:-awm_qwen397b_552env_3312tasks_code_skill_v0}"
export SCENARIO_LIMIT="${SCENARIO_LIMIT:-552}"
export TASK_IDS="${TASK_IDS:-0-5}"
export WORKERS="${WORKERS:-48}"
export BASE_PORT="${BASE_PORT:-35000}"
export PORT_STRIDE="${PORT_STRIDE:-200}"
export VERIFY_MODE="${VERIFY_MODE:-code}"
export TEMPERATURE="${TEMPERATURE:-0.6}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export SKILL_DIR="${SKILL_DIR:-$REPO_ROOT/outputs/arena3312_skill_v0/skills_qwen397b_v0/by_scenario}"

exec scripts/run_local_qwen397b_score.sh
