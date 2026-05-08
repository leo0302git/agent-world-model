#!/usr/bin/env bash
set -euo pipefail

MAIN_REPO="/data1/jczhong/repos/agent-world-model"
SYSTEM_REPO="/data1/jczhong/repos/agent-world-model-skill-system"
SKILL_DIR="$MAIN_REPO/outputs/arena3312_skill_v0/skills_qwen397b_v0/by_scenario"

AFTER_RUN_NAME="${AFTER_RUN_NAME:-awm_qwen397b_552env_3312tasks_code_skill_after_listtools_v1}"
SYSTEM_RUN_NAME="${SYSTEM_RUN_NAME:-awm_qwen397b_552env_3312tasks_code_skill_system_v1}"

COMMON_ENV=(
  START_SERVER=0
  SCENARIO_LIMIT=552
  TASK_IDS=0-5
  WORKERS=48
  PORT_STRIDE=200
  VERIFY_MODE=code
  TEMPERATURE=0.6
  MAX_ITERATIONS=30
  MAX_TOKENS=4096
  SKILL_DIR="$SKILL_DIR"
  JUDGE_API_URL=
  JUDGE_API_KEY=
  JUDGE_MODEL=
  JUDGE_PROVIDER=
)

echo "===== $(date -Is) after-listtools run ====="
cd "$MAIN_REPO"
git rev-parse --short HEAD
env "${COMMON_ENV[@]}" \
  AWM_RUN_NAME="$AFTER_RUN_NAME" \
  BASE_PORT=35000 \
  scripts/run_qwen397b_skill_code_arena3312.sh

echo "===== $(date -Is) preparing system-injection worktree ====="
cd "$MAIN_REPO"
if [[ ! -d "$SYSTEM_REPO/.git" && ! -f "$SYSTEM_REPO/.git" ]]; then
  git worktree add "$SYSTEM_REPO" 07b394a
fi
if [[ ! -e "$SYSTEM_REPO/.venv" ]]; then
  ln -s "$MAIN_REPO/.venv" "$SYSTEM_REPO/.venv"
fi

echo "===== $(date -Is) system-prompt run ====="
cd "$SYSTEM_REPO"
git rev-parse --short HEAD
env "${COMMON_ENV[@]}" \
  AWM_RUN_NAME="$SYSTEM_RUN_NAME" \
  BASE_PORT=45000 \
  PYTHON="$MAIN_REPO/.venv/bin/python" \
  scripts/run_qwen397b_skill_code_arena3312.sh

echo "===== $(date -Is) done ====="
