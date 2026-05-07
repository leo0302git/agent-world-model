#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/data1/jczhong/datasets/AgentWorldModel-1K}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://newapi2.frontis.top/v1}"
MODEL="${MODEL:-qwen3.5-397b-a17b}"
SCENARIO_LIMIT="${SCENARIO_LIMIT:-2}"
TASK_IDS="${TASK_IDS:-0 1 2}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
AWM_RUN_NAME="${AWM_RUN_NAME:-awm_qwen397b_$(date +%Y%m%d_%H%M%S)}"

export DATA OPENAI_BASE_URL MODEL AWM_RUN_NAME

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

echo "DATA=$DATA"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "MODEL=$MODEL"
echo "AWM_RUN_NAME=$AWM_RUN_NAME"
echo "SCENARIO_LIMIT=$SCENARIO_LIMIT"
echo "TASK_IDS=$TASK_IDS"

jq -r '.scenario' "$DATA/gen_tasks.jsonl" | head -n "$SCENARIO_LIMIT" | while read -r scenario; do
  for task_id in $TASK_IDS; do
    echo "Running scenario=$scenario task_id=$task_id"

    uv run awm agent \
      --scenario "$scenario" \
      --task_id "$task_id" \
      --envs_path "$DATA/gen_envs.jsonl" \
      --tasks_path "$DATA/gen_tasks.jsonl" \
      --db_path "$DATA/gen_db.jsonl" \
      --sample_path "$DATA/gen_sample.jsonl" \
      --api_url "$OPENAI_BASE_URL" \
      --model "$MODEL" \
      --max_iterations "$MAX_ITERATIONS" \
      --temperature "$TEMPERATURE" \
      --max_tokens "$MAX_TOKENS"

    uv run awm verify \
      --input "outputs/runs/$AWM_RUN_NAME/$scenario/task_$task_id" \
      --mode code \
      --verifier_code_path "$DATA/gen_verifier.pure_code.jsonl"
  done
done

echo "Run root: outputs/runs/$AWM_RUN_NAME"
if [[ -f "outputs/runs/$AWM_RUN_NAME/summary.json" ]]; then
  jq . "outputs/runs/$AWM_RUN_NAME/summary.json"
fi
