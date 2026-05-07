#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-/data1/jczhong/datasets/AgentWorldModel-1K}"
MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen/Qwen3.5-397B-A17B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-397b-a17b}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

SCENARIO_LIMIT="${SCENARIO_LIMIT:-2}"
TASK_IDS="${TASK_IDS:-0 1 2}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
AWM_RUN_NAME="${AWM_RUN_NAME:-awm_local_qwen397b_$(date +%Y%m%d_%H%M%S)}"

# Set START_SERVER=1 to launch vLLM from MODEL_PATH before scoring.
START_SERVER="${START_SERVER:-0}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_LOG="${VLLM_LOG:-outputs/runs/${AWM_RUN_NAME}/vllm.log}"

export DATA OPENAI_BASE_URL OPENAI_API_KEY AWM_RUN_NAME

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

wait_for_models_api() {
  local url="${OPENAI_BASE_URL%/}/models"
  for _ in $(seq 1 120); do
    if curl -fsS "$url" -H "Authorization: Bearer $OPENAI_API_KEY" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: timed out waiting for $url" >&2
  return 1
}

if [[ "$START_SERVER" == "1" ]]; then
  mkdir -p "$(dirname "$VLLM_LOG")"
  echo "Starting vLLM server for $MODEL_PATH"
  nohup vllm serve "$MODEL_PATH" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    > "$VLLM_LOG" 2>&1 &
  echo "$!" > "outputs/runs/${AWM_RUN_NAME}/vllm.pid"
  wait_for_models_api
else
  echo "START_SERVER=0, assuming model server is already available at $OPENAI_BASE_URL"
  wait_for_models_api
fi

echo "DATA=$DATA"
echo "MODEL_PATH=$MODEL_PATH"
echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
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
      --model "$SERVED_MODEL_NAME" \
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
