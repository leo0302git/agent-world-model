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
TASK_IDS="${TASK_IDS:-0-1}"
WORKERS="${WORKERS:-2}"
BASE_PORT="${BASE_PORT:-31000}"
PORT_STRIDE="${PORT_STRIDE:-100}"
VERIFY_MODE="${VERIFY_MODE:-code}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
AWM_RUN_NAME="${AWM_RUN_NAME:-awm_local_qwen397b_$(date +%Y%m%d_%H%M%S)}"
JUDGE_API_URL="${JUDGE_API_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-}"

# Set START_SERVER=1 to launch SGLang from MODEL_PATH before scoring.
# The server is intentionally left running after scoring so GPU memory stays resident
# and can be reused by follow-up runs.
START_SERVER="${START_SERVER:-0}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-8}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
SGLANG_WATCHDOG_TIMEOUT="${SGLANG_WATCHDOG_TIMEOUT:-3600}"
SGLANG_LOG="${SGLANG_LOG:-outputs/services/qwen3.5-397b-a17b-tp8.log}"
SGLANG_SESSION="${SGLANG_SESSION:-sglang_qwen397b_8gpu}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/usr/bin/python}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-360}"

export DATA OPENAI_BASE_URL OPENAI_API_KEY AWM_RUN_NAME
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

wait_for_models_api() {
  local url="${OPENAI_BASE_URL%/}/models"
  for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
    if curl -fsS "$url" -H "Authorization: Bearer $OPENAI_API_KEY" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: timed out waiting for $url" >&2
  return 1
}

if [[ "$START_SERVER" == "1" ]]; then
  mkdir -p "$(dirname "$SGLANG_LOG")"
  echo "Starting SGLang server for $MODEL_PATH in tmux session $SGLANG_SESSION"
  if tmux has-session -t "$SGLANG_SESSION" 2>/dev/null; then
    echo "ERROR: tmux session already exists: $SGLANG_SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$SGLANG_SESSION" "
cd '$REPO_ROOT'
CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}' \\
SGLANG_DISABLE_CUDNN_CHECK=1 \\
'$SGLANG_PYTHON' -m sglang.launch_server \\
  --model-path '$MODEL_PATH' \\
  --served-model-name '$SERVED_MODEL_NAME' \\
  --host '$SGLANG_HOST' \\
  --port '$SGLANG_PORT' \\
  --tp-size '$SGLANG_TP_SIZE' \\
  --mem-fraction-static '$SGLANG_MEM_FRACTION_STATIC' \\
  --context-length '$SGLANG_CONTEXT_LENGTH' \\
  --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder \\
  --watchdog-timeout '$SGLANG_WATCHDOG_TIMEOUT' \\
  --model-loader-extra-config '{\"enable_multithread_load\": true}' \\
  > '$SGLANG_LOG' 2>&1
"
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
echo "WORKERS=$WORKERS"
echo "VERIFY_MODE=$VERIFY_MODE"
echo "JUDGE_API_URL=${JUDGE_API_URL:-<unset>}"
echo "JUDGE_MODEL=${JUDGE_MODEL:-<unset>}"

judge_args=()
if [[ -n "$JUDGE_API_URL" ]]; then
  judge_args+=(--judge-api-url "$JUDGE_API_URL")
fi
if [[ -n "$JUDGE_API_KEY" ]]; then
  judge_args+=(--judge-api-key "$JUDGE_API_KEY")
fi
if [[ -n "$JUDGE_MODEL" ]]; then
  judge_args+=(--judge-model "$JUDGE_MODEL")
fi
if [[ -n "$JUDGE_PROVIDER" ]]; then
  judge_args+=(--judge-provider "$JUDGE_PROVIDER")
fi

"$PYTHON" scripts/run_parallel_local_score.py \
  --data "$DATA" \
  --api-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$SERVED_MODEL_NAME" \
  --workers "$WORKERS" \
  --base-port "$BASE_PORT" \
  --port-stride "$PORT_STRIDE" \
  --scenario-limit "$SCENARIO_LIMIT" \
  --task-ids "$TASK_IDS" \
  --verify-mode "$VERIFY_MODE" \
  --max-iterations "$MAX_ITERATIONS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --run-name "$AWM_RUN_NAME" \
  "${judge_args[@]}"

echo "Run root: outputs/runs/$AWM_RUN_NAME"
if [[ -f "outputs/runs/$AWM_RUN_NAME/summary.json" ]]; then
  jq . "outputs/runs/$AWM_RUN_NAME/summary.json"
fi
