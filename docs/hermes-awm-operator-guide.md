# Hermes AWM Operator Guide

最小手册。目标：让 Hermes 能在 `/data1/jczhong/repos/agent-world-model` 中创建、监控、续跑、杀停 AWM 跑分任务，并给出简明报告。

## 固定路径

```bash
REPO=/data1/jczhong/repos/agent-world-model
DATA=/data1/jczhong/datasets/AgentWorldModel-1K
PY=/data1/jczhong/repos/agent-world-model/.venv/bin/python
RUNS=/data1/jczhong/repos/agent-world-model/outputs/runs
SERVICES=/data1/jczhong/repos/agent-world-model/outputs/services
```

常用脚本：

```bash
$PY $REPO/scripts/run_parallel_local_score.py
$PY $REPO/scripts/run_parallel_api_score.py
$PY $REPO/scripts/monitor_run.py $RUNS/<run_name>
$PY $REPO/scripts/sglang_manage.py status --api-url http://127.0.0.1:<port>/v1 --api-key EMPTY
```

重要输出：

```text
outputs/runs/<run_name>/manifest.json
outputs/runs/<run_name>/summary.json
outputs/runs/<run_name>/<scenario>/task_<id>/trajectory.json
outputs/runs/<run_name>/<scenario>/task_<id>/runner_agent.log
outputs/runs/<run_name>/<scenario>/task_<id>/runner_verify.log
outputs/runs/<run_name>/<scenario>/task_<id>/verify.code.json
```

## 指挥格式

用户通常会这样指挥 Hermes：

```text
目标：启动/监控/续跑/杀停/分析
模型：qwen3.5-397b-a17b / qwen3.6-27b / qwen2.5-7b-instruct / gpt-5.5 / ...
服务：本地 http://127.0.0.1:<port>/v1 或远端 https://www.dmxapi.cn/v1
规模：scenario-limit、task-ids、预期 task 数
并发：workers
端口：base-port、port-stride
run_name：必须明确
特殊要求：是否先杀旧 run、是否保留模型服务、是否 resume、是否只报告
```

Hermes 若信息不全，应采用保守默认值并报告实际参数。涉及删除数据、停模型服务、强杀进程、改代码、提交 git，必须确认。

## 启动本地模型服务

### 8 卡 Qwen397B

```bash
cd $REPO
mkdir -p $SERVICES
tmux new-session -d -s sglang_qwen397b_8gpu '
cd /data1/jczhong/repos/agent-world-model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SGLANG_DISABLE_CUDNN_CHECK=1 \
python -m sglang.launch_server \
  --model-path /data1/models/Qwen/Qwen3.5-397B-A17B \
  --served-model-name qwen3.5-397b-a17b \
  --host 127.0.0.1 \
  --port 8000 \
  --tp 8 \
  --context-length 32768 \
  --mem-fraction-static 0.88 \
  > outputs/services/qwen3.5-397b-a17b-8gpu.log 2>&1
'
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
```

### 27B / 7B 已有常用端口

```text
qwen3.6-27b: http://127.0.0.1:8000/v1
qwen2.5-7b-instruct: http://127.0.0.1:8001/v1
```

检查：

```bash
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null
curl -fsS http://127.0.0.1:8001/v1/models >/dev/null
nvidia-smi
```

## 创建跑分任务

### 本地模型 100-task

```bash
cd $REPO
RUN=awm_test_qwen36_27b_100
tmux new-session -d -s "$RUN" "
cd $REPO
$PY scripts/run_parallel_local_score.py \
  --data $DATA \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen3.6-27b \
  --workers 4 \
  --base-port 23000 \
  --port-stride 100 \
  --scenario-limit 50 \
  --task-ids 0-1 \
  --verify-mode code \
  --max-iterations 30 \
  --max-tokens 4096 \
  --temperature 0.7 \
  --run-name $RUN \
  2>&1 | tee outputs/runs/$RUN.launch.log
"
```

### 远端 DMX 100-task

不要把 API key 写进文档、命令历史或日志。优先让 runner 从环境变量读取 `OPENAI_API_KEY`。

```bash
cd $REPO
test -n "$OPENAI_API_KEY"
RUN=awm_dmx_qwen35_397b_100
tmux new-session -d -s "$RUN" "
cd $REPO
$PY scripts/run_parallel_api_score.py \
  --data $DATA \
  --api-url https://www.dmxapi.cn/v1 \
  --model qwen3.5-397b-a17b \
  --workers 4 \
  --base-port 26000 \
  --port-stride 100 \
  --scenario-limit 50 \
  --task-ids 0-1 \
  --verify-mode code \
  --max-iterations 20 \
  --max-tokens 4096 \
  --temperature 0.6 \
  --run-name $RUN \
  2>&1 | tee outputs/runs/$RUN.launch.log
"
```

如果必须从已有 `awm_api` tmux shell 继承 key，先在该 shell 内执行 `tmux set-environment -g OPENAI_API_KEY "$OPENAI_API_KEY"`，再创建新 session。

### Qwen397B 近官方规模

简单对齐官方规模：`552 env * task 0-5 = 3312 tasks`。

```bash
cd $REPO
RUN=awm_qwen397b_552env_3312tasks
tmux new-session -d -s "$RUN" "
cd $REPO
$PY scripts/run_parallel_local_score.py \
  --data $DATA \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen3.5-397b-a17b \
  --workers 32 \
  --base-port 31000 \
  --port-stride 100 \
  --scenario-limit 552 \
  --task-ids 0-5 \
  --verify-mode code \
  --max-iterations 30 \
  --max-tokens 4096 \
  --temperature 0.7 \
  --run-name $RUN \
  2>&1 | tee outputs/runs/$RUN.launch.log
"
```

## 监控

```bash
cd $REPO
$PY scripts/monitor_run.py outputs/runs/<run_name>
pgrep -af '<run_name>' || true
tail -n 80 outputs/runs/<run_name>.launch.log
```

本地模型还要看：

```bash
curl -fsS http://127.0.0.1:<api_port>/v1/models >/dev/null
nvidia-smi
tail -n 100 outputs/services/<model>.log
```

持续刷新：

```bash
$PY scripts/monitor_run.py outputs/runs/<run_name> --watch --interval 5
```

报告模板：

```text
run_root:
task_dirs / verified / pending:
score: complete / verified
counts:
bad_verify:
runner 是否还活着:
活跃 agent/server 数:
模型服务是否可用:
GPU 是否忙:
异常: agent_failed / verify_failed / no free MCP port / OOM / API error
下一步:
```

注意：`monitor_run.py` 的 `score` 是 `complete / verified`，不是 `complete / 总任务数`。

## LLM Judge 重判

用途：把 pure-code 的 `others` 进一步拆成 `complete / incomplete / server_error / agent_error / judge_error`。这适合分析 500、422、环境瑕疵和模型错误归因。

配置 judge endpoint：

```bash
cd $REPO
export OPENAI_BASE_URL='https://www.dmxapi.cn/v1'
export OPENAI_API_KEY='<不要写进文档或日志>'
export AWM_SYN_OVERRIDE_MODEL='gpt-5.5'
```

`AWM_SYN_OVERRIDE_MODEL` 是 judge 模型，不是被测 agent 模型。Azure endpoint 需额外设置 `AWM_SYN_LLM_PROVIDER=azure`、`AZURE_ENDPOINT_URL`、`AZURE_OPENAI_API_KEY`。

单个 task 重判：

```bash
$REPO/.venv/bin/awm verify \
  --input $RUNS/<run_name>/<scenario>/task_<id> \
  --mode sql \
  --verifier_path $DATA/gen_verifier.jsonl \
  > $RUNS/<run_name>/<scenario>/task_<id>/runner_verify_sql.log 2>&1
```

整 run 批量重判：

```bash
RUN=$RUNS/<run_name>
find "$RUN" -mindepth 2 -maxdepth 2 -type d -name 'task_*' | sort | while read -r task_dir; do
  test -f "$task_dir/trajectory.json" || continue
  test -f "$task_dir/initial.db" || continue
  test -f "$task_dir/final.db" || continue
  $REPO/.venv/bin/awm verify \
    --input "$task_dir" \
    --mode sql \
    --verifier_path "$DATA/gen_verifier.jsonl" \
    > "$task_dir/runner_verify_sql.log" 2>&1
  echo "done: $task_dir"
done
```

结果文件：

```text
verify.sql.json
runner_verify_sql.log
```

关键字段：

```bash
jq '.reward_type, .llm_judge.classification, .llm_judge.judge_result.evidence.error_signals' \
  $RUNS/<run_name>/<scenario>/task_<id>/verify.sql.json
```

注意：LLM judge 不会修复环境，也不会重跑 agent。它只读取已有 `trajectory.json`、`initial.db`、`final.db` 和 SQL verifier 结果。含 500 的样本若 DB 已完成，可能被判 `complete`；若 agent 被环境 5xx 阻塞，通常应判 `server_error`。

## 杀停

只停跑分，保留模型服务：

```bash
RUN=<run_name>
pgrep -af "[a]$RUN"
pgrep -f "[a]$RUN" | xargs -r kill -TERM
sleep 5
pgrep -af "[a]$RUN" || true
$PY scripts/monitor_run.py outputs/runs/$RUN
```

仍有残留且用户明确允许强杀：

```bash
pgrep -f "[a]$RUN" | xargs -r kill -KILL
```

释放显存时才停 SGLang：

```bash
pgrep -af 'sglang.launch_server.*qwen3.5-397b-a17b'
pgrep -f 'sglang.launch_server.*qwen3.5-397b-a17b' | xargs -r kill -TERM
sleep 10
nvidia-smi
```

杀停后报告：杀了哪些 pid、是否保留模型服务、是否还有残留、当前 verified/pending、是否可 resume。

## 续跑

复用同一个 `--run-name`。runner 默认 resume，已有合法 `verify.code.json` 的 task 会跳过。

```bash
cd $REPO
RUN=<run_name>
tmux new-session -d -s "${RUN}_resume" "
cd $REPO
$PY scripts/run_parallel_local_score.py \
  --data $DATA \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen3.5-397b-a17b \
  --workers 40 \
  --base-port 31000 \
  --port-stride 100 \
  --scenario-limit 552 \
  --task-ids 0-5 \
  --verify-mode code \
  --max-iterations 30 \
  --max-tokens 4096 \
  --temperature 0.7 \
  --run-name $RUN \
  --resume \
  2>&1 | tee -a outputs/runs/$RUN.resume.log
"
```

续跑时不要改变模型/API/数据语义，避免结果混杂。可以调整 `workers`。

## 端口规则

每个 worker 使用一个端口段：

```text
worker i: base_port + i * port_stride 到 base_port + i * port_stride + port_stride - 1
```

常用规划：

```text
7B:       21000 或 13000
27B:      23000 或 9100
DMX/API:  25000 起
397B:     31000 起
模型 API: 8000/8001
```

例：`workers=32, base-port=31000, port-stride=100` 使用 `31000-34199`。不要让活跃 run 端口段重叠。

## 快速诊断

失败文件：

```bash
cat outputs/runs/<run>/<scenario>/task_<id>/verify.code.json
tail -n 160 outputs/runs/<run>/<scenario>/task_<id>/runner_agent.log
tail -n 120 outputs/runs/<run>/<scenario>/task_<id>/server.log
tail -n 120 outputs/runs/<run>/<scenario>/task_<id>/runner_verify.log
```

统计非 complete：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/runs/<run_name>")
bad = []
for vf in root.glob("*/*/verify.code.json"):
    x = json.load(open(vf))
    if x.get("reward_type") != "complete":
        bad.append((str(vf.parent.relative_to(root)), x.get("reward_type"), x.get("task")))
print("non_complete", len(bad))
for row in bad[:30]:
    print(row)
PY
```

常见失败类：

```text
tool-call 格式错误
重复 list_tools / 多 tool call 只执行第一个
schema 422 / 参数类型错误
server 500 / 环境实现错误
相对日期基准错误
枚举或字段值不符合 verifier
自然语言完成但 DB 未满足验证
hit max_iterations
agent_failed / verify_failed
no free MCP port
OOM / API error
```

## 执行前检查

```bash
cd $REPO
git diff -- awm/core/agent.py scripts/awm_parallel_runner.py
test -d $DATA
test -f $DATA/gen_tasks.jsonl
pgrep -af '<run_name>' || true
```

本地模型：

```bash
curl -fsS http://127.0.0.1:<api_port>/v1/models >/dev/null
nvidia-smi
```

远端 API：

```bash
test -n "$OPENAI_API_KEY"
```

不要打印 `OPENAI_API_KEY`。

## 安全规则

- 不删除 run、模型、数据集，除非用户明确要求。
- 不默认停 SGLang；释放显存必须用户明确要求。
- 不把 API key 写入文档、commit、命令输出或日志。
- 不默认改官方 prompt/runner/verifier；需要改代码时先说明影响面。
- 汇报分数必须说明口径。
