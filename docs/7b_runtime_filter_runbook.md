# 7B Runtime Filter 并发启动经验

本文记录在 `static_code_verify/allowlist.jsonl` 通过的任务上，用常驻 7B 模型做 task runtime 可用性筛选时的启动经验。

## 目标

静态筛选只排除 verifier/DB 明显坏的 task。7B runtime run 用来进一步观察真实轨迹：

- 工具是否稳定：`call_tool` 是否 500、timeout、schema/route 错。
- task 是否可操作：任务实体是否存在，工具返回是否非空，是否有合理工具路径。
- 多次重复后，只有每次都暴露硬问题的 task 才进入 reject。

## Proxy 必须处理

本地 OpenAI-compatible endpoint 是 `127.0.0.1`/`localhost`，agent 子进程不能继承外部 SOCKS proxy。

否则 `httpx`/OpenAI client 会读到 `HTTP_PROXY`、`HTTPS_PROXY` 或 `ALL_PROXY`，在 venv 没有安装 `socksio` 时直接失败：

```text
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

经验：

- 对本地 endpoint，runner 应在子进程环境里清掉 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 及小写版本。
- 同时设置 `NO_PROXY=127.0.0.1,localhost,::1`。
- 这类失败不是 task 无效，不能进入 task reject 统计。

当前 `scripts/awm_parallel_runner.py` 已加入 `disable_proxy_for_local_api()` 来处理本地 API。

## Endpoint 与 Worker

当前 7B 服务是 8 个 endpoint：

```text
http://127.0.0.1:8100/v1
http://127.0.0.1:8101/v1
http://127.0.0.1:8102/v1
http://127.0.0.1:8103/v1
http://127.0.0.1:8104/v1
http://127.0.0.1:8105/v1
http://127.0.0.1:8106/v1
http://127.0.0.1:8107/v1
```

低并发如 `workers=8` 只能保证每个 endpoint 一个 agent。AWM task 不是纯推理压测，每个 task 会做 DB reset、MCP server 启动、工具调用和 verifier，模型请求中间有大量 CPU/IO/HTTP 等待。因此 `workers=8` 通常不能打满 GPU。

经验：

- 7B 可以用 8 卡、8 endpoint、多 worker 并发把计算效率拉满。
- 如果每张卡开 32 并发，总 worker 数应设为 `8 * 32 = 256`。
- runner 会按 `worker_id % len(api_urls)` 分配 endpoint，因此 256 workers 会平均分到 8 个 endpoint，每个 endpoint 约 32 个 worker。

推荐配置：

```bash
--workers 256
--api-url http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1,http://127.0.0.1:8103/v1,http://127.0.0.1:8104/v1,http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1,http://127.0.0.1:8107/v1
```

## Tool Help 与 Skill 注入

当前 agent 的工具发现协议已经改成更省上下文的版本：

- `list_tools` 只返回可用 tool 的名字和 description，不再返回 parameter 和 example response。
- `tool_help` 是新的 meta MCP tool，输入 tool 名，返回单个 tool 的 parameter、input schema 和 example response。
- system prompt 会要求模型先用 `list_tools`，并在不确定参数时先查 `tool_help`，而不是凭记忆猜参数。
- `call_tool` 仍然负责真实工具调用。

skill 注入也改成两段式：

- 初始 system prompt 后立刻插入 scenario skill。
- 长轨迹中每隔 `--skill-reminder-interval` 次工具循环重新插入一次 skill reminder，默认值是 `4`。
- 如果只是做 task runtime 可用性筛选，不想让 skill 影响轨迹，可以不传 `--skill-dir`，并把 `--skill-reminder-interval` 设成 `0`。

轨迹里会记录这些字段，方便判断协议是否真的生效：

```text
list_tools_format
list_tools_calls
tool_help_calls
helped_tools
tool_calls_without_prior_help
skill_injection_position
skill_reminder_interval
skill_reminder_count
skill_injected_iterations
```

10 task smoke 命令，用来确认本地 7B endpoint、极简 `list_tools`、`tool_help` 和 skill reminder 都能跑通：

```bash
uv run python scripts/run_parallel_local_score.py \
  --api-url http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1,http://127.0.0.1:8103/v1,http://127.0.0.1:8104/v1,http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1,http://127.0.0.1:8107/v1 \
  --model qwen2.5-7b-instruct \
  --run-name awm_qwen25_7b_10_toolhelp_skill_smoke \
  --scenario-limit 5 \
  --task-ids 0-1 \
  --workers 8 \
  --base-port 42000 \
  --port-stride 100 \
  --verify-mode code \
  --skill-dir outputs/arena3312_skill_v1/skills_qwen397b_v3_tooldb_500task/by_scenario \
  --skill-reminder-interval 4 \
  --max-iterations 30 \
  --temperature 0.6
```

检查协议统计：

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("outputs/runs/awm_qwen25_7b_10_toolhelp_skill_smoke")
totals = {
    "trajectories": 0,
    "list_tools_calls": 0,
    "tool_help_calls": 0,
    "skill_reminder_count": 0,
    "tool_calls_without_prior_help": 0,
}
for path in run.glob("*/*/trajectory.json"):
    data = json.loads(path.read_text())
    totals["trajectories"] += 1
    for key in totals:
        if key == "trajectories":
            continue
        value = data.get(key, 0)
        totals[key] += len(value) if isinstance(value, list) else int(value or 0)
print(json.dumps(totals, ensure_ascii=False, indent=2))
PY
```

如果机器上已经有 256-worker runtime run 占满 7B endpoint，这个 smoke 只能用于功能检查，吞吐和失败率都不代表真实能力。正式比较 skill 是否有效时，应单独跑干净的 no-skill 与 skill run。

## MCP Port 分配

每个 running task 会启动一个 MCP server，需要独占本地端口。

runner 的端口分配规则是：

```text
worker_port_start = base_port + worker_id * port_stride
可用范围 = [worker_port_start, worker_port_start + port_stride)
```

重要经验：`workers=256` 时不能把 `port_stride` 设成 `1`。

原因是 task 数可能大于 worker 数。例如 300 task、256 workers 时，`global_idx=0` 和 `global_idx=256` 都会分配到 `worker_id=0`。ThreadPoolExecutor 不保证同一个 `worker_id` 的前一个 task 已结束后，下一个同 `worker_id` task 才开始。如果 `port_stride=1`，两个 task 会争同一个端口，出现：

```text
no free MCP port in worker 0 range 48000-48000
```

这类 `runner exception` 是端口配置问题，不是 task 无效。

推荐：

- 对 `workers=256`，`port_stride` 至少用 `4`，更稳用 `8` 或 `16`。
- 端口上限不能超过 `65535`。
- 可用最大端口大约是：

```text
base_port + workers * port_stride + port_stride
```

可选安全配置：

```text
workers=256
base_port=48000
port_stride=16
端口范围约 48000-52095
```

启动前最好预检查端口段是否空闲。发现 `47000` 段有占用时，不要硬用，换到空闲段。

## 300 Task 三重复推荐命令

先固定 300 task 子集：

```bash
python - <<'PY'
from pathlib import Path
src = Path("outputs/task_allowlists/static_code_verify/allowlist.jsonl")
out = Path("outputs/task_allowlists/static_code_verify/allowlist_300.jsonl")
with src.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
    for i, line in enumerate(f):
        if i >= 300:
            break
        g.write(line)
PY
```

单轮 run 模板：

```bash
uv run python scripts/run_parallel_local_score.py \
  --api-url http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1,http://127.0.0.1:8103/v1,http://127.0.0.1:8104/v1,http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1,http://127.0.0.1:8107/v1 \
  --model qwen2.5-7b-instruct \
  --run-name awm_qwen25_7b_static300_runtime_w256_v1 \
  --workers 256 \
  --base-port 48000 \
  --port-stride 16 \
  --verify-mode code \
  --task-allowlist-jsonl outputs/task_allowlists/static_code_verify/allowlist_300.jsonl \
  --skill-reminder-interval 0 \
  --max-iterations 30 \
  --temperature 0.6
```

上面是不注入 skill 的 runtime filter 模板。若要收集带 skill 影响的轨迹，把 `--skill-reminder-interval 0` 改为 `4`，并加上：

```bash
--skill-dir outputs/arena3312_skill_v1/skills_qwen397b_v3_tooldb_500task/by_scenario
```

三轮建议使用不同 run name 和不同 base port，避免旧 MCP server 残留造成假冲突：

```text
v1: base_port=48000
v2: base_port=53000
v3: base_port=58000
```

如果上一个 run 异常中断，先清理对应 run name 的进程，再启动下一轮。

## 判定策略

三轮 runtime 后再分析轨迹：

```bash
python scripts/analyze_run_trajectories.py \
  outputs/runs/awm_qwen25_7b_static300_runtime_w256_v1 \
  outputs/runs/awm_qwen25_7b_static300_runtime_w256_v2 \
  outputs/runs/awm_qwen25_7b_static300_runtime_w256_v3 \
  --allowlist-jsonl outputs/task_allowlists/static_code_verify/allowlist_300.jsonl \
  --out-dir outputs/task_allowlists/runtime_7b_static300_w256 \
  --mode code
```

最终原则：

- 三轮都出现同类硬问题，才 hard reject。
- 单轮 7B 输出格式错误、没完成、没找到工具路径，只能进入 review，不能直接说明 task 无效。
- `runner exception`、proxy、端口冲突、agent 启动失败属于实验配置问题，应修配置重跑，不应算 task 无效。

## 计时

`scripts/awm_parallel_runner.py` 的 `summary.json` 已记录：

- `started_at`
- `finished_at`
- `elapsed_seconds`
- `elapsed`

比较吞吐时应使用实际完成 verify 的 task 数：

```text
每 100 task 耗时 = elapsed_seconds / total_verified * 100
```

其中 `total_verified` 对应 `summary.json` 里的 `total`，不是 manifest 里的 task 总数。若有 runner exception 或 agent_failed，要单独报告，不能混入有效 task 吞吐。
