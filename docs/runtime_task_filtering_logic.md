# Runtime Task 清洗判定逻辑

本文记录在 `static_code_verify/allowlist.jsonl` 通过的 task 上，用 7B 三轮 runtime run 进一步筛无效 task 的逻辑，便于后续复盘。

## 输入与产物

实验输入：

- 静态初筛 300 task：
  - `outputs/task_allowlists/static_code_verify/allowlist_300.jsonl`
- 三轮 7B runtime run：
  - `outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v1`
  - `outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v2`
  - `outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v3`

strict 清洗产物：

- `outputs/task_allowlists/runtime_7b_static300_w256p16_3runs_strict/keep.jsonl`
- `outputs/task_allowlists/runtime_7b_static300_w256p16_3runs_strict/review.jsonl`
- `outputs/task_allowlists/runtime_7b_static300_w256p16_3runs_strict/rejected.jsonl`
- `outputs/task_allowlists/runtime_7b_static300_w256p16_3runs_strict/task_report.jsonl`
- `outputs/task_allowlists/runtime_7b_static300_w256p16_3runs_strict/per_run_report.jsonl`
- `outputs/task_allowlists/runtime_7b_static300_w256p16_3runs_strict/stats.json`

生成命令：

```bash
python scripts/analyze_run_trajectories.py \
  outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v1 \
  outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v2 \
  outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v3 \
  --allowlist-jsonl outputs/task_allowlists/static_code_verify/allowlist_300.jsonl \
  --out-dir outputs/task_allowlists/runtime_7b_static300_w256p16_3runs_strict \
  --mode code \
  --strict-all-hard-reject
```

## Strict Reject 原则

一个 task 只有满足以下条件才进入 `rejected.jsonl`：

1. 三轮里没有任何一次 `reward_type == complete`。
2. 三轮都有可分析轨迹。
3. v1/v2/v3 每一轮都出现 hard reason。

也就是：

```text
任意一轮 complete -> keep
三轮都 hard -> reject
其他情况 -> review
```

这个策略的目的不是筛“7B 能否完成”，而是筛“task/env/tool 是否反复表现出不可用”。单轮失败、两轮失败、7B 输出格式问题、某轮缺失轨迹，都不直接 hard reject。

## Hard Reason

### http_500

含义：业务 tool 调用返回 `500` 或 `Internal Server Error`。

这更像工具服务、后端实现、路由处理或环境状态本身有问题。单轮 500 可能是偶然问题，strict 模式要求三轮都有 hard reason 才拒绝。

示例：

```text
accounting_1/task_0
```

任务：

```text
Create a new customer named 'Acme Design Studio' with billing email
'billing@acmedesign.com', payment_terms of 'Net 30', and set their default
tax_category to 'Standard Sales Tax 8.5%'.
```

三轮现象：

```text
v1: call_tools=6, errors={'http_500': 4}
v2: call_tools=4, errors={'http_500': 2}
v3: call_tools=4, errors={'http_500': 2}
```

证据：

```text
tool: mcp_server_create_customer
response: Error calling create_customer. Status code: 500. Response: Internal Server Error
```

判断：同一个创建 customer 的工具三轮都 500，更像 tool/backend 问题，不是 7B 一次偶然失败。

### high_tool_error_rate

含义：某一轮中，非 agent 参数格式类错误占 `call_tool` 的比例超过阈值，默认 `>= 30%`。

注意：`agent_argument_parse_error` 不计入错误率。7B 经常把 `call_tool.arguments` 的 JSON 写坏，这类是 agent 格式问题，不应作为 task/tool 无效证据。

示例同 `accounting_1/task_0`：

```text
v1: 4/6 call_tool 是 http_500
v2: 2/4 call_tool 是 http_500
v3: 2/4 call_tool 是 http_500
```

判断：三轮工具错误率都很高，且错误不是 agent JSON 格式问题，因此进入 hard reject。

### repeated_schema_or_route_error

含义：同一轮里出现重复 schema/route 类错误，例如 validation、required field、404、unknown route、参数/路由不一致等。

示例：

```text
accounting_2/task_3
```

任务：

```text
Create a recurring monthly expense of $89 for the vendor
'Zoom Video Communications' categorized as 'Software & Subscriptions'
starting on January 1, 2025, set to recur on the first of every month
with no end date, and paid from the 'Business Credit Card' account.
```

三轮现象：

```text
v1: errors={'http_500': 2, 'validation': 1}
v2: errors={'http_500': 2, 'validation': 1}
v3: errors={'http_500': 1, 'validation': 3}
```

证据：

```text
tool: mcp_server_create_recurring_expense
response: Error: Input validation error: None is not of type 'string'

tool: mcp_server_create_recurring_expense
response: Error calling create_recurring_expense. Status code: 500. Response: Internal Server Error
```

判断：同一个业务工具多轮出现 validation/500 混合问题，更像 schema 暴露、路由实现或工具可用性问题。

### only_empty_or_not_found_tool_responses

含义：一轮中 agent 调用了工具，但所有工具结果都为空、not found、does not exist、empty 等，没有任何非空有效返回。

示例：

```text
application_management_1/task_9
```

任务：

```text
As a quality assurance manager, list all applications in the 'MBA Online'
program at Metro Business School for the Spring 2025 cycle where at least
one required checklist item is missing or where any uploaded document has
a verification state of 'rejected', and return their applicant IDs and
missing or invalid checklist items.
```

三轮现象：

```text
v1: call_tools=1, empty=1, non_empty=0
v2: call_tools=1, empty=1, non_empty=0
v3: call_tools=1, empty=1, non_empty=0
```

证据：

```text
tool: mcp_server_report_qa_missing_or_rejected_checklist
response:
{
  "applications": []
}
```

判断：三轮都调用了非常贴合任务的 report 工具，但都返回空列表，说明任务要求的数据很可能不存在或不可达。

### timeout

含义：tool 调用 timeout。

这次 strict `rejected.jsonl` 里没有 timeout 类 hard reject。

复盘原则：

- 单次 timeout 不应直接 reject，可能是机器负载或并发抖动。
- 三轮都 timeout 或三轮都有其他 hard reason 时，才可考虑 hard reject。

### no_call_tool_path

含义：agent 没有真正调用业务 tool。

这次 strict `rejected.jsonl` 里没有 `no_call_tool_path` 类 hard reject。

复盘原则：

- `no_call_tool_path` 很可能是 7B 能力不足或输出不稳定，不说明 task 本身无效。
- 只有三轮都没有工具路径，且没有任何 complete 时，才作为 hard reject 候选。

### missing_or_bad_trajectory

含义：轨迹文件缺失或损坏。

这次 strict `rejected.jsonl` 里没有此类 hard reject。

类似情况有 `missing_run_records`，但都进入 review，例如：

```text
accounting_finance_1/task_8 -> review, missing_run_records:1
asset_management_2/task_4 -> review, missing_run_records:1
asset_management_4/task_2 -> review, missing_run_records:1
```

复盘原则：

- 轨迹缺失、runner exception、agent_failed、proxy、端口冲突都是实验执行问题。
- 这些问题应修配置重跑，不应直接判断 task 本身无效。

## 非 Hard Reason

### agent_argument_parse_error

含义：7B 把 `call_tool.arguments` 的 JSON 写坏，导致 validation error。

常见现象：

```text
Error: Input validation error: 'city' is a required property
```

如果底层原因是内层 JSON 没正确转义，脚本会归为 `agent_argument_parse_error`。

复盘原则：

- 这是 agent 输出格式问题，不是 task/tool 必坏。
- 不计入 `high_tool_error_rate`。
- 不单独触发 hard reject。

### never_completed

默认没有启用 `--require-complete`。

复盘原则：

- 当前目标是筛 task/env/tool 是否有效，不是筛 7B 能不能完成。
- 7B 三轮没 complete 不能直接证明 task 无效。
- 没 complete 但没有三轮 hard reason 的 task 进入 review。

## 本次 Strict 结果

```text
tasks_analyzed: 300
keep: 84
review: 160
reject: 56
allowlist: 244
```

strict reject 的 hard reason 统计：

```text
high_tool_error_rate: 48
http_500: 35
repeated_schema_or_route_error: 25
only_empty_or_not_found_tool_responses: 8
```

其中 `allowlist.jsonl` 是 `keep + review`，可作为后续更强 agent 或 397B 实验输入。`rejected.jsonl` 是当前根据三轮 7B runtime 证据建议排除的 task。

## 复查建议

复查某个被 reject 的 task 时，优先看：

1. `task_report.jsonl` 中该 task 的 `runs[*].analysis.evidence`。
2. 对应 run 目录下的 `trajectory.json`。
3. 对应 run 目录下的 `runner_agent.log`。
4. 如果是 500 或 schema/route 错，进一步看该 scenario 生成的 MCP server/API 实现。

不要只看 `reward_type=others`，因为 7B 没完成不等于 task 无效。重点看三轮是否重复暴露工具、数据或环境问题。
