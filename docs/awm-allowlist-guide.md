# AWM Allowlist Guide

本文记录当前 AWM task allowlist 的几版口径。后续跑分、训练、skill 蒸馏或课程学习时，应按目标选择不同 allowlist。

统计时间：2026-05-09。

## Overview

| Name | Path | Tasks | Scenarios | 主要用途 |
| --- | --- | ---: | ---: | --- |
| static_code_verify | `outputs/task_allowlists/static_code_verify/allowlist.jsonl` | 8314 | 1000 | 最大覆盖的静态初筛 |
| 7B runtime v2 | `outputs/task_allowlists/runtime_7b_static8314_w256_5runs_v2/allowlist.jsonl` | 7204 | 1000 | 保守 runtime validity arena |
| 397B runtime v3 | `outputs/task_allowlists/runtime_qwen397b_v2allowlist_w48_v1_v3/allowlist.jsonl` | 6465 | 995 | 当前推荐主 arena |
| 7B mixed v4 | `outputs/task_allowlists/runtime_7b_5runs_has_complete_and_other_v4/allowlist.jsonl` | 1545 | 764 | 7B skill2lora 初期训练/验证 |

另有一个常用辅助子集：

| Name | Path | Tasks | Scenarios | 主要用途 |
| --- | --- | ---: | ---: | --- |
| 397B complete subset | `outputs/task_allowlists/runtime_qwen397b_v2allowlist_w48_v1_v3/keep.jsonl` | 3337 | 953 | 高可信成功轨迹、skill 蒸馏、SFT 数据 |

## static_code_verify

路径：

```text
outputs/task_allowlists/static_code_verify/allowlist.jsonl
```

规模：

```text
8314 tasks
1000 scenarios
```

口径：

只做静态和 verifier 层面的 sanity check。保留满足以下条件的 task：

- env / task / DB / sample 文件存在。
- DB reset 正常。
- verifier 文件存在且可 compile。
- noop verifier 执行成功。

这个 allowlist 只能说明 task 的基础文件和 code verifier 不明显坏，不能说明真实 MCP tool runtime 可用。

适用：

- 最大覆盖分析。
- 新筛选 pipeline 的输入。
- 不建议直接作为最终跑分 arena。

## 7B Runtime V2

路径：

```text
outputs/task_allowlists/runtime_7b_static8314_w256_5runs_v2/allowlist.jsonl
```

规模：

```text
7204 tasks
1000 scenarios
```

来源：

在 `static_code_verify/allowlist.jsonl` 的 8314 条上，用 Qwen2.5-7B 跑 5 轮 runtime：

```text
outputs/runs/legacy/qwen7b_static8314_w256p64_v1
outputs/runs/legacy/qwen7b_static8314_w256p64_v2
outputs/runs/legacy/qwen7b_static8314_w256p64_v3
outputs/runs/legacy/qwen7b_static8314_w256p64_v4
outputs/runs/legacy/qwen7b_static8314_w256p64_v5
```

这些 run 已按 legacy 精简规则归档：每个 run 保留根目录 summary 类文件和 3 个完整 task 目录，其它 task 目录已删除。原始目录名分别为 `awm_qwen25_7b_static8314_runtime_w256p64_v1` 到 `v5`。

口径：

```text
任意一轮 complete -> keep
五轮都有 hard runtime issue -> reject
其他 -> review
allowlist = keep + review + missing safeguard
```

hard runtime issue 包括：

- `http_500`
- `timeout`
- `high_tool_error_rate`
- repeated schema / route / validation error
- only empty / not found tool responses
- no valid `call_tool` path

适用：

- 保守 runtime validity 过滤。
- 比 static 更干净，但仍保留大量 7B 没完成的 review task。
- 可作为强模型进一步筛选的输入。

## 397B Runtime V3

路径：

```text
outputs/task_allowlists/runtime_qwen397b_v2allowlist_w48_v1_v3/allowlist.jsonl
```

规模：

```text
6465 tasks
995 scenarios
```

来源：

在 7B runtime v2 的 7204 条上，用 Qwen3.5-397B-A17B 跑一轮 runtime：

```text
outputs/runs/legacy/qwen397b_v2allowlist_w48_v1
```

该 run 最后有 18 条 tail task 被 kill 时没有生成 trajectory / verify，因此单独记录并保守保留。
该 run 也已按 legacy 精简规则归档，原始目录名为 `awm_qwen397b_v2allowlist_runtime_w48_v1`。

产物组成：

```text
397B keep: 3337
397B review: 3110
missing_tail: 18
397B hard reject: 739，被删除
```

口径：

```text
v3 allowlist = 397B complete + 397B review + missing_tail
```

其中：

- `complete` 表示强模型实际跑通并通过 code verifier。
- `review` 表示强模型没有完成，但这一轮没有观察到明确 hard runtime issue。
- `missing_tail` 表示尾部卡住或被 kill，证据不足，不直接删除。
- `hard reject` 表示强模型也观察到工具、环境、schema 或 runtime 硬问题。

适用：

- 当前推荐主 arena。
- 后续正式跑分优先使用这个 allowlist。
- 相比 v2，它剔除了强模型也暴露硬问题的 task。

## 7B Mixed V4

路径：

```text
outputs/task_allowlists/runtime_7b_5runs_has_complete_and_other_v4/allowlist.jsonl
```

规模：

```text
1545 tasks
764 scenarios
```

来源：

仍然基于 5 轮 7B runtime 结果，但改用课程学习口径：

```text
5 次 7B run 中至少 1 次 complete
并且至少 1 次 others
```

含义：

- 至少一次 `complete`：task/env/verifier 大概率可闭环。
- 至少一次 `others`：7B 并非稳定完成，仍有训练和 skill 提升空间。

397B 在 V4 上的表现：

```text
complete: 1312
others: 230
judge_error: 1
missing: 2
score over all V4: 84.92%
score over verified V4: 85.03%
```

适用：

- 7B skill2lora 初期训练和验证。
- 构造“弱模型可学但不稳定”的 curriculum。
- 不适合作为最终 validity allowlist，因为它会删掉 7B 从未 complete 但 397B 可以完成的有效难题，也会删掉 7B 五次都 complete 的简单题。

## Auxiliary: 397B Complete Subset

路径：

```text
outputs/task_allowlists/runtime_qwen397b_v2allowlist_w48_v1_v3/keep.jsonl
```

规模：

```text
3337 tasks
953 scenarios
```

口径：

```text
397B runtime 中 reward_type == complete
```

可信度最高，因为它满足：

```text
强模型实际执行 agent trajectory
+ code verifier 执行成功
+ reward_type == complete
```

适用：

- 高质量成功轨迹池。
- skill distillation。
- SFT 数据。
- oracle skill / oracle LoRA 训练。

限制：

- 这是 success subset，不是完整评测集。
- 只覆盖强模型能完成的 task，会低估 benchmark 难度分布。

## Recommendation

按目标选择：

| 目标 | 推荐 allowlist |
| --- | --- |
| 正式主 arena 跑分 | 397B runtime v3 |
| 高质量成功轨迹 / skill 蒸馏 | 397B complete subset |
| 7B skill2lora 初期训练 | 7B mixed v4 |
| 最大覆盖、继续筛选 | static_code_verify 或 7B runtime v2 |
| 保守 runtime 有效性分析 | 7B runtime v2 |

简化原则：

```text
评测主集用 v3。
训练高质量正例用 397B complete subset。
7B curriculum 用 v4。
需要最大覆盖时回退到 v2 或 static。
```

## Run Artifact Cleanup

为节省磁盘，以下旧 runtime run 已删除：

```text
outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v1
outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v2
outputs/runs/awm_qwen25_7b_static300_runtime_w256p16_v3
```

以下 run 已按 `outputs/runs/legacy` 中的精简规则归档：

```text
awm_qwen25_7b_static8314_runtime_w256p64_v1 -> outputs/runs/legacy/qwen7b_static8314_w256p64_v1
awm_qwen25_7b_static8314_runtime_w256p64_v2 -> outputs/runs/legacy/qwen7b_static8314_w256p64_v2
awm_qwen25_7b_static8314_runtime_w256p64_v3 -> outputs/runs/legacy/qwen7b_static8314_w256p64_v3
awm_qwen25_7b_static8314_runtime_w256p64_v4 -> outputs/runs/legacy/qwen7b_static8314_w256p64_v4
awm_qwen25_7b_static8314_runtime_w256p64_v5 -> outputs/runs/legacy/qwen7b_static8314_w256p64_v5
awm_qwen397b_v2allowlist_runtime_w48_v1 -> outputs/runs/legacy/qwen397b_v2allowlist_w48_v1
```

精简规则：

```text
保留根目录 summary 类文件。
保留 3 个完整 task 目录。
删除其它 task 目录。
删除因此变空的 scenario 目录。
写入 cleanup_manifest.json 记录来源、保留和删除数量。
```

本次归档后保留的示例 task 目录均为：

```text
account_management_1/task_0
account_management_1/task_1
account_management_1/task_2
```
