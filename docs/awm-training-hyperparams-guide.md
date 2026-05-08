# AWM 论文训练超参数与 SFT/RL 参考

本文档总结 `Wang 等 - 2026 - Agent World Model Infinity Synthetic Environments for Agentic Reinforcement Learning.pdf` 中和训练有关的超参数、奖励设计、环境并发、history 管理与评测设置，并给出用于后续 SFT/RL 的实践参考。

## 1. 论文中的训练对象与规模

论文训练的是多轮 tool-use agent，底座模型为 Qwen3 thinking models，覆盖 4B、8B、14B 三个规模。训练框架基于 AgentFly + verl，算法为 GRPO。

完整 AWM 合成资源包含：

| 项目 | 数值 |
| --- | ---: |
| 合成环境数 | 1,000 |
| 合成任务数 | 10,000 |
| 平均每环境工具数 | 35.1 |
| 平均每环境代码行数 | 1,984.7 |
| 平均每任务 agent 步数 | 8.5 |
| 最大交互轮数 | 20 |

受算力限制，论文实际 RL 训练使用的是子集：

| 项目 | 数值 |
| --- | ---: |
| 训练环境数 | 526 / 1,000 |
| 训练任务数 | 3,315 / 10,000 |
| 优化步数 | 最多 96 |
| 每步环境实例数 | 1,024 |

实践含义：不要只关注总任务数。AWM 的关键是环境分布多样性。论文的 scaling curve 显示，10 个环境会明显过拟合，100 个环境有显著提升，526 个环境继续提升。

## 2. GRPO 超参数

论文 Table 8 给出的 GRPO 配置如下：

| 类别 | 超参数 | 数值 |
| --- | --- | ---: |
| GRPO | learning rate | `7e-7` |
| GRPO | batch size | `64` |
| GRPO | mini-batch size | `16` |
| GRPO | rollouts per task `G` | `16` |
| GRPO | instances per step | `1,024` |
| GRPO | max optimization steps | `96` |
| GRPO | KL coefficient | `0.001` |
| GRPO | entropy coefficient | `0.0` |
| GRPO | clip ratio high | `0.28` |
| Rollout | rollout temperature | `1.0` |
| Rollout | max response length | `2,048` |
| Rollout | max model context | `32,000` |
| Agent | max interaction turn | `20` |
| Agent | history window size | `3` |

补充说明：

- `clip ratio high = 0.28` 高于常规 GRPO，论文说这是参考 DAPO，用来鼓励 agent 探索。
- KL 系数是 `0.001`，属于较弱约束，适合保留底座能力同时允许 tool-use 行为变化。
- entropy coefficient 为 `0.0`，探索主要来自 rollout temperature 和较高 clip ratio，而不是 entropy bonus。
- 论文还使用 sequence-level importance sampling，用于缓解 rollout engine 和 training engine 之间的分布偏移。

## 3. Reward 设计

AWM 使用混合奖励：step-level format correctness + task-level outcome verification。

### 3.1 任务级奖励

每个 rollout 结束后，执行代码 verifier 比较初始数据库和最终数据库，再把 verifier 结构化结果和 agent trajectory 一起交给 GPT-5 judge。judge 输出四类标签，并映射为奖励：

| Judge 分类 | 奖励 |
| --- | ---: |
| `Completed` | `1.0` |
| `Partially Completed` | `0.1` |
| `Agent Error` | `0.0` |
| `Environment Error` | `0.0` |

论文强调不要只用纯 LLM judge，也不要只用纯代码 verifier。纯 LLM 容易被 agent 的成功声明骗过；纯代码 verifier 遇到幂等任务、瞬时工具错误、环境瑕疵时容易误判。AWM 用 verifier 提供数据库证据，用 LLM judge 结合轨迹处理歧义。

### 3.2 步级格式奖励

训练中每一步都会做格式校验。违反以下规则会提前终止 rollout，并给 `rt = -1`：

- assistant 消息必须有非空 `<think>...</think>`；
- 不允许调用未由 `list_tools` 返回的工具；
- 工具参数必须是符合 schema 的合法 JSON；
- 必须先且只调用一次 `list_tools`；
- 多轮交互时，不能只 `list_tools` 而没有后续成功工具调用。

如果 MCP server 返回 timeout、500 等服务端错误，则归为 environment error，提前终止并给 `rt = 0`。

正常完成 rollout 后，最终任务奖励 `Rτ` 会 broadcast 到所有 action steps。也就是说，格式错误是局部强惩罚，任务完成是 outcome reward。

实践含义：agentic RL 不建议只给最终成功/失败奖励。先把工具协议、JSON、工具名、参数 schema 这些约束做成 step-level validator，会显著降低无效 rollout。论文的消融显示，去掉格式奖励后，50 步后格式错误率仍超过 20%，任务完成率低于 40%；加上格式奖励后，平均 rollout 时间减少约 27%。

## 4. History-aware Training

论文认为多轮 agent 训练存在 train/inference mismatch：训练时很多 RL 框架用完整历史一次 forward 优化所有 action，但推理时常常只保留截断历史。

AWM 的做法：

- 使用 sliding window history truncation；
- 训练 history window size `w = 3`；
- 一个完整 rollout 如果有 `T` 个 assistant turns，会拆成 `T` 个训练样本；
- 第 `t` 个样本输入包括：
  - system prompt；
  - 初始 user message；
  - 第一轮 assistant-tool exchange，也就是 `list_tools`；
  - 当前 turn 前最近 `w = 3` 轮历史；
- loss 只打在第 `t` 个 assistant turn 的 token 上，之前上下文 token mask 为 0。

实践含义：

- 如果你的推理框架会截断历史，RL/SFT 最好也用同样或相近的 history 策略。
- 对长轨迹直接全历史训练会让模型依赖推理时未必存在的上下文。
- `list_tools` 被保留在每个样本中，保证模型始终知道工具集合来源。

## 5. 环境并发与状态隔离

论文每个训练 step 启动 `1,024` 个隔离环境实例。每个实例都是独立 MCP server，并绑定自己的 SQLite database copy。

关键工程设置：

- 并发 rollout 之间数据库状态互不影响；
- 每个 rollout 结束后，通过恢复初始数据库重置环境；
- 环境启动、MCP server spawn、数据库 copy 是 online RL bottleneck；
- 论文使用 pre-fetching：当前 batch 做梯度更新时，后台线程提前准备下一批环境。

实践含义：

- 如果资源不足，不必一开始追 `1,024` instances/step，但一定要保证状态隔离。
- agentic RL 的吞吐瓶颈通常不只是模型推理，还有环境启动和工具响应。
- 可优先实现环境池、数据库快照恢复、异步 judge、下一批环境预热。

## 6. 评测解码和上下文设置

论文评测时不是用训练 rollout 的 temperature，而是使用 Qwen3 推荐解码：

| 配置 | 数值 |
| --- | ---: |
| evaluation temperature | `0.6` |
| top-k | `20` |
| top-p | `0.95` |
| eval context window | `131,072` tokens |
| eval history window | `10` turns |

训练 rollout temperature 是 `1.0`，评测 temperature 是 `0.6`。评测时通过 RoPE scaling 扩展到 131k context，主要是因为 BFCLv3 multi-turn 等任务历史很长。

此外，不同 benchmark 的工具调用格式不同，论文用 converter 适配：

- AWM 内部统一为 `list_tools` + `call_tool`；
- τ²-bench 原生使用 direct tool names；
- BFCLv3 使用 function-calling syntax；
- MCP-Universe 使用 MCP protocol。

实践含义：评测时要避免把“格式不兼容”误当成能力不足。最好在 benchmark adapter 层转换格式，让指标尽量反映 task-solving ability。

## 7. 对 SFT 的参考建议

论文主要是 RL，不是 SFT recipe，但它的设计对 SFT 数据构造很有参考价值。

建议 SFT 样本结构：

1. 保留统一 agent prompt 和工具协议。
2. 每个样本从 `list_tools` 开始，训练模型先发现工具，再调用工具。
3. 对多轮轨迹做 sample splitting，避免只训练完整长上下文。
4. loss mask 只打 assistant action/answer，不打 observation。
5. 对工具调用格式做静态过滤，剔除 JSON 错、工具名错、参数 schema 错的样本。
6. 样本中保留足够的工具返回 observation，尤其是 ID、状态、错误码、候选列表，训练模型学会基于 observation 决策。

建议 SFT 数据配比：

| 样本类型 | 建议 |
| --- | --- |
| 成功完成轨迹 | 主体数据，用于学习标准工具使用流程 |
| 部分完成轨迹 | 少量保留，用于学习错误恢复和信息不足时的行为 |
| 环境错误轨迹 | 谨慎使用，最好只训练识别和报告环境错误，不训练错误工具调用 |
| 格式错误轨迹 | 不建议作为正样本；可作为偏好/RL 负样本 |
| 拒绝/不可完成任务 | AWM 原文覆盖不足；如果目标 benchmark 有 BFCLv3 类 hallucination resistance，SFT 阶段应额外补充 |

SFT 起步配置可参考，而不是论文原文：

| 配置 | 建议起点 |
| --- | --- |
| max turns | `20` 或按任务预算设定 |
| history window | `3-5` |
| max response length | `2,048` |
| context length | 至少 `32k`，长 benchmark 另配 `64k+` |
| loss mask | assistant tool call + final answer |
| observation loss | mask 掉 |

## 8. 对 RL 的参考建议

如果复现或改造 AWM 风格 RL，可以按资源分层。

### 8.1 资源充足时接近论文配置

| 配置 | 建议 |
| --- | --- |
| algorithm | GRPO |
| learning rate | `7e-7` |
| KL coefficient | `0.001` |
| rollouts per task | `16` |
| batch size | `64` |
| mini-batch size | `16` |
| rollout temperature | `1.0` |
| max response length | `2,048` |
| max context | `32k` |
| max turns | `20` |
| history window | `3` |
| clip ratio high | `0.28` |
| entropy coefficient | `0.0` |

### 8.2 资源受限时优先保留的设计

优先级从高到低：

1. 保留 executable environment + database state verification。
2. 保留 step-level format validator，格式错给负奖励并 early stop。
3. 保留多 rollout group，`G` 可从 `4` 或 `8` 起步，资源够再到 `16`。
4. 保留状态隔离，每个 rollout 独立数据库副本。
5. 保留 history truncation 和 sample splitting。
6. 环境数优先扩到 `100+`，再追每环境任务数。

不建议首先牺牲：

- 工具 schema 校验；
- 数据库前后状态 verifier；
- 环境重置；
- 格式错误 early stop。

可以按资源缩放：

- `instances per step`: 从 `128/256` 起步，资源够再上 `1,024`。
- `rollouts per task G`: 从 `4/8` 起步，稳定后到 `16`。
- `max optimization steps`: 先小步数观察 reward、format error、environment error。
- `judge`: 如果 GPT judge 成本高，先用 code verifier 做粗筛，再对不确定样本调用 judge。

## 9. 关键监控指标

论文显示，agentic RL 不能只看最终 benchmark 分数。训练中建议至少监控：

| 指标 | 含义 |
| --- | --- |
| format error rate | 工具协议是否学稳 |
| environment error rate | 环境质量和服务稳定性 |
| completed rate | 任务完成率 |
| partially completed rate | 是否有中间进展 |
| agent error rate | agent 真实失败率 |
| average rollout length | 是否过早终止或无效拖长 |
| blocked task rate | 任务是否被环境 bug 阻塞 |
| tool hallucination rate | 是否调用不存在工具 |
| invalid JSON/schema rate | 参数生成是否可靠 |
| judge/code verifier disagreement | 评分器是否存在系统性偏差 |

AWM 论文中环境错误率训练时约 `4%`，可作为一个粗略参考。若 environment error 或 blocked task 明显偏高，优先修环境，而不是直接调 RL 超参。

## 10. 结论性建议

用于后续 SFT/RL 时，可以把 AWM 的经验压缩为几条原则：

1. 先做高质量工具协议 SFT，再做 outcome RL，通常比直接 RL 更稳。
2. RL 的核心不是只给最终分，而是把格式错误、环境错误、任务完成拆开。
3. 多轮 agent 要对齐训练和推理的 history 策略。
4. 环境状态必须隔离、可重置、可验证。
5. 环境多样性比单一环境内堆大量任务更重要。
6. 对真实 benchmark，要补充 AWM 弱覆盖的能力：拒绝/抗幻觉、信息检索、浏览器自动化、多轮对话。

