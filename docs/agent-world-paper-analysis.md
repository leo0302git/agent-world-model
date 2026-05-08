# Agent-World 论文分析

论文：Dong 等，2026，Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence

项目页：https://agent-tars-world.github.io/-/

本地 PDF：`docs/Dong 等 - 2026 - Agent-World Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence.pdf`

## 核心结论

Agent-World 可以看作 AWM 思路的进一步扩展：不只是合成可执行环境并训练 agent，而是把环境本身变成一个持续诊断、持续生成任务、持续 RL 的 self-evolving arena。

它的两个核心组件是：

1. Agentic Environment-Task Discovery
   - 从真实世界主题、MCP server specs、tool docs、工业 PRD 中收集环境主题。
   - 用 deep-research agent 挖掘真实结构化数据，构造数据库。
   - 用 coding agent 生成工具和单测，再通过 cross-validation 保留可编译、测试准确率大于 0.5、环境内至少有有效工具和测试的工具。
   - 通过 graph-based 和 programmatic 两种方式生成可验证任务。

2. Continuous Self-Evolving Agent Training
   - 在多环境 agent-tool-database rollouts 上做 GRPO。
   - 用 arena 动态评估当前 agent，诊断弱环境和失败模式。
   - 针对弱点重新合成 targeted tasks，必要时 complexify database，然后继续 RL。

这篇文章的主张是：环境规模、环境真实度、任务可验证性、以及基于失败轨迹的 targeted data evolution，要一起做，不能只静态堆环境。

## 数据和环境规模

Agent-World 的规模明显大于 AWM：

| 项目 | Agent-World |
| --- | ---: |
| retained environments | 1,978 |
| tools | 19,822 |
| 一级环境类别 | 20 |
| 二级环境类别 | 50 |
| evaluation benchmarks | 23 |

环境来源更接近真实世界，主要来自：

- MCP Servers：Smithery 等真实 MCP server specs；
- Tool Documentations：真实工具使用场景；
- Industrial PRDs：产品需求文档里的领域流程和系统接口；
- Web mining：deep-research agent 从网页挖掘数据库。

这和 AWM 的差别是：AWM 更强调从 seed scenario 合成完整 SQL-backed API environment；Agent-World 更强调从真实工具生态和真实数据源挖掘，然后再程序化验证。

## 任务合成逻辑

Agent-World 有两条任务合成路径。

### Graph-Based Task Synthesis

先构造工具依赖图，再通过 random walk 生成 tool-call sequence，最后反推用户任务。

依赖边分三类：

- strong dependency：后一个工具严格依赖前一个工具输出；
- weak dependency：后一个工具可由前一个输出推导，也可由其他方式得到；
- independent edge：无参数依赖，用来保证图连通。

生成流程：

1. 在工具图上随机游走得到工具链；
2. 根据工具输出和数据库采样填参数；
3. LLM 审核并去冗余；
4. sandbox 执行工具链；
5. LLM 根据真实 execution trace 改写自然语言任务和 ground-truth answer；
6. ReAct agent 跑 5 次，至少 2 次成功才保留。

### Programmatic Task Synthesis

直接让 LLM 生成复杂任务和 end-to-end Python solution，用代码表达循环、分支、聚合、排序、过滤等复杂逻辑。

然后再生成 executable verification script，用 sandbox debug，保证任务可以自动验证。

这条路径比 AWM 常见的 CRUD/state-diff 任务更强，因为它支持非线性控制流和复杂聚合。

## Reward 和训练

Agent-World 使用两类 verifiable reward：

- Graph-based tasks：rubric-conditioned LLM-as-judge，按多个 criteria 计算 pass rate。
- Programmatic tasks：执行 task-specific validation script，验证答案或数据库状态。

RL 算法是 GRPO。

论文给出的关键训练配置：

| 配置 | 数值 |
| --- | ---: |
| cold-start SFT trajectories | 40K |
| RL samples | 5K |
| backbone | Qwen3-8B / Qwen3-14B |
| rollouts per task | 8 |
| tasks per training step | 32 |
| rollout temperature | 1.0 |
| top_p | 1.0 |
| max trajectory length | 80K tokens |
| max generation length per step | 32K tokens |
| clip ratio low | 0.2 |
| clip ratio high | 0.28 |

和 AWM 相比，Agent-World 的 max trajectory/context 预算更大，任务更偏长链和复杂控制流。AWM 论文中 max context 是 32K，max response length 是 2,048；Agent-World 直接把 trajectory 放到 80K，每步 generation 到 32K。

## Self-Evolving Arena

Agent-World 最值得注意的是 self-evolving loop。

每一轮：

1. 从分层环境 taxonomy 中构造 arena，每个一级类别采样 K=5 个环境；
2. 为 arena 环境动态生成新任务和 verifier；
3. 当前 policy 在这些 fresh tasks 上跑；
4. diagnosis agent 分析失败轨迹、工具日志、validator feedback、环境统计；
5. 输出 weak environments 和 task-generation guidelines；
6. 针对弱点重新生成 targeted tasks，必要时扩展数据库；
7. 继续 RL 得到下一轮 policy。

这个机制很适合我们当前的方向：不要只从固定 bench 一次性采 trajectory，而是把失败轨迹转成下一批 skill/task/env 生成条件。

## 主结果

Agent-World 在 23 个 benchmark 上评估。主表包含 MCP-Mark、BFCL V4、τ²-Bench。

关键结果：

| 方法 | MCP-Mark Avg | BFCL V4 Avg | τ²-Bench Avg |
| --- | ---: | ---: | ---: |
| GPT-5.2 High | 53.1 | 62.9 | 80.2 |
| Claude Sonnet-4.5 | 33.3 | 73.2 | 84.7 |
| Gemini-3 Pro | 50.8 | 72.5 | 85.4 |
| Qwen3-8B | 2.4 | 40.4 | 26.2 |
| Qwen3-14B | 3.4 | 41.0 | 32.4 |
| EnvScaler-8B | 5.6 | 47.6 | 37.9 |
| AWM-8B | 2.4 | 40.0 | 34.4 |
| AWM-14B | 5.1 | 42.4 | 39.0 |
| Agent-World-8B | 8.9 | 51.4 | 61.8 |
| Agent-World-14B | 13.3 | 55.8 | 65.4 |

注意：Agent-World 在 MCP-Mark 上仍显著低于 GPT-5.2 High 和 Gemini-3 Pro，但在 BFCL/τ²-Bench 上已经逼近或超过部分强模型。论文强调的是小模型训练后跨环境泛化能力，而不是绝对超过所有 proprietary models。

## Scaling 结论

论文逐步增加训练环境数：0、10、100、500、1000、2000。

平均分从 18.4% 上升到 38.5%，提升 20.1 点。增益最明显的阶段是：

- 10 -> 100 environments；
- 100 -> 500 environments。

500 以后仍有正收益，但边际下降。

这对我们很直接：如果做 skill/lora 或 SFT/RL 数据，优先追环境多样性。先覆盖 100+、再到 500+，比在少数环境里堆大量 trajectory 更有价值。

## 对当前 AWM 工作的启发

1. 强模型 teacher trajectory 不一定直接高分
   - Agent-World 结果也显示强 proprietary models 在 MCP-Mark 这种 stateful MCP 环境里并不稳定。
   - 我们观察到 DMX 强模型在 AWM 上偏保守、会追问、会自然语言改写枚举值，这和论文结论一致：frontier model 不等于 benchmark protocol-aligned agent。

2. Teacher trajectory 需要协议适配 prompt
   - 必须强调不要追问用户；
   - 枚举值、ID、国家/状态缩写必须优先用 observation 原值；
   - 不要自行补充 schema 未要求的参数；
   - 如果环境提供 create/update 工具，就继续执行，不要因为真实世界缺外部文件而拒绝。

3. 清洗数据时要保留失败类型
   - Agent-World 的 diagnosis agent 依赖 failure traces。
   - 对我们来说，`others` 不应该全丢；应该至少保留一份 failure analysis 数据，用于生成 targeted skill 或修 prompt。

4. 后续可以做 AWM-lite self-evolving loop
   - 用当前 AWM run 统计失败任务；
   - 聚类失败原因：枚举值错误、过度追问、漏工具、错 ID、final answer 缺字段；
   - 生成 targeted prompt 或 targeted SFT samples；
   - 重新跑强模型/SLM；
   - 比较同任务和新任务 gain。

5. 对 skill2lora 的意义
   - 不要只收集 `(task, trajectory)`；
   - 更应该收集 `(environment/tool schema, failure mode, corrected trajectory, distilled skill)`；
   - skill 可以来自 failure diagnosis，而不只是成功轨迹总结。

## 建议下一步

短期：

1. 在 AWM 强模型跑分里加 protocol-adapter prompt，专门修复强模型保守追问和枚举改写问题。
2. 把 `others` 轨迹分成可修复 agent error 和环境/verifier error。
3. 对 DMX 输给 27B 的 case 做小批量 prompt ablation。

中期：

1. 以 AWM 的 scenario taxonomy 做环境级采样，优先覆盖 100+ scenario。
2. 对每轮失败生成 targeted tasks 或 targeted skills。
3. 用 cleaned successful trajectories 做 SFT，用 corrected failure trajectories 做 preference/RL 数据。

长期：

1. 把 AWM 从静态 benchmark 变成 mini self-evolving arena。
2. 用 diagnosis -> skill synthesis -> LoRA/SFT/RL -> re-eval 闭环，复现 Agent-World 的核心机制。
