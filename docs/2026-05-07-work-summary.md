# 2026-05-07 工作摘要

今天完成 AWM 并发跑分基线、MCP 端口回收修复、运行监控、轨迹质量分析和运维文档沉淀。

## 代码与脚本

- 跑分脚本：`run_parallel_local_score.py`、`run_parallel_api_score.py`、`awm_parallel_runner.py`。
- 运维脚本：`sglang_manage.py`、`monitor_run.py`。
- 修复 MCP server 回收：进程组启动/终止，避免残留端口触发 `no free MCP port`。
- 已提交：`553570c` 并发基线，`c5a1086` 端口回收修复，`3883ce4` 启动指南。

## 当前跑分数据

```text
Qwen2.5-7B full: verified 9931/10000, score 0.2981
complete 2960, others 6964, judge_error 7, bad_verify 0, disk 7.6G

Qwen3.6-27B full: verified 1941/2125, score 0.4843
complete 940, others 999, judge_error 2, bad_verify 0, disk 1.7G

DMXAPI gpt-5.5 stage1: verified 100/100, score 0.3800
complete 38, others 62, bad_verify 0, disk 89M

DMXAPI prompt v2: verified 86/90, score 0.4767
complete 41, others 45, bad_verify 0, disk 90M
```

## 主要结论

- 7B 已接近全量跑完，系统稳定但分数低；主要失败来自 malformed nested JSON、参数截断、重复工具错误。
- 27B 明显好于 7B，基本没有 assistant 原地复读，但仍受 422/500 和环境工具质量影响。
- DMXAPI 强模型轨迹更干净，适合继续小批量对照；100 task 分数低于 10 task smoke，暂不建议直接全量。
- verify 不是瓶颈：7B agent median 18.33s、p90 60.02s；code verify p90 0.01s。
- `Qwen3.5-397B-A17B` 约 752G，BF16 双卡 H200 放不下；需 8 卡、量化版或外部 API。

## 文档产出

- `docs/hermes-awm-operator-guide.md`
- `docs/awm-local-sglang-parallel-score-guide.md`
- `docs/awm-training-hyperparams-guide.md`
- `docs/agent-world-paper-analysis.md`

## 后续建议

1. 给 7B 加重复工具错误提前终止。
2. 增强 `call_tool` 参数解析，兼容对象参数和 json_repair 错层字段。
3. 跑完 DMXAPI prompt v2 的 100 task，再决定是否扩大到 500。
4. 轨迹收集优先保留 `reward_type=complete`，过滤 tool error 多和多 tool call 被跳过过多的样本。
