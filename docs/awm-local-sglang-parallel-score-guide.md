# AWM 本地 SGLang 并发跑分启动指南

本文档记录在本机启动 SGLang 服务、运行 AWM 并发跑分、监控进度和停止服务的推荐命令。所有命令都使用绝对路径，避免因为当前工作目录或 Python 环境不同导致行为不一致。

## 当前环境状态

AWM 仓库路径：

```bash
/data1/jczhong/repos/agent-world-model
```

AWM 数据集路径：

```bash
/data1/jczhong/datasets/AgentWorldModel-1K
```

推荐本地模型路径：

```bash
/data1/models/Qwen/Qwen3.6-27B
/data1/jczhong/models/Qwen2.5-7B-Instruct
```

项目虚拟环境 Python：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python
```

已在项目 `.venv` 中安装：

```text
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cudnn-cu12==9.16.0.29
```

安装命令记录：

```bash
/usr/local/bin/uv pip install \
  --python /data1/jczhong/repos/agent-world-model/.venv/bin/python \
  nvidia-cudnn-cu12==9.16.0.29
```

注意：当前 `.venv` 里只有 AWM 运行依赖和新的 CuDNN/CUBLAS/NVRTC wheel，没有安装 `torch` 和 `sglang`。当前系统里能 import SGLang 的 Python 是：

```bash
/usr/bin/python
```

如果使用 `/usr/bin/python -m sglang.launch_server`，它当前看到的是：

```text
torch 2.9.1+cu129
cudnn 9.10
```

因此，之前 SGLang 日志里的 CuDNN 检查错误不会因为只升级项目 `.venv` 而自动消失。要长期稳定启动 SGLang，推荐把 SGLang/Torch 也安装进项目 `.venv`，或者升级当前 SGLang 所在的 `/usr/bin/python` 环境中的 CuDNN。临时绕过可设置 `SGLANG_DISABLE_CUDNN_CHECK=1`，但这可能带来性能和显存风险。

## 1. 检查 SGLang 服务状态

检查本机 `8000` 端口是否已经有 OpenAI-compatible 服务：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py status \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.pid
```

如果输出：

```text
models_api: True
```

说明服务可用，可以直接跑分。

如果输出 `502 Bad Gateway` 或连接失败，说明该地址没有正常可用的 SGLang/OpenAI-compatible 服务。

## 2. 启动 SGLang 服务

推荐先使用 7 号卡、64K 上下文启动 `Qwen3.6-27B`：

```bash
cd /data1/jczhong/repos/agent-world-model


SGLANG_DISABLE_CUDNN_CHECK=1 \
CUDA_VISIBLE_DEVICES=7 \
/usr/bin/python -m sglang.launch_server \
  --model-path /data1/models/Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b \
  --host 127.0.0.1 \
  --port 8000 \
  --context-length 65536 \
  > /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.log 2>&1 &

echo $! > /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.pid
```

说明：

- 这条命令使用 `/usr/bin/python`，因为当前 SGLang 安装在系统 Python 环境中。
- `SGLANG_DISABLE_CUDNN_CHECK=1` 是临时绕过方式。更稳的长期方案是让 SGLang 所在 Python 环境使用 CuDNN 9.15+。
- 跑分脚本不会自动停止 SGLang；用户中止跑分后，SGLang 仍会驻留显存。

启动后等待一段时间，再检查：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py status \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.pid
```

查看日志：

```bash
tail -n 200 /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.log
```

## 3. 使用管理脚本启动 SGLang

也可以通过管理脚本启动。注意当前管理脚本内部启动命令是 `python -m sglang.launch_server`，所以实际使用哪个 Python 取决于调用环境的 `PATH`。如果要严格控制 Python 环境，优先使用上一节的显式 `/usr/bin/python -m sglang.launch_server` 命令。

管理脚本启动命令：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py start \
  --model-path /data1/models/Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b \
  --gpu 7 \
  --host 127.0.0.1 \
  --port 8000 \
  --context-length 65536 \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.pid \
  --log-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.log
```

如果该地址已有可用 `/v1/models`，管理脚本会直接复用，不重复启动。

## 4. 本地并发跑分

确认 SGLang 服务可用后，运行一个小规模 smoke run：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/run_parallel_local_score.py \
  --data /data1/jczhong/datasets/AgentWorldModel-1K \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen3.6-27b \
  --workers 2 \
  --base-port 9100 \
  --port-stride 100 \
  --scenario-limit 2 \
  --task-ids 0-2 \
  --verify-mode code \
  --max-iterations 30 \
  --max-tokens 4096 \
  --temperature 0.7 \
  --run-name awm_qwen36_27b_smoke
```

输出目录：

```bash
/data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_smoke
```

每个 task 的目录结构类似：

```text
outputs/runs/<run_name>/<scenario>/task_<task_id>/
  initial.db
  final.db
  trajectory.json
  server_code.py
  server.log
  verify.code.json
  runner_agent.log
  runner_verify.log
```

并发 runner 会在所有 worker 结束后统一生成：

```text
outputs/runs/<run_name>/manifest.json
outputs/runs/<run_name>/results.jsonl
outputs/runs/<run_name>/summary.json
```

## 5. 同时跑 27B 和 7B

目标：

- `Qwen3.6-27B` 使用 7 号卡，SGLang API 端口 `8000`，AWM MCP 端口段从 `9100` 开始。
- `Qwen2.5-7B-Instruct` 使用 6 号卡，SGLang API 端口 `8001`，AWM MCP 端口段从 `9300` 开始。
- `gen_tasks.jsonl` 有 1000 个 scenario，`--task-ids 0-9` 会跑每个 scenario 的 10 个 task，总计 10000 个 task。
- 两个跑分 run 的 MCP 端口段不重叠，因此不会互相抢 AWM task server 端口。
- 两个跑分输出目录、runner 日志、pid 文件也互不重叠。

如果 7 号卡的 27B 服务已经在 `8000` 上运行，可以跳过 27B 服务启动命令。检查命令：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py status \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.pid
```

如需启动 6 号卡的 7B 服务：

```bash
mkdir -p /data1/jczhong/repos/agent-world-model/outputs/services

CUDA_VISIBLE_DEVICES=6 \
SGLANG_DISABLE_CUDNN_CHECK=1 \
nohup /usr/bin/python -m sglang.launch_server \
  --model-path /data1/jczhong/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b-instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --context-length 32768 \
  > /data1/jczhong/repos/agent-world-model/outputs/services/qwen2.5-7b.log 2>&1 &

echo $! > /data1/jczhong/repos/agent-world-model/outputs/services/qwen2.5-7b.pid
```

检查 7B 服务：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py status \
  --api-url http://127.0.0.1:8001/v1 \
  --api-key EMPTY \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen2.5-7b.pid
```

启动 27B 全量跑分：

```bash
mkdir -p /data1/jczhong/repos/agent-world-model/outputs/runs

nohup /data1/jczhong/repos/agent-world-model/.venv/bin/python \    
    /data1/jczhong/repos/agent-world-model/scripts/run_parallel_local_score.py \
    --data /data1/jczhong/datasets/AgentWorldModel-1K \
    --api-url http://127.0.0.1:8000/v1 \
    --api-key EMPTY \
    --model qwen3.6-27b \
    --workers 32 \
    --base-port 9100 \
    --port-stride 100 \
    --task-ids 0-9 \
    --verify-mode code \
    --max-iterations 30 \
    --max-tokens 4096 \
    --temperature 0.7 \
  echo $! > /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_full.pid.log 2>&1 < /dev/null &

echo $! > /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_full.pid
```

启动 7B 全量跑分：

```bash
mkdir -p /data1/jczhong/repos/agent-world-model/outputs/runs

nohup /data1/jczhong/repos/agent-world-model/.venv/bin/python \
    /data1/jczhong/repos/agent-world-model/scripts/run_parallel_local_score.py \
    --data /data1/jczhong/datasets/AgentWorldModel-1K \
    --api-url http://127.0.0.1:8001/v1 \
    --api-key EMPTY \
    --model qwen2.5-7b-instruct \
    --workers 64 \
    --base-port 13000 \
    --port-stride 100 \
    --task-ids 0-9 \
    --verify-mode code \
    --max-iterations 30 \
    --max-tokens 4096 \
    --temperature 0.7 \
  echo $! > /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen25_7b_full.pid.log 2>&1 < /dev/null &
```

查看两个跑分日志：

```bash
tail -f /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_full.nohup.log
tail -f /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen25_7b_full.nohup.log
```

监控两个 run 的完成情况：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/monitor_run.py \
  /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_full

/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/monitor_run.py \
  /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen25_7b_full
```

如果要中止某个跑分，只杀 runner pid；SGLang 服务仍会驻留显存，后续可以复用：

```bash
kill -TERM "$(cat /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_full.pid)"
kill -TERM "$(cat /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen25_7b_full.pid)"
```

如果要释放 7B 服务显存：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py stop \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen2.5-7b.pid \
  --timeout 30 \
  --kill
```

## 6. 断点续跑

默认启用 resume。只要某个 task 目录下存在合法的 `verify.code.json`，该 task 就会被跳过。

重新运行同一个 `--run-name` 即可续跑：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/run_parallel_local_score.py \
  --data /data1/jczhong/datasets/AgentWorldModel-1K \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model qwen3.6-27b \
  --workers 2 \
  --base-port 9100 \
  --port-stride 100 \
  --scenario-limit 2 \
  --task-ids 0-2 \
  --verify-mode code \
  --run-name awm_qwen36_27b_smoke \
  --resume
```

注意：

- `reward_type=complete` 算完成。
- `reward_type=others` 也算完成，因为这是有效判题结果。
- `reward_type=judge_error` 也算完成一次 verify；是否重跑由后续人工决定。
- 没有合法 `verify.code.json` 的 task 会重跑。

## 7. 运行中监控

监控某个 run 的当前进度：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/monitor_run.py \
  /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_smoke
```

持续刷新：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/monitor_run.py \
  /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen36_27b_full \
  --watch \
  --interval 1

  /data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/monitor_run.py \
  /data1/jczhong/repos/agent-world-model/outputs/runs/awm_qwen25_7b_full \
  --watch \
  --interval 1

    /data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/monitor_run.py \
  /data1/jczhong/repos/agent-world-model/outputs/runs/awm_prompt_v2_dmxapi_w4_100 \
  --watch \
  --interval 1
```

该脚本只读扫描 run 目录，不写任何文件。

## 8. 停止 SGLang 并释放显存

跑分脚本不会自动停止 SGLang。需要释放显存时，显式执行：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py stop \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.pid \
  --timeout 30 \
  --kill
```

停止后检查 GPU：

```bash
nvidia-smi
```

## 9. 远端 API 并发跑分

如果使用远端 OpenAI-compatible API，不需要启动 SGLang：

```bash
  nohup /data1/jczhong/repos/agent-world-model/.venv/bin/python \
    /data1/jczhong/repos/agent-world-model/scripts/run_parallel_api_score.py \
    --data /data1/jczhong/datasets/AgentWorldModel-1K \
    --api-url https://www.dmxapi.cn/v1 \
    --api-key "$OPENAI_API_KEY" \
    --model gpt-5.5 \
    --workers 4 \
    --base-port 17000 \
    --port-stride 100 \
    --scenario-limit 50 \
    --task-ids 0-1 \
    --verify-mode code \
    --max-iterations 20 \
    --max-tokens 4096 \
    --temperature 0.6 \
    --run-name awm_dmxapi_stage1_w4_100 \
    > /data1/jczhong/repos/agent-world-model/outputs/runs/awm_dmxapi_stage1_w4_100.nohup.log 2>&1 < /dev/null &

  echo $! > /data1/jczhong/repos/agent-world-model/outputs/runs/awm_dmxapi_stage1_w4_100.pid

```

远端 API 跑分仍然会为每个 task 本地启动 MCP server，因此仍需要 `--base-port` 和 `--port-stride`。

## 10. 常见问题

### SGLang 日志出现 CuDNN 检查错误

错误类似：

```text
PyTorch 2.9.1 & CuDNN Compatibility Issue Detected
Current Environment: PyTorch 2.9.1+cu129 | CuDNN 9.10
```

原因是当前启动 SGLang 的 Python 环境仍在使用 CuDNN 9.10。项目 `.venv` 中安装的 CuDNN 9.16 不会自动影响 `/usr/bin/python` 环境。

可选处理：

1. 推荐：在真正启动 SGLang 的 Python 环境中升级 `nvidia-cudnn-cu12==9.16.0.29`。
2. 临时：启动时设置 `SGLANG_DISABLE_CUDNN_CHECK=1`。
3. 更干净：把 `torch` 和 `sglang` 安装到项目 `.venv`，然后统一用 `/data1/jczhong/repos/agent-world-model/.venv/bin/python` 启动。

### local score 脚本报 `/models` 不可用

先检查：

```bash
/data1/jczhong/repos/agent-world-model/.venv/bin/python \
  /data1/jczhong/repos/agent-world-model/scripts/sglang_manage.py status \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --pid-file /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.pid
```

如果 `models_api: False`，说明推理服务没有正常就绪，先看日志：

```bash
tail -n 200 /data1/jczhong/repos/agent-world-model/outputs/services/qwen3.6-27b.log
```

### MCP 端口冲突

并发 runner 会给每个 worker 分配独立端口段：

```text
worker 0: base_port + 0 * port_stride
worker 1: base_port + 1 * port_stride
...
```

如果某个端口被占用，会在该 worker 自己的端口段内找下一个可用端口，不跨 worker 段。

如果整段都不可用，增大 `--port-stride` 或换一个 `--base-port`。
