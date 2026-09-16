# pd-kv-transfer

> English abstract: Optimizing KV-cache transfer between vLLM prefill/decode nodes
> over Mooncake TCP transport on commodity hardware (no RDMA, no sudo).
> The GPU staging path is being rebuilt: from 3520 synchronous per-64KB pageable
> copies (~935 MB/s) to a pinned-memory staging pool with single-shot async copies,
> aiming for a pipelined design at ~30ms per 115MB transfer.

## 问题

vLLM PD 分离（P 和 D 各一张 RTX 5090），P 算完的 KV cache 经 Mooncake TCP
传给 D。Qwen2.5-7B，28 层，单请求 KV ~115MB。

**TTFT 权威口径**：客户端发请求 → 收到第一个 token（`scripts/measure_ttft_serving.py` 计时）。

## 当前数字

| 方案 | TTFT (4×16k字符, seed 777) | 传输时间 / 115MB | 有效带宽 | 状态 |
|---|---|---|---|---|
| 自研 TCP connector (bf16) | 718.71 ms | — | ~450 MB/s | 存档 |
| Mooncake TCP 原生 GPU 路径 | 406.63 ms ※ | ~123 ms | 935 MB/s | 存档（含噪声） |
| 原生路径（安静机重测，A 组） | 394~427 ms（3 seeds） | ~150 ms | ~780-910 MB/s | ✅ 对照基线 |
| + GPU staging v1（锁页池，B 组） | **352~359 ms** | — | — | ✅ **已验证，−54ms / −13%** |

※ 旧基线在有第三方负载时测得，仅存档，不参与对比。三轮 A/B 对照
（seed 777/888/123，B 全部优于 A）：
`experiments/2026-09-16-v1-reverify.md`。

※ 旧基线在有第三方负载时测得，仅存档，不参与对比。A/B 对照实验：
`experiments/2026-09-16-v1-reverify.md`。

| 下一步 | 传输时间目标 | 状态 |
|---|---|---|
| 多槽流水 v2（拷贝‖发送） | ~30 ms | 📋 设计 |

标尺：裸 TCP 单连接 ~5.5 GB/s；Mooncake CPU buffer 路径 ~3.4 GB/s。

## 复现

```bash
# 起服务（prefill → decode → proxy 依次）
bash scripts/serve_pd_mooncake.sh prefill
bash scripts/serve_pd_mooncake.sh decode
bash scripts/serve_pd_mooncake.sh proxy

# 测 TTFT（第 1 条 warmup 自动丢弃）
python scripts/measure_ttft_serving.py \
    --num-prompts 4 --target-chars 16000 --max-tokens 10 --seed 777

# 传输层带宽拆包 bench
python scripts/mooncake_tcp_bench.py --buf gpu
```

环境：vLLM 源码版（0.26.1.dev + 本地 patch）、Mooncake v0.3.13.post1 +
GPU staging patch（fork 分支 `gpu-staging-v1`，见 ROADMAP）。

## 铁律（归因纪律）

1. 端到端数字必须配**代码路径证据**（日志计数 / WARNING / xfer time），
   不能只报 TTFT。参考 `experiments/` 里 2026-09-15 那次事故的教训。
2. 每次实验记录**当时机器负载**（共享机，别人的任务会污染读数）。
3. 关键结论至少两个 seed 复跑。

## 目录

- `ROADMAP.md` — 做过什么、放弃了什么、下一步
- `experiments/` — 一次实验一张卡
- `scripts/` — 实验脚本（从 my_pd_demo 抽出的主力三件套）
