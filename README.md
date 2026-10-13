# pd-kv-transfer

> English abstract: Optimizing KV-cache transfer between vLLM prefill/decode
> nodes over Mooncake TCP transport on commodity hardware (no RDMA, no sudo).
> Final architecture: the vLLM connector is a thin protocol layer (segment
> tables / manifests / scheduling only); staging slots, gather/scatter copies
> and network transfer all live in the Mooncake transfer engine
> ("sinking the data plane into the engine"). At 100k-token prompt scale the
> new path succeeds 4/4 while the legacy per-descriptor path deadlocks 2/3
> (175k descriptors > engine queue limit); network phase 2.4-2.9s → 0.31-0.41s
> via batched memcpy + multi-lane split.

## 问题

vLLM PD 分离（P 和 D 各一张 RTX 5090），P 算完的 KV cache 经 Mooncake TCP
传给 D。Qwen2.5-7B，28 层；16k 字符请求 KV ~346MB，100k token 请求 ~5.7GB。

**TTFT 权威口径**：客户端发请求 → 收到第一个 token（`scripts/measure_ttft_serving.py` 计时）。

## 最终架构：数据面下沉到 Mooncake 引擎

```
P: vLLM connector 只交段表 [{gpu_addr,len}×N] + 对端槽基址
   Mooncake 引擎：领 staging 槽 → cudaMemcpyBatchAsync 一次批量 gather
   → 拆 K 条描述符（吃满多 TCP lane）异步发送 → 完成回调还槽
D: Mooncake 引擎：字节落引擎槽 → cudaMemcpyBatchAsync 批量 scatter H2D
   vLLM connector 只按 manifest 校验、标记 KV 就绪
```

- vLLM connector 只剩**协议薄层**：段表/manifest 构建（block 布局知识）、
  调度、XDBG 打点。**vLLM 锁页内存占用：P 侧 0、D 侧 0**（此前各 8GB）。
- Mooncake fork 新增通用引擎能力：`submit_gather_write` /
  `scatterHostToDevice` / 槽池管理——任何"GPU 散块→对端连续 buffer"
  场景可复用，是对上游社区的自包含贡献形态。

## 关键数字

**100k token 长 prompt**（YaRN-128k 模型，2P1D，同轮 A/B，
实验卡 §13/§14）：

| 路径 | 成功率 | 总耗时 | prefill | gather | 网络 | scatter |
|---|---|---|---|---|---|---|
| 新路（下沉 + M2b 拆条） | **4/4** | **18.4-19.5s** | 16.5s | 0.25s | **0.31-0.41s** | 0.35s |
| 新路单描述符（M2b 前） | 4/4 | 21.1-21.7s | 16.5s | 0.25s | 2.4-2.9s | 0.35s |
| 老路（逐描述符） | **1/3**（2 发挂死） | — | — | — | — | — |

- 老路在 100k 下**正确性级别不可靠**：block 碎片化时退化成 17.5 万个
  描述符，超引擎 65535/peer 队列上限 → submitTransfer 失败 → D 侧
  永久挂死。新路对 block 布局完全免疫。
- `cudaMemcpyBatchAsync` 把 17.5 万段 gather 从 **11.1s 压到 0.24s**
  （launch-bound → 带宽 bound）。
- M2b 拆条（1 描述符 → 16 条）把 5.7GB 传输从单 lane ~2.2GB/s 吃到
  ~15GB/s；100k 全链路传输占比降到 **5%**，剩下全是 prefill 计算本身。

**16k 字符 × 32 请求并发 16**（实验卡 §9，引擎侧 gather 落地轮）：
TTFT p50 1729→**1570ms（-9.2%）**，mean **-20.4%**，尾部 max
1827→1659ms。

早期基线（16k 字符 × 4 请求）：自研 TCP connector 718.71ms →
Mooncake staging 池 352~359ms（`experiments/2026-09-16-v1-reverify.md`）。

## 代码资产

- **Mooncake fork** 分支
  [`gpu-staging-v1`](https://github.com/Czzzzz-5/Mooncake/tree/gpu-staging-v1)：
  staging 槽池（`gpu_staging_pool.h`）、多 io 线程修复、引擎侧
  gather/scatter（`submit_gather_write` + `cudaMemcpyBatchAsync`）、
  拆条多 lane、槽上限修复。
- **vLLM 侧改动**：`patches/vllm-mooncake-gpu-staging.patch`（单文件
  全量 patch，110KB，基于上游 `ba07e4a48f`，已验证干净应用；
  演进故事见 `patches/README.md`）。
- 测量工具：`scripts/`（TTFT 测量、epoch 打点解剖、gather/scatter
  自检、prompt 生成器）；工具链文档 `PROFILING.md`。

## 目录

- `ROADMAP.md` — 做过什么、放弃了什么、下一步
- `experiments/` — 一次实验一张卡；**主卡
  `2026-10-10-sink-staging-to-engine.md`（§1-14）记录整个下沉方案的
  设计、分期实施、A/B 数据与踩坑**
- `results/raw/` — 原始测量档
- `patches/` — vLLM 侧改动的权威形态
- `models/Qwen2.5-7B-Instruct-yarn128k/` — 100k 实验用 YaRN 配置
  （仅 config，权重不入库）

## 复现

```bash
# 起服务（prefill → prefill2 → decode → proxy）
bash scripts/serve_pd_mooncake.sh prefill
bash scripts/serve_pd_mooncake.sh prefill2
bash scripts/serve_pd_mooncake.sh decode
bash scripts/serve_pd_mooncake.sh proxy2p

# 测 TTFT（第 1 条 warmup 自动丢弃）
python scripts/measure_ttft_serving.py \
    --num-prompts 4 --target-chars 16000 --max-tokens 10 --seed 777

# 100k prompt 生成（精确 token 数校准）
python scripts/gen_prompt_tokens.py
```

环境：vLLM 源码版（上游 `ba07e4a48f` + `patches/` 全量 patch）、
Mooncake fork 分支 `gpu-staging-v1`。

## 铁律（归因纪律）

1. 端到端数字必须配**代码路径证据**（日志计数 / WARNING / xfer time），
   不能只报 TTFT。
2. 每次实验记录**当时机器负载**（共享机，别人的任务会污染读数）。
3. 关键结论至少两个 seed 复跑。
