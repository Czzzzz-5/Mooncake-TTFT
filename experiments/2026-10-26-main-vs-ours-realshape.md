# 2026-10-26 realshape 测速：Mooncake main vs ours 同轮 A/B

## 1. 设计

- **A 组（main）**：origin/main e58ad953，git worktree `~/mooncake_build/Mooncake-main`，
  手工插入 `MC_TCP_PHASE_TIMING=1` 纯测量打点（`[MC-PHASE]` copy/io 计时），编 wheel 装入 venv
- **B 组（ours）**：10-19 重编版本（`MC_TCP_PHASE_TIMING=1` 时打 `[XDBG]` PH1/PH2/PH3 事件）
- 切换工具：`~/mooncake_build/switch_mooncake.sh {ours|main}`，ours 有整包快照备份
- 三模型 KV shape：k3（24 层 MLA，27.6 KB/tok）、dsv3（61 层 MLA，70.3 KB/tok）、
  qwen7b（28 层 GQA，56 KB/tok）
- 16 档 token（128 → 1M）× 每档 3 次实测 + 1 次 warmup，P=GPU1、D=GPU2，TCP loopback
- 测量命令由用户本人敲（`./run_pair.sh <模型> <main|20261026>`），两侧同天紧邻跑，
  唯一变量是 mooncake 版本；main 不认识 `MC_TCP_GPU_STAGING_SLOT_MB`，静默忽略，命令两侧统一

## 2. 环境

- 3× RTX 5090（32GB），本轮用 GPU1/GPU2；12:42–14:17 顺序跑完 6 轮（3 模型 × 2 版本）
- 跑后 14:30 确认三张卡显存回 ~2MiB、无残留 bench 进程
- main 轮次 12:42–13:24，ours 轮次 13:54–14:17（同日下午，环境无变化）

## 3. 结果总表（总时间，每档 3 次取平均）

全档表见 `mooncake_data/realshape_{k3,dsv3,qwen7b}_{main,20261026}.jsonl`，
复现：`python3 ab_compare.py`。128k 档汇总：

| 模型 | 数据量 | main ms | ours ms | main GB/s | ours GB/s | 提速比 |
|---|---|---|---|---|---|---|
| k3 | 3624 MB | 2055.9 | 1295.2 | 1.76 | 2.80 | **1.59x** |
| dsv3 | 9211 MB | 5963.2 | 2895.5 | 1.54 | 3.18 | **2.06x** |
| qwen7b | 7516 MB | 4919.3 | 2445.2 | 1.53 | 3.07 | **2.01x** |

全档趋势：ours 在各档一致领先（1.6x–2.5x）；小档（≤4k）领先更多（main 固定开销占比高）。
两侧带宽平台期：main ~1.5 GB/s，ours ~3.1 GB/s。


## 4. 128k 分段解剖

main = 各 lane copy/io 求和（多 lane 并行，求和 > 墙钟，看结构占比）；
ours = 各段事件 span（流水线交叠，span 之和 ≥ 总时间属正常）。

### main

| 模型 | PH1 P侧D2H ms | PH2 网络 ms | PH3 D侧H2D ms | copy 结构占比 |
|---|---|---|---|---|
| k3 | 971 | 7704 | 384 | 15.0% |
| dsv3 | 2712 | 22428 | 1062 | 14.4% |
| qwen7b | 2112 | 18763 | 839 | 13.6% |

main 单 lane 内"同步拷贝→收发→下一块"严格串行：每 lane 时间线 ~14% 在拷显存、~86% 在等 socket。

### ours

| 模型 | PH1 P侧D2H ms | PH2 网络 ms | PH3 D侧H2D ms | 总时间 ms |
|---|---|---|---|---|
| k3 | 1246.2 | 1289.7 | 1266.9 | 1295.2 |
| dsv3 | 2830.9 | 2890.0 | 2871.5 | 2895.5 |
| qwen7b | 2281.8 | 2439.7 | 2279.6 | 2445.2 |

ours 三段 span 各自 ≈ 总时间：D2H、网络、H2D 几乎完全流水交叠，
墙钟被最慢一段（网络）摊平，拷贝不再串行占位。



