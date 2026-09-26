# 实验卡：V3 合账轮 —— 315ms 瀑布全拆解 + P 侧完成通知延迟定位

日期：2026-09-23　操作人：wuzichun（测量命令本人执行）

## 目的与方法

给测量脚本加客户端 epoch 打点（`[epoch] t0/t1`，与引擎 XDBG 同为
CLOCK_REALTIME，同机合法配对），把 gather 后的 TTFT 完整拆段对账。
D 侧调度器 [TTFT] 打点（perf 时钟）用"第 k 个 req_queued ↔ 第 k 个
D_SEND_PULL"按序配对标定偏移。

## TTFT 瀑布（V3 轮 seed 123，四请求一致，对账闭合）

| 段 | 耗时 | 判定 |
|---|---|---|
| client→proxy→P→prefill→D 发 pull | 21~24ms | 正常 |
| **pull 到达 P → P 侧 request_finished** | **~164ms** | ⛔ 最大残留 |
| 就绪→gather→提交发送 | 15~29ms | 有压缩空间 |
| wire 传输 148MB（1 描述符） | 54~56ms（2.7GB/s） | 距 R1 bench 地板 ~18ms |
| 响应回传 + scatter 打散 | ~8ms | 小 |
| D 排上→首 token→客户端 | ~34ms | 正常 |
| **TTFT 合计** | **312~333ms** | ✓ 逐段加总闭合 |

## ⛔ 主要发现：P 侧完成通知延迟 ~148ms（未修复，遗留）

V3b 轮用同进程同时钟（perf）打点证实：P 引擎每请求只跑 1 步前向
（正常），但 `first_token_generated → connector.request_finished`
间隔：warmup1 = 2.8ms，**warmup2 及全部真实请求 = 146~150ms**。

- 调度器在窗口内只跑了 1 个 schedule 周期——引擎没有空转；
- 窗口内引擎核心栈在 `_process_input_queue > wait`（py-spy V3c 轮，
  虽采样窗口未精确覆盖测量点，但空闲态确认）；
- 疑似：请求完成后引擎核心循环进入空闲等待，完成处理（finish→
  connector 通知）等某个 ~150ms 量级的唤醒。旁证：紧贴上一请求的
  warmup1 只要 2.8ms（循环还醒着）。
- **若修复，TTFT 预期 ~315ms → ~165ms。** 这是当前已知最大的单项余量。
- 下一步线索：`vllm/v1/engine/core.py` 的空闲等待/输入队列唤醒机制。

## ⚠️ 作废声明

本轮分析中我曾报告"排上→执行存在 ~110ms 空等"——**该结论作废**：
打点每个 decode step 重复触发，我的统计误用了后写覆盖（last-wins），
把"准入时刻"与"末步执行时刻"错配。以 first-wins 重算后，D 侧
排上→首 token 为健康的 ~13ms。教训与 ttft-timing skill 已记录的
"必须 first-wins" 一致，本次系我重犯。

## 归档

- 原始档：`results/raw/V3_seed123.txt`；日志 `/tmp/ttft_exp/V3*`
  （未入库：V3b/V3c 日志与 V3c py-spy json 在 /tmp，如需复核请尽快拷出）
- 工具：`scripts/dissect_gather.py`（gather 轮分段）
