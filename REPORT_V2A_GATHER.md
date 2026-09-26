# v2a gather 合批优化报告

日期：2026-09-22 ~ 09-23　状态：**已验证上线**（开关默认关，`VLLM_MOONCAKE_GATHER=1` 启用）

## 一句话

KV cache 传输从"按显存碎片逐段传（最多 4228 次）"改为"GPU 一次收拢 →
内存整发一次 → 对端一次打散"，TTFT **−31ms / −9%**（三轮 A/B 焊死），
并彻底消除困扰多日的 TTFT 奇偶双模。

## 问题

PD 分离中，一个请求的 KV（~148MB）散在 ~162 个显存 block 里。旧路径按
物理连续段传输：快档 364 段 / 慢档 4228 段，每段一次"借锁页槽→拷贝→
发送→还槽"。实测拷贝本身只要 5ms，**逐段手续费才是大头**（慢档传输
169ms vs 纯净地板 ~39ms），且碎片程度按请求奇偶交替 → TTFT 双模
（~300/~410ms）。

## 方案

```
P：散块 ─[torch gather：一次索引拷贝收拢，实测 ~0.4ms]→ 预分配锁页 buffer
   ─[Mooncake CPU 内存路径，1 个描述符整发，54~56ms]→
D：锁页 buffer ─[torch scatter 打回散块，~6ms]→ GPU → 标记 KV 就绪
```

关键决策：放在 **vLLM connector（Python）** 而非 Mooncake（C++）——
connector 两头自控、走现成 CPU 路径（实测比槽池路径还快）、零 C++ 编译。
传输迭代 4228→1，对碎片程度免疫（不解 allocator 的根因，直接绕过）。

## 结果（V2 轮，三轮 A/B，seeds 777/888/123，4×16000 字符/轮）

| seed | v1 (ms) | gather (ms) | Δ |
|---|---|---|---|
| 777 | 343.10 | 316.85 | −26.3 |
| 888 | 355.27 | 320.27 | −35.0 |
| 123 | 345.59 | 314.14 | −31.5 |

均值 −30.9ms / −8.9%。形态：v1 散布 130ms（双模）→ gather 散布 23ms（单峰）。

生效证据：全部传输 `descs=1`、双端 gather buffer 注册、19/19 笔无回退无失败、
张量级往返自检通过。累计：406.63ms（接入）→ v1 −54ms → v2a −31ms ≈ **315ms（−22%）**。

## 遗留余量（按大小排）

1. **~148ms：P 侧完成通知延迟**（V3 轮定位，未修复）——请求生成完首 token 后，
   引擎要等 ~148ms 才调用 `request_finished`（发送触发点），疑似引擎空闲
   唤醒机制问题。修复后 TTFT 有望 ~165ms。详见
   `experiments/2026-09-23-v3-budget-closure.md`。
2. ~18ms：wire 2.7GB/s vs bench 地板 4.03GB/s（引擎内 CPU 路径写块调度）。
3. ~6ms→3ms：scatter 合并层操作。
4. v2b 多槽流水：参数已测定（SLOT_MB=32/SLOTS=4），但 KV 走 CPU 路径后
   对 KV 无增益，降级为可选。

## 代码位置

- 实现：`vllm-repo/.../mooncake/mooncake_connector.py`（`_gather_and_send_blocks` /
  `_scatter_gathered_buffer` 等，本地 commit `7006576afd`），开关在 `envs.py`
- 实验卡：`experiments/2026-09-23-v2a-gather.md`（A/B）、
  `experiments/2026-09-23-v3-budget-closure.md`（瀑布拆解 + 148ms 发现）
- 工具：`scripts/gather_scatter_selftest.py`、`scripts/dissect_gather.py`
