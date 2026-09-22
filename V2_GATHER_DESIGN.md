# v2 gather 合批 详细设计方案

日期：2026-09-22　状态：待评审　前置数据：TP1（描述符碎片化 364/4228）、R1（分段带宽测定）

## 1. 问题定义（为什么是它）

一个请求的 KV（~148MB）在显存中是 ~162 个散落 block。connector 拼描述符时
只能合并**物理连续**的 block 组（`mooncake_connector.py` `_can_coalesce_block_transfers`），
结果快档 364 段 / 慢档 4228 段，逐段走"申请槽→拷→发→还→H2D flush"。

TP1 实测：4228 次拷贝 GPU 只忙 5ms——**拷贝本身不贵，贵的是每段一次的传输迭代手续**
（槽申请、event 同步、asio 调度、TCP 小包停等）。慢档 169ms vs bench 地板 38.9ms，
差距全在手续费。且碎片程度按请求奇偶交替（双模），根因在 block allocator，难解。

**思路：不治碎片，让传输对碎片免疫。** 把散块先收拢成连续内存，再少数几次大传输。

## 2. 数据流对比

```
现状（v1，P push，每段独立）：
  P: 散块 ─拷→ 槽 ─发→ ─拷→ 槽 ─发→ ... ×364~4228 次（每段：acquire+拷贝+发送+释放）
  D: 每段收进槽 → H2D flush → 释放，同样 ×364~4228 次

gather（本方案）：
  P: 散块 ──[1 次 gather：torch 按 block 索引收拢]──→ 锁页大 buffer（连续）
     ──[CPU buffer 路径，1 个描述符，~28.5ms/115MB（R1 实测 4.03GB/s）]──>
  D: 锁页大 buffer ──[1 次 scatter：torch index_copy 回散块]──→ GPU
```

传输迭代次数：4228 → 1。槽池（4×192MB）在 KV 路径上不再使用
（CPU buffer 路径本来就不需要它；池子保留，服务其他 GPU 直传场景）。

## 3. 关键决策：gather 放在 connector（Python/torch），不放 Mooncake（C++）

评审过两个摆放位置：

- ❌ **Mooncake 引擎内**（`TcpTransport::submitTransferTask` 能看到整批）：
  只能收拢 P 侧源地址；**D 侧目标地址仍是散的**，wire 格式要扩协议带段表，
  D 收到后还要按表打散——改动深、要动协议、C++ 重新编译调试慢。
- ✅ **vLLM connector 内**（本方案）：P 侧 gather 用 torch（vLLM 必带），
  D 侧 scatter 同理；走 Mooncake **现成的 CPU buffer 路径**，零协议改动、
  零 C++ 编译；R1 实测 CPU 路径（28.5ms/115MB）**比 staging 路径（38.9ms）还快**。
  ROADMAP 旧注"staging 上移会多一层拷贝"在此不成立：gather 拷贝**替代**了
  逐段 staging 拷贝，不是新增。

## 4. 预期收益（基于 R1/TP1 实测外推）

| | 现状快档 | 现状慢档 | gather 后（估） |
|---|---|---|---|
| 引擎内传输段 | ~62ms | ~169ms | gather ~3ms + wire ~37ms(148MB) + scatter ~3ms ≈ **43ms** |

双模消失（快慢档都收拢成一样的一块）。TTFT 层面预期 −20~−130ms 量级，
**显著大于多槽流水的 ~4ms**——所以先做 gather，流水降级为可选配套。

## 5. 实现要点

**P 侧（`mooncake_connector.py` sender 路径，`_send_blocks` 前）**：
1. 预分配锁页 buffer：`torch.empty(MAX_KV_BYTES, uint8, pin_memory=True)`，
   按并发度预建 2 个（~200MB×2），注册进 Mooncake（`register_memory`，
   CPU 路径）。
2. gather：按今天拼描述符的**同一份 block 配对顺序**，逐层
   `kv_cache[:, block_ids]` 索引 → 拷入 pinned buffer 对应偏移
   （non_blocking + 一次 synchronize 后再提交传输）。布局 = 现有描述符
   顺序的拼接，D 侧用同一顺序打散，天然对齐。
3. 传输：`batch_transfer_sync_write` 只发 1 个描述符
   （src=P pinned buf，dst=D pinned buf，len=总字节数）。

**D 侧（receiver 路径，收完回调处）**：
4. 预先同样注册 D 侧 pinned buffer，基址通过现有 agent 元数据交换透露给 P
   （复用现有 kv_region base_addr 交换机制，新增一个 region 条目）。
5. 收完（现有完成语义不变）后 scatter：按同一 block 配对顺序
   `index_copy` 回 D 的 GPU blocks，然后才标记 KV 就绪（保 "applied at
   destination" 语义）。

**校验**：首版逐字节核对——gather/scatter 前后各抽若干 block 与按旧路径
传输的结果比对（可用一条 DEBUG 配置开关跑一次全量 hash）。

## 6. 验证计划（V2 轮）

- L1 正确性：单请求 seed 固定，gather 路径 vs v1 路径输出 token 逐字一致
  + KV 抽查 hash 一致。
- L2 引擎 A/B（三轮 seed 777/888/123，同轮紧邻，测量命令用户敲）：
  A=v1（现状 4×192MB），B=gather。checklist：descs=1（XDBG P_SEND_EXEC
  日志直接可读）、pull finished 计数、无 WARNING。
- 预期判定：B 组双模消失（4 个请求 TTFT 方差收敛）且均值显著低于 A。

## 7. 风险与退路

- pinned buffer 并发冲突：同时两个请求 send → 第二个等第一个发完
  （现有负载是串行客户端，风险低；仍加排队保护）。
- block 布局假设错误 → 校验步骤兜底，失败回退 v1（开关
  `VLLM_MOONCAKE_GATHER=0`，默认 on/off 待实现时定）。
- 只覆盖 P→D KV 路径；metadata/其他传输不动。

## 8. 与既有计划的关系

- 多槽流水方案（2026-09-20 计划）：**降级为可选**。gather 走 CPU 路径后
  KV 不经过槽池，流水对 KV 无增益；池子与流水机制保留给其他场景。
- 本方案落地后 ROADMAP v2 条目更新为：gather（本方案）→（可选）流水。
