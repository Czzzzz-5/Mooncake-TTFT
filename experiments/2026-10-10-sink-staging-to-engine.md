# 下沉方案设计：gather/staging 所有权从 vLLM connector 移到 Mooncake 引擎（10-10）

状态：**M1 已完成并出数（10-10）**　操作人：wuzichun

## 1. 动机（实证，不是拍脑袋）

T8 §5b 的 A/B 与相位拆解给出两个硬事实：

1. **P 侧串行链是主要瓶颈**：`_gather_lock` 下 8 个并发 pull/P 排队
   （排队段 640-760ms），锁内 holder 还要同步等 submitTransfer 完成
   （P_SEND_EXEC→DONE 69-83ms）。每 pull 串行 75-145ms。
2. **同步等待全在 Python binding 忙轮询**（`batchTransferSync` 纯 spin
   等 `getBatchTransferStatus`，transfer_engine_py.cpp:646-674），C++ 引擎
   本身是完全异步的——vLLM 用 sync API 是自选串行。

学长的定性判断被数据证实：拷贝/发送方向掌控在 vLLM 手里，就没法并行
多 buffer。本方案把 gather/staging/多 buffer 流水的所有权下沉到 Mooncake
引擎，vLLM 只传 **GPU 地址段表**，接口保持薄薄一层。

## 2. 关键代码事实（设计前提，explore 调查 10-10）

- **提交链**：connector `_send_blocks` →
  `engine.batch_transfer_sync_write(src_ptrs, dst_ptrs, lengths)`
  （mooncake_connector.py:1848）→ pybind `batchTransferSync`
  （transfer_engine_py.cpp:574）→ 每 entry 一个 `TransferRequest`
  （transport.h:60-75）→ `submitTransfer`。**同步 spin 在 binding 层**。
- **已有可复用骨架**：`ScatterTransferRange` + `submitScatter` /
  `transferScatter`（transfer_engine.h:137-180）已是"段表 + per-fragment
  完成回调 + async operation"形态，只是 source 假设单一连续 buffer；
  `Impl::build()`（transfer_engine.cpp:1194-1202）只需把
  `.source = base + offset` 推广成 per-segment 地址。
- **binding 已有 async 原语**：`batch_transfer_async_write` +
  `get_batch_transfer_status`（transfer_engine_py.cpp:1270-1315），
  零 C++ 改动即可用。
- **GpuStagingPool**（gpu_staging_pool.h）：槽=pinned buffer+独立
  non-blocking stream+event；acquire/release 有锁；stage 原语是"整请求
  一次拷贝 + event 同步阻塞"——**阻塞假设是下沉要改的核心**。
- **gather buffer 握手走 vLLM 自己的 ZMQ 控制面**（connector
  receive_kv_from_single_worker :2319-2373，metadata 里带
  gather_buffer_addr），不经 Mooncake metadata；段表不过 wire（源端本地
  信息），**P 侧下沉不需要协议改动**。
- **协议协商模式**（v2 先例：SegmentDesc.tcp_proto_version 能力位 +
  opcode 高位 flag）：D 侧若做"按段 scatter"，照搬即可。

## 3. 目标架构

```
现在（v2a，vLLM 掌管拷贝+发送）：
  P: 散块 ─[torch gather, _gather_lock 串行]→ 锁页槽0 ─[sync 轮询等完成]→ TCP
  D: TCP → 锁页槽 ─[torch scatter ~6ms]→ GPU

目标（引擎掌管 staging + 流水）：
  P: vLLM 只交段表 [{gpu_addr,len}×N] + 对端 buffer 基址
     引擎内：段表→槽打包调度器（每槽一 stream，串行 cudaMemcpyAsync
            天然流水，非阻塞 event 查询）→ 每槽一个 slice 异步发送
            （MC_TCP_IO_THREADS 并行 io）→ 槽释放挂在 slice 完成回调
  D: 字节收齐一段（per-fragment 回调）→ cudaMemcpyAsync H2D 该段
     （引擎线程内直接发起，不等整请求）
```

接口形态：新顶层 API（不改 `TransferRequest` 结构体——它的布局被
pybind 按值拷贝 / C API 镜像结构 / TransferTask 指针 / 各 transport
prepareTransfer 依赖，改它 ABI 风险大）：

- C++：`submitGather(hostname, gpu_segments, remote_addr, slot_hint, ...)`
  → 返回 async operation（poll/abort/per-slot-complete 回调），骨架照
  ScatterTransferOperation。
- pybind：新方法 `submit_gather_write(...)`；现有绑定全部不动。

## 4. 分期实施（每期独立可验证，每期都 A/B）

### M1（零 C++，只动 connector，半天）：消灭 sync 轮询 + gather 并发化

- `_gather_and_send_blocks` 改用 `batch_transfer_async_write`，完成
  靠状态查询/回调，锁内不再 spin 等 wire；
- P 侧 gather 槽从"1 槽 + `_gather_lock`"改成"多槽 + 每槽一把锁"
  （slot 轮转代码 D 侧已有，搬过来）——K 个 pull 真并行 gather。
- 预期：吃掉锁串行的等待部分（排队 640-760ms 的大头）。
- 风险：近零。回滚 = 还原一个函数。

### M2（C++ P 侧下沉，主力，2-3 天）：gather + 流水进引擎

- 新 `submitGather` API + 槽打包调度器（段表 → N 槽，块对齐/碎片处理）；
- stage 原语改非阻塞（record event → 发送前 query 就绪队列）；
- vLLM 侧删 gather 槽池代码，`_gather_and_send_blocks` 缩成一次 API 调用。
- 预期：P 侧每 pull 占用从 75-145ms 降到 ~10ms 级（gather 3ms + 分摊），
  TTFT p50 从 ~1567ms 有望回落 300-600ms（按 §1 排队段估算）。

### M3（C++ D 侧下沉，可选，1-2 天）：scatter + 流式 H2D

- per-fragment 完成回调（ScatterTransferOperation 骨架）→ 该段字节到齐
  即 cudaMemcpyAsync H2D；
- 消掉 D 侧 scatter ~6ms + 整请求等待；为 T6（100k 分片）铺路。
- 需要协议能力位（tcp_gather_version）+ 老节点回退判断。

## 5. 对照设计（跑数时遵守实验流程规范）

- 每阶段同轮 A/B：A=旧路径（v2a），B=新路径；唯一变量是当期改动；
  2P1D 栈，GATHER 开，seed 至少 3 个（567/123/888），32 请求/并发 16；
- 口径：客户端 TTFT（mean/p50/p90/p99）；传输相位 D_SEND_PULL→
  D_RESP_RECV 用 dissect 脚本拆解复核；
- 生效证据 checklist（每期）：
  - [ ] P 侧日志出现新 API 调用计数 = pull 数（无静默回退 v1）
  - [ ] `D_SEND_PULL slot=None`（降级）≈0
  - [ ] 无 `all slots busy` / `exceeds slot capacity` WARNING
  - [ ] `pulling kv_caches finished` = warmup + 请求数，failed = 0
  - [ ] T1 式正确性抽检（16 并发不同 prompt，乱码零例）
  - [ ] 原始档入 results/raw/

## 6. 风险与开放问题

1. **段表传参开销**：~162 段/请求，pybind 传 list[tuple] 可接受；若成
   瓶颈改 tensor/指针传递。
2. **槽打包碎片化**：28 层块大小 × 槽容量（192MB）的装箱算法；最坏
   情况退化成一槽一段（拷贝次数=段数，仍优于现状）。
3. **回调线程做 CUDA**：D 侧 H2D 回调跑在引擎 io 线程 → cudaSetDevice +
   per-槽 stream；与 GpuStagingPool 现有 event 模型合并。
4. **正确性（T1 污染教训）**：槽调度必须保证同请求并发子传输写不同
   槽区间；M2 开发时先写 gather/scatter 往返 hash 自检
   （gather_scatter_selftest.py 已有）。
5. **MC_TCP_IO_THREADS 与流水的相互作用**：M2 后并行度 = 槽数 × io
   线程数 × lanes，回归时确认无新瓶颈/无过度订阅。
6. **N=16 单流回退**（T8 §5b）：M2 的槽流水正好绕开单 session chunk
   串行，预期一并治愈；验证时带单流 bench。

## 7. 与社区贡献的对接

- M2 的 `submitGather` 是引擎侧通用能力（任何"GPU 散块→TCP 对端连续
  buffer"场景可用），对上游是自包含新 API + 现有协商模式，PR 形态干净；
- 叙事对齐 RFC #4242（owner 侧 gather/staging）：TCP+PD 数据可作证据；
- vLLM 侧 M1 的改动是 connector 内部实现细节（async 调用替换 sync），
  上游可收性低但独立有用，可先行验证收益再定去向。

---

## 8. M1 实施记录（10-10 完成）

### 8.1 实现

只改 `mooncake_connector.py` 一个文件，三处：

1. `_register_gather_buffer`：P 侧也开 K 槽（去掉 `is_kv_consumer` 分支），
   日志确认 "16 x 512 MiB pinned"。
2. 旧 `_gather_and_send_blocks` 拆成两个函数：
   - `_gather_to_slot(gather_plan, req_order)`：同步，跑在 sender executor。
     `_gather_lock` 内只做 `_gather_free_slots.pop()`（锁只护空闲表）；
     gather 主循环、manifest、P_GATHER_DONE 打点原样搬移，写
     `self._gather_buffers[slot_idx]`，torch.cuda.synchronize() 后返回
     `(slot_idx, manifests, total_bytes)`；异常时还槽 + `_gather_broken=True`。
   - `_gather_and_send_blocks_async(...)`：先 executor 跑 gather，然后
     `engine.batch_transfer_async_write(...)` 拿 batch_id（0=失败），
     事件循环里每 1ms 轮询 `transfer_check_status(batch_id)`
     （1=完成已 free，-1=失败已 free，-2=timeout，0=进行中；300s 上限后
     放弃并 error——batch 无 free API 会泄漏，manifests 丢弃保证 D 不
     scatter 脏数据）；finally 还槽。XDBG 口径不变：提交打 P_SEND_EXEC
     descs=1，完成打 P_SEND_DONE ret=0。
3. 调用点（send 协程里）从 `run_in_executor(_gather_and_send_blocks)` 改成
   直接 `await self._gather_and_send_blocks_async(...)`；ret=None 时逐描述符
   fallback（`_send_blocks`）不动。

### 8.2 环境

2P1D（Qwen2.5-7B bf16，TCP，GPU0/1=P，GPU2=D），GATHER=1，
GATHER_SLOTS=16，512MiB/槽，LANES=16，SLICE=1MB，IO_THREADS=16（脚本默认）。
B 组=新代码；A 组=同轮 `git stash` 回旧代码重起栈（P 侧日志回退为
"1 x 512 MiB"，确认对照成立）。seed 567/123/888，32 请求/并发 16/16k 字符。

### 8.3 结果（客户端 TTFT，ms）

| 组 | seed | mean | p50 | p90 | p99 | min | max |
|----|------|------|-----|-----|-----|-----|-----|
| A 旧 | 567 | 1680.3 | 1637.9 | 2080.7 | 2198.1 | 543.5 | 2211.0 |
| A 旧 | 123 | 1637.9 | 1592.8 | 2073.7 | 2205.9 | 511.9 | 2223.4 |
| A 旧 | 888 | 1703.8 | 1666.0 | 2092.7 | 2208.5 | 526.6 | 2216.3 |
| B 新 | 567 | 1749.1 | 1758.6 | 1791.4 | 1805.5 | 1669.2 | 1806.1 |
| B 新 | 123 | 1823.0 | 1794.0 | 1911.6 | 1915.5 | 1591.1 | 1916.5 |
| B 新 | 888 | 1870.1 | 1860.5 | 1918.4 | 1932.0 | 1778.2 | 1936.4 |

p50 均值 A=1632.2 / B=1804.4（**B 慢 +172ms**）；p90 均值 A=2082.4 /
B=1907.1（**B 快 -175ms**）；分布形态从 A 的双峰（队头 540ms/队尾 2.2s）
变成 B 的窄带（1.6-1.9s）。

### 8.4 相位归因（XDBG FIFO 配对，每流 ~148MB）

- B 组 gather→submit（锁排队替代指标）p50=1.3/2.0ms（两 P）——
  旧代码实测 640-760ms 的 `_gather_lock` 排队**确认消失**；
- wire（P_SEND_EXEC→DONE）：A 组 p50=80.8/83.2ms（串行独占带宽，
  ~1.8GB/s）vs B 组 p50=167.4/179.2ms（并发流摊薄，~870MB/s/流）——
  单流传输时间 ×2.1，正好吃掉省下的排队时间还倒贴。

### 8.5 结论（修正 §1 的判断）

**M1 机制目标达成但端到端无收益**：锁排队是真瓶颈没错（确实消掉了），
但在当前负载（16 并发 × 148MB）下，并行传输摊薄单流 TCP 带宽，两者
相抵。P 侧锁已经不是端到端 TTFT 的净瓶颈——真正的墙是**传输聚合带宽**
（T8 lanes/chunk 线）。M2 如果只做"引擎内 gather+流水"，同样会撞这堵墙；
M2 的价值主张要改写：要么把 lanes/chunk/io 线程调优并进 M2 验证
（预期一并治愈 N=16 单流回退），要么先单独把聚合带宽提上去再谈下沉。

### 8.6 生效证据 checklist

- [x] P_SEND_EXEC 计数 = pull 数（103/103，两 P 合计，3 seed+冒烟）
- [x] 降级≈0：全轮 fallback 日志零条（no free slot / falling back /
      gather failed 均无）
- [x] 无 all slots busy / exceeds slot capacity WARNING
- [x] 192 压测请求 fail=0，冒烟中文连贯无乱码（T1 式抽检）
- [x] 原始档入 results/raw/M1/（summary.txt + 8 份日志 gz）

### 8.7 遗留

- 轮询上限 300s 后 batch 泄漏（binding 无 free API，error 日志已注明）；
- `dissect_gather.py` 假设事件不交错，M1 并发下会 KeyError——
  分析时用 FIFO 聚合口径（§8.4），脚本待适配多槽并发。
