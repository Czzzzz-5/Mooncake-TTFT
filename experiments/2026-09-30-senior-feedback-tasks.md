# 学长反馈处置轮（进行中）

开始：2026-09-30　操作人：wuzichun　**本卡暂不入库（本地记录）**

## 任务来源（学长反馈六点）

1. 正确性压测（p50/p99）；gather 在高并发下疑似 KV 静默污染（输出胡话）→
   先复现；实验 prompt 从 ~2.5k token 升到 100k token
2. wire ~150ms 与 首token→request_finished ~150ms 的双 150ms 疑点：
   单机（无 PD）对照，有则归档不追究，无则深挖
3. 社区是否有人做类似工作 → 已查（见下）
4. 拓扑从 1P1D 扩到 2P1D
5. vLLM 贡献不了 → 优化如何落回 vLLM / 或做成 Mooncake 侧形态
6. （认知澄清）v2a 发生在哪层

## 社区核查结论（09-30）

无人做"TCP GPU 路径 staging 池 / PD KV gather 合批"。最近邻：RFC #4242
（09-20，open）——RDMA+Store 场景的 owner 侧 gather + staging，与我们
同思路不同层。**我们的 TCP+PD 数据可作其证据，是贡献切入点。**

## 任务顺序与理由

| 序 | 任务 | 理由 |
|---|---|---|
| T1 | 复现污染（gather 开 × 并发） | 正确性 > 一切；先确认 bug 真实存在 |
| T2 | 对照（gather 关 × 同压） | bug 的 A/B：锁凶 gather 而非 PD 本身 |
| T3 | 修复 + 回归 | 嫌疑：D 侧单 buffer 被并发网络写入竞争 |
| T4 | 150ms 单机对照 | 便宜独立；`free_request` 打点已埋好 |
| T5 | p50/p99 压测 | 修复后压测才有意义 |
| T6 | 100k token | 需先解决 buffer 扩容/分片（512MB << 5.7GB） |
| T7 | 2P1D 拓扑 | 部署形态实验 |
| T8 | 社区贡献（RFC #4242 证据 / PR 化） | 收官 |

## 实验记录

（每完成一项，在此追加：日期、方法、数据、结论、证据链接）

### T4（完成，09-30）

**150ms 是 vLLM 引擎固有行为，归档不追究**（按学长定的判定规则）。
单机无 PD 对照（GPU0 单引擎，`free_request` 打点）：首 token→请求释放
= 149~155ms（5/6 请求；首个请求 40.6ms 属冷启动差异）。与 PD 场景的
~148ms 一致 → 与 PD/connector 无关。
附带发现：单机 TTFT（2670 tok prompt, max_tokens=1）~192ms，其中也含这
~150ms——该延迟在普通推理路径就存在（疑在输出处理/detokenize 节拍）。
wire ~150ms 是学长环境的数据；本机 wire 实测 54~56ms（2.7GB/s）。

### T8（进行中，10-09）：Mooncake TCP 并发模型调查 + 瓶颈解除

**调查结论**（explore 深读 mooncake-transfer-engine 源码，分支
gpu-staging-v1）——"单 TCP 排队"说法修正，实际三因素叠加：

1. **lanes 数 clamp 并发**：每 peer 默认 4 条 lane（连接），
   `MC_TCP_LANES_PER_PEER` 可调上限 16（tcp_transport.h:286）；
   第 5 个起的 transfer 在 per-peer FIFO 排队（lane_impl.h:733-752）。
2. **全引擎单 io 线程**：accept+全部连接读写+调度共 1 线程
   （tcp_transport.cpp:639-651）。
3. **chunk 严格串行无流水**：64KB/chunk，完成回调后才发下一个
   （session_impl.h:1037-1074）。4 lanes × per-chunk 事件开销
   ≈ 聚合 720MB/s，与 T4.5 实测吻合。
4. 附带发现（别人会踩）：115MB 请求超 staging pool 默认 16MB 槽时
   **静默落回逐 chunk 同步 cudaMemcpy**；本仓库 serve 脚本已设
   SLOT_MB=192 规避，gather 源端本就是 CPU 锁页内存，不踩。

**第一步（零代码验证）**：serve 脚本默认 `MC_TCP_LANES_PER_PEER=16`
+ `MC_TCP_SLICE_SIZE=1MB`，与 K=16 基准（p50 1532ms, seed=567）
同 prompt 对照。显著回升 → 诊断坐实；不回升 → 单 io 线程是硬顶，
需 C++ 多线程改造（改动点已定位：tcp_transport.cpp:393-401
io_context 加 worker 线程）。
后续代码方向（按侵入度排序）：多 io 线程 → lane 内多会话流水 →
chunk 滑动窗口 → staging pool 适配大请求（均为 T8 正式改动）。

### T4.5（10-03→10-09 完成）：定 K 实验 + P 侧槽裁剪

**背景**：T3 修复后 D 侧开 K=4 槽，但 P 侧也分配 K 槽只用槽 0（浪费
K-1 槽锁页内存）。实测拷贝:传输 = 1:14（D2H pinned 115MB≈2.0ms
@57.6GB/s vs Mooncake TCP push 115MB≈28.5ms @~4GB/s）→ 槽的使命是
**并发占位**而非拷贝/传输流水。定槽规则（Little 定律）：槽数 ≥ 期望并发
in-flight pull 数；超出的请求 addr=0 优雅降级（自动退回 v1 老路径）。

**代码改动（vllm-repo，未 commit）**：`_register_gather_buffer` 按角色
分配——consumer 开 K 槽（`VLLM_MOONCAKE_GATHER_SLOTS` 可调），
producer 固定 1 槽（其 gather 本就 `_gather_lock` 串行只用槽 0，
查证 mooncake_connector.py:1918-1920）；kv_both 按 consumer 算。

**定 K 实验设计（5 组）**：固定 2P1D + gather 开 + 16 并发 + 32 个不同
prompt（`load_test.py`，新写：mean/p50/p90/p99 + 吞吐 + epoch 打点）。
K ∈ {1,2,4,8,16}，每档重启双引擎（connector 改动需重启生效）。
判定标准：p99 不再下降且日志 `D_SEND_PULL slot=None`（降级）≈0 的
**最低 K**（每槽 512MiB 锁页，不浪费）。

**流式 scatter 评估**：边传边 H2D 理论上省拷贝 2ms（占 TTFT 0.6%），
但需 Mooncake C++ 引擎加进度回调 + 块对齐处理 → **现在不做**；
该思路即 T6 的 100k 分片流水设计（5.7GB KV、wire ~1.4s 时收益才显著）。

**K 扫描数据（2P1D，load_test 32请求/并发16/16k字符，seed=123）**：

| K | TTFT p50 | p99 | mean | gather 命中/降级 |
|---|---|---|---|---|
| 1 | 2062 ms | 3477 ms | 2270 ms | 5/29（85% 降级） |
| 2 | 2068 ms | 3029 ms | 2108 ms | 7/27（79% 降级） |
| 4 | 2032 ms | 3020 ms | 2066 ms | 11/24（69% 降级） |
| 8 | 1694 ms | 2365 ms | 1570 ms | 24/35（31% 降级） |
| 16 | 1532 ms | 2103 ms | 1572 ms | 35/35（0% 降级） |

**逐档观察**：
- K=1：85% 请求降级走逐描述符老路，p50 比 gather 快路慢 ~6 倍
  （描述符 4228 vs 364 的往返开销在并发下叠加）。
- K=2：p50 与 K=1 几乎持平——低 K 时瓶颈不是 gather 串行化，而是降级
  请求挤爆逐描述符路径，**K 低于并发数时收益被降级路径整体锁死，
  不存在"2 槽够用"的中间态**。
- K=4 dissect 实锤（XDBG 相位拆解）：gather 命中 pull 的
  D_SEND_PULL→D_RESP_RECV 仅 **15.9ms**；降级 pull 同相位 p50=
  **1803ms**（单描述符 ~0.4ms 开销 × 4228 个，单 peer 连接串行）。
- K=8：降级率降到 31% 后 gather pull 自身也变慢（相位 mean 1211ms，
  空闲时 16ms）；负载尾批请求 TTFT 363-761ms → 安静时 gather 全链路
  ~360ms 是真实快路径。瓶颈转移为 **P→D 单条 TCP 连接 per-peer
  串行排队**（3.7GB / 5.1s ≈ 720MB/s 聚合，远低于单流 bench 4GB/s）。

**定 K 结论（10-09，五档齐）**：选 **K=16**（降级率≈0 的最低档，
16 槽全部真实轮转，内存 8GiB 无压力——机器可用 600GB）。收益递减：
K=16 仅比 K=8 快 ~160ms——降级清零后瓶颈转移为 P→D 单条 TCP 连接
排队（35 路 gather 全挤 2 条连接，gather pull 相位 mean 1321ms vs
安静时 16ms）。**Python/connector 侧已调优到头，下一步须动 Mooncake
引擎（T8）**：多连接并发或单连接内 transfer 流水化。K 须随负载重定
（T6 100k 时单批容量变大，峰值并发 pull 数会涨）。
K=8 dissect 第二层发现：降级率降到 31% 后，**gather pull 自身也变慢**
（mean 1211ms，空闲时 16ms；负载尾批请求 req16-20 TTFT 仅 363-761ms
→ 安静时 gather 全链路 ~360ms 是真实快路径）。瓶颈转移为
**P→D 单条 TCP 连接排队**：32×115MB≈3.7GB / wall 5.1s ≈ 720MB/s
聚合（远低于单流 bench 的 4GB/s），所有 transfer 在 2 条连接
（P1→D、P2→D）上 per-peer 串行排队。这是 K 之外的新瓶颈，
记入 T8（Mooncake C++ 侧多连接/流水化才有解）。

### T3（完成，09-30）

**修复：D 侧单 buffer → K 槽轮转**（`VLLM_MOONCAKE_GATHER_SLOTS=4`，每槽
512MiB）。每次 pull 领一个空闲槽、槽地址随 pull 元数据告诉 P；scatter 完
还槽（finally 兜底）；无空闲槽则广播 addr=0，P 自动回退逐描述符路径
（优雅降级不死锁）。P 侧零改动（发送仍用槽 0 + 原锁串行）。

**回归（2P1D × 并发16 × 不同 prompt，修复后）**：5/16 措辞级差异（与 T2
对照组背景噪声同质同量级），**乱码零例 → 污染消除**。槽证据：D 侧
`D_SEND_PULL slot=` 分布 0/1/2/3 槽都被真实使用（slot1 复用 17 次）。

代码：vllm-repo 本地 commit（connector + envs）。已知小浪费：P 侧也分配
K 槽但只用槽 0（角色单一时多占 K-1 槽锁页内存，后续可按角色裁剪）。

### T2（完成，09-30）

**对照干净，锅锁定 gather**。同 2P1D 拓扑、gather 关（v1 槽池路径）、同压：
4/16 轻微措辞差异（batch 数值噪声，v1/gather 共有的背景噪声，与污染无关），
**零乱码**。对比 T1 的 15/16 发散 + 2 例乱码 → 污染来自 gather 的 D 侧
单 buffer 并发竞争。（教训：复现脚本须区分"乱码级"与"措辞级"发散。）

### T1（完成，09-30）

**复现成功**。方法：`repro_kv_pollution.py`（v2：16 个互不相同的长 prompt 并发 +
串行参照逐题比对）。1P1D×并发4/16 均未复现（P 侧锁串行 + ~165ms 就绪延迟
无意间保护了竞争窗口；v1 脚本同 prompt 的缺陷：16 份 KV 内容相同，混写不可见）。
**2P1D × 并发16 × 不同 prompt：15/16 发散，其中 2 例彻底乱码**（`请以\n0<|<...`、
`1000+ less off[contains...`）——KV 静默污染实锤。时间线证据：双 P 的 wire
窗口真实交叠（如 P1 528.388-528.469 与 P2 528.361-528.431），同写 D 的单一
gather buffer（无锁保护）。
