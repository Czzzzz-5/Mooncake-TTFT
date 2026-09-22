# ROADMAP

## 已完成

- [x] **Mooncake TCP 接入**（09-06）：官方 MooncakeConnector，P2P 直连，
      TTFT 718.71 → **406.63 ms**，传输 ~123ms / 935 MB/s。踩坑记录：
      描述符上限需 `MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535`；
      官方 proxy 才透传 `kv_transfer_params`。
- [x] **带宽归因**（09-07）：定位 GPU 路径瓶颈在 staging 拷贝
      （每 64KB new + 同步 cudaMemcpy + delete，3520 次/传输），
      与单连接 TCP 5.5 GB/s 的标尺差距即优化空间。

## 已评估并放弃

- [x] **改法 A：`MC_TCP_SLICE_SIZE=1MB`** — CPU buffer 路径实测负面结果
      （单 io 线程被大包霸占，均值无收益、尾延迟恶化）；GPU 路径收益
      被 v1 完全覆盖。不再投入。（教训：零代码改法也要走完整实验流程，
      不能当"顺手一试"。）

## 主线：降传输段（首 token → D 排上 KV，TTFT 最大头）

- [x] **v1 锁页池 + 异步拷贝**（09-14/15 代码完成，**09-16 三轮 A/B 验证焊死：
      −54ms / −13%，seeds 777/888/123**）：Mooncake fork 新增
      `gpu_staging_pool.h`，预分配 N 槽锁页内存，socket 直接对接锁页槽。
      当前配置 `SLOT_MB=192, SLOTS=4`。
      ⚠️ 两个已查明的坑：① 09-15 场次 −45ms 归因**作废**（16MB 槽 < 115MB
      传输 → 静默回退 legacy，且基线场有第三方负载）；② TP1（09-19）发现
      "整传输一次大拷贝"的形态未实现——实际按描述符 364~4228 次小异步拷贝，
      收益真实但形态不同，碎片即双模真凶（v2a 的靶子）。

- [ ] **v2a gather 合批（主战场，09-22 设计）**

      **问题**：KV 是 ~162 个散落 block，connector 只能合并物理连续段
      → 快档 364 / 慢档 4228 个描述符，逐段"借槽→拷→发→还"（TP1）。
      拷贝本身只要 5ms，贵在每段一次的传输手续；慢档 169ms vs R1 实测
      bench 地板 38.9ms，差距全在手续费。碎片根因在 block allocator，难解
      → 不治碎片，让传输对碎片免疫。

      **做法（数据流）**：
      ```
      P：散块 ─[torch gather 按 block 索引收拢，~3ms]→ 预分配锁页 buffer
         ─[Mooncake CPU buffer 路径，1 个描述符整发，28.5ms/115MB]→
      D：锁页 buffer ─[torch scatter 打回散块，~3ms]→ GPU → 标记 KV 就绪
      ```
      传输迭代 4228→1，双模抹平。预期传输段 62~169ms → **~43ms**。

      **摆放决策：放 vLLM connector（Python），不放 Mooncake（C++）**。
      Mooncake 侧只能收拢 P 侧源地址、D 侧目标仍散，要扩协议带段表；
      connector 侧两头自控、走现成 CPU 路径（R1 实测比 staging 路径还快）、
      零 C++ 编译。旧判断"staging 上移到 connector 会多一层拷贝"在此
      **不成立**：gather 拷贝替代逐段 staging 拷贝，不是新增（原"改法 C"
      由此并入本项，见下）。

      **实现要点**：
      1. 双端各预分配 ~2×200MB 锁页 buffer（`torch.empty(pin_memory=True)`）
         并 `register_memory`；D 侧基址经现有 agent 元数据交换透露给 P
      2. gather/scatter 按现有描述符的同一份 block 配对顺序
         （P 侧在 `_send_blocks` 前，D 侧在收完回调后、标记 KV 就绪前），
         布局天然对齐
      3. 首版逐字节校验（抽 block 对比 hash），开关 `VLLM_MOONCAKE_GATHER=0`
         可回退 v1

      **验证（V2 轮）**：单请求输出 token 逐字一致 + KV hash 抽查 →
      三轮 A/B（A=v1 现状 4×192MB，B=gather，seeds 777/888/123），
      判定标准：双模消失 + 均值显著下降。生效证据：XDBG `P_SEND_EXEC`
      日志 descs=1。

- [ ] **v2b 多槽流水（降级为可选配套，Mooncake fork 侧）**：传输切片 +
      多槽循环复用，拷第 i+1 片与发第 i 片重叠。设计参数（R1 实测）：
      拷贝 57GB/s vs TCP 4.1GB/s（32MB 起平台），t_copy:t_send≈1:7 →
      SLOT_MB=32、SLOTS=4。注意：v2a 走 CPU 路径后 KV 不经过槽池，
      v2b 对 KV 无增益，仅服务其他 GPU 直传场景；预期收益仅 ~4ms/115MB。
      正确性三件套：槽生命周期 / event 排序（cudaStreamPerThread 隐患
      待验证）/ v2 ack 须在 H2D 落显存后。

- [ ] **v3（重定义）**：gather 与 prefill 计算按层 overlap（算完一层收拢
      一层，不等全算完）。原"cudaMemcpyBatchAsync 压逐片提交开销"已被
      v2a 的 torch gather 覆盖，不再单列。

## 备选

- ~~改法 C：staging 上移至 vLLM connector~~ — **已并入 v2a**（09-22）：
  旧顾虑"多一层拷贝"经 R1/TP1 数据证伪，connector 侧收拢即 v2a 本体。

## 代码资产

- Mooncake fork 分支 `gpu-staging-v1`：`gpu_staging_pool.h`（新文件）+
  `tcp_transport_session_impl.h`（4 处 GPU 分支）。
- 测量工具：`scripts/pinned_copy_bench.py`（纯拷贝带宽）、
  `scripts/mooncake_tcp_bench.py`（TCP/staging 全程）、
  `scripts/dissect_epoch.py`（epoch 打点解剖）。
- 工具链文档：`PROFILING.md`（nsys/torch profiler/py-spy 用法与坑）。

## 远期（需要硬件条件）

- GPUDirect / RDMA：网卡直通显存，把 2 次拷贝降为 0 次。当前环境无
  RDMA 网卡，不在近期路线。
