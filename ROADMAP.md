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

## 主线：Mooncake 源码 patch（锁页 staging 池）

- [x] **v1 锁页池 + 异步拷贝**（09-14/15 代码完成，**09-16 三轮 A/B 验证焊死：
      −54ms / −13%，seeds 777/888/123**）：新增 `gpu_staging_pool.h`，预分配
      N 槽锁页内存，socket 直接对接锁页槽。当前配置 `SLOT_MB=192, SLOTS=4`。
      ⚠️ 两个已查明的坑：① 09-15 场次 −45ms 归因**作废**（16MB 槽 < 115MB
      传输 → 静默回退 legacy，且基线场有第三方负载）；② TP1（09-19）发现
      "整传输一次大拷贝"的形态未实现——实际按描述符 364~4228 次小异步拷贝，
      收益真实但形态不同，碎片即双模真凶（v2 gather 的靶子）。
- [ ] **v2a gather 合批**（主战场，设计文档 `V2_GATHER_DESIGN.md`，09-22）：
      P 侧 torch gather 收拢散块进锁页 buffer → CPU buffer 路径 1 个描述符
      整发（R1 实测 28.5ms/115MB，比 staging 路径还快）→ D 侧 scatter 回散块。
      传输迭代 4228→1，双模免疫，预期传输段 62~169ms → ~43ms。
- [ ] **v2b 多槽流水**（降级为可选配套）：传输切片 + 多槽循环复用。
      设计参数（R1 轮 09-20 实测）：拷贝 57GB/s vs TCP 4.1GB/s（32MB 起平台），
      t_copy:t_send≈1:7 → SLOT_MB=32、SLOTS=4。注意：gather 走 CPU 路径后
      KV 不经过槽池，v2b 对 KV 无增益，仅服务其他 GPU 直传场景。
      正确性三件套：槽生命周期 / event 排序（注意 cudaStreamPerThread
      隐患待验证）/ v2 ack 须在 H2D 落显存后。
- [ ] **v3**：`cudaMemcpyBatchAsync`（CUDA 12.8+）压掉逐片提交开销 +
      与 prefill 计算按层 overlap。

## 备选（仅当放弃维护 Mooncake fork 时启用）

- [ ] **改法 C：staging 上移至 vLLM connector** — vLLM 侧拷进锁页 buffer，
      Mooncake 走现成 CPU 内存路径。多一层拷贝，不动 Mooncake 源码。

## 代码资产

- Mooncake fork 分支 `gpu-staging-v1`：`gpu_staging_pool.h`（新文件）+
  `tcp_transport_session_impl.h`（4 处 GPU 分支）。

## 远期（需要硬件条件）

- GPUDirect / RDMA：网卡直通显存，把 2 次拷贝降为 0 次。当前环境无
  RDMA 网卡，不在近期路线。
