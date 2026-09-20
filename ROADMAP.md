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

- [x] **v1 锁页池 + 单次异步拷贝**（09-14/15，代码完成）：新增
      `gpu_staging_pool.h`，预分配 N 槽锁页内存，整传输一次
      `cudaMemcpyAsync` 进出槽，socket 直接对接锁页槽。
      ⚠️ 09-15 场次的 −45ms 归因存疑（16MB 槽 < 115MB 传输 → 静默回退
      legacy，且基线场有第三方负载）。脚本已改 `SLOT_MB=192, SLOTS=4`，
      **待重跑验证**。
- [ ] **v2 多槽流水**：传输切片 + 多槽循环复用，拷第 i 片与发第 i-1 片
      重叠（发送端），收第 i 片与 H2D 第 i-1 片重叠（接收端）。
      预期 ~30ms/115MB，触到单连接 TCP 物理地板。
      设计参数（R1 轮 09-20 实测）：拷贝 57GB/s vs TCP 4.1GB/s（32MB 起平台），
      t_copy:t_send≈1:7 → **SLOT_MB=32、SLOTS=4、115MB 切 4 片**；
      实现顺序：先 gather/描述符合并（追 38.9ms bench 地板），再多槽流水
      （仅再省 ~4-6ms，二阶）。
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
