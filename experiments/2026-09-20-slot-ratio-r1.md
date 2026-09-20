# 实验卡：R1 槽数比测定 —— 拷贝段 vs TCP 段纯测量（v2 槽数/切片设计输入）

日期：2026-09-20　操作人：wuzichun（测量命令本人执行）

## 设计

不起引擎，把 staging 路径拆成三段分别纯测量，求 `t_copy : t_send` 时间比，
为 v2 流水的槽数与切片大小提供实测依据：

1. **纯拷贝**：`scripts/pinned_copy_bench.py`（本轮新增），GPU0，pinned D2H/H2D
   + pageable 参照，尺寸 4→115MB 扫描
2. **纯 TCP**：`mooncake_tcp_bench.py --buf cpu`（无 staging），同尺寸扫描
3. **交叉验证**：`mooncake_tcp_bench.py --buf gpu` + staging 环境变量（192MB×4），
   全程应 ≈ ①+②（串行）

## 环境

- 开跑前核查：三块卡全 ~2MiB、load ~0、无其他用户任务、无 vllm 进程（铁律 4 通过）
- bench 双进程同机回环（127.0.0.1），gpu 组双进程同占 GPU0
- 生效证据（第 3 步）：`GpuStagingPool: 4x 201326592 bytes pinned slots ready` ✓

## 结果

**① 纯拷贝**（`results/raw/R1_pinned_copy.txt`）：pinned 双向全线性 **~57 GB/s**
（4MB 小片也有 51 GB/s，per-copy 启动开销测不出）；115MB = D2H 2.00 / H2D 2.05ms。
pageable 参照仅 13~16 GB/s（115MB 要 7~8.7ms）——legacy 的 pageable 税量化。

**② 纯 TCP**（`results/raw/R1_tcp_cpu.txt`）：

| 尺寸 | 4MB | 8MB | 16MB | 32MB | 64MB | 115MB |
|---|---|---|---|---|---|---|
| GB/s | 3.20 | 3.33 | 3.39 | **4.09** | **4.11** | **4.03** |
| ms | 1.3 | 2.4 | 4.7 | 7.8 | 15.6 | 28.5 |

**平台起点 32MB（~4.1 GB/s）**；≤16MB 掉档 ~17%（每次传输固定开销 ~0.3ms）。

**③ staging 全程**（`results/raw/R1_staging_gpu.txt`）：115MB = 38.9ms（2.96 GB/s）；
32/64MB 档约 2.9~3.0 GB/s；≤16MB 与纯 TCP 持平（拷贝占比太小看不出来）。

## 分析

- **时间比**：115MB 时 t_copy(两端合计 4.05ms) : t_send(28.5ms) ≈ **1 : 7**
  （单端 1 : 14）。**拷贝远快于发送**，流水里发送是唯一瓶颈级。
- **槽数结论**：t_copy < t_send 的 regime 下，**2 个槽双缓冲即可让发送不空转**
  （发送槽 i 期间拷贝填槽 i+1，0.5ms 的活有 7.8ms 的窗口），第 3~4 个槽只是
  与计算争抢时的抖动余量。⛔ **旧说法"copy:send=1:3 故需 4 槽"作废**——从未落盘，
  且被实测方向性证伪（真实比值 1:7~1:14，且方向相反）。
- **切片结论**：拷贝侧对尺寸不敏感，瓶颈在 TCP 的 32MB 平台起点。
  **v2 切片候选 = 32MB 级**（115MB 拆 4×28.75MB），16MB 会白丢 ~17% TCP 带宽。
  推荐配置 `SLOT_MB=32, SLOTS=4`——锁页总量 128MB，比当前 4×192MB 省 6 倍。
- **交叉验证残差**：③ − (①+②) = 115MB 时 **+6.3ms**（32MB +1.8，64MB +3.9，
  ≤16MB ≈0）。隐含"从锁页槽 send"比"从 pageable send"慢 ~20%，或双进程同卡
  干扰，**未归因，待查**。若属实，v2 流水地板 ~35ms/115MB 而非 28.5ms。
- **战略含义**：v1 串行全程 bench 地板 38.9ms/115MB，流水化最多回收拷贝串行的
  ~4ms + 残差 ~6ms；而引擎内实测传输 62~169ms（TP1）远高于 38.9ms 地板——
  **描述符碎片化（364 vs 4228）才是与地板之间的大头**，v2 的 gather 合批是
  一阶收益，流水 overlap 是二阶收益。

## 结论

1. 槽数不由"1:3"决定：实测 2 槽够用、3~4 槽为抖动余量；当前 4×192MB  oversized。
2. v2 设计参数：**SLOT_MB=32、SLOTS=4、切片 4×28.75MB/115MB**。
3. v2 实现顺序：先 gather/描述符合并（追 38.9ms 地板），再多槽流水（再省 ~4-6ms）。
4. 待查：锁页槽 send 比 pageable 慢 ~20% 的残差（下轮可用 `--descs` 对照或
   perf 确认）。

## 归档

- 原始档：`results/raw/R1_{pinned_copy,tcp_cpu,staging_gpu}.txt`
- 测量脚本：`scripts/pinned_copy_bench.py`（新）；临时 wrapper 脚本已删
- 清场：bench 进程全杀，三块卡 ~2MiB ✓
