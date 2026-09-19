# 实验卡：nsys+py-spy 纯 profiler 拆解 TTFT ④ 段(NS1 轮)

日期：2026-09-19　操作人：wuzichun（测量命令本人执行）

## 目的与方法

不用手工打点（上个方法在 09-17 因跨时钟配对出错），改用外挂式 profiler：
- 双引擎 `nsys profile -t cuda,nvtx,osrt` 包裹启动
- decode 引擎外套 `py-spy record --subprocesses`（因 `ptrace_scope=1` 无法事后 attach，只能让引擎做 py-spy 的子进程）
- 用户跑 `measure_ttft_serving.py --num-prompts 4 --target-chars 16000 --max-tokens 10 --seed 123`

## 客户端结果（含 profiler 开销）

req: 357.98 / 491.28 / 353.85 / 477.62 ms，avg 420.18ms
（比无 profiler 的 X2 轮 328ms 膨胀 ~90ms：osrt 每 syscall 插桩，每请求 ~2.4 万次 send;
绝对值不可用，结构可用。双模交替依旧：快/慢/快/慢。)

## 核心发现（wire 级，epoch 精确）

**④ 段三段拆解**（P显存→P内存 → TCP → D内存→D显存）:

| 段 | 耗时 | 依据 |
|---|---|---|
| ① P GPU→锁页拷贝 | **~2-3ms** | took(70/197ms) − 网络窗口(69/194ms) ≈ 2-3ms;v1 一次异步大拷贝已把这段压没 |
| ② TCP 传输（50MB) | **69.1 / 194.1 / 71.3 / 194.1 ms** | 发送线程 send syscall 簇起止；双模严格按请求奇偶交替 |
| ③ D 锁页→GPU 拷贝 | ~3ms（代码级推断） | `stageHostToDevice` 收完后一次异步拷贝+event 同步（gpu_staging_pool.h:139) |

**② 段慢档的病征**：慢档 syscall 数是快档 3 倍（8700 vs 2800，平均写大小 5.7KB vs 18KB);
慢档中发送线程 125ms/194ms 在空等（socket 缓冲满）、接收线程也在等——
**TCP 流控停滞，双方 CPU 都不忙**。

**已排除的假设**：
- ~~每请求新建 TCP 连接/慢启动~~：连接在 warmup 时建好 4-5 条，4 个请求全部复用，无新连接
- ~~①③ 拷贝耗时~~：均为毫秒级
- ~~D 侧接收 CPU 不足~~：接收线程大部分时间在等数据而非干活

**传输后残差**（X2 轮用修正偏移 1789386860334.9 重算，发送完→D 排上）:
68.4 / 149.9 / 72.7 / 145.4 ms——**同样按奇偶交替**，与 ② 段同相位。
③ 拷贝只有 ~3ms，所以这 68/150ms 是"D 端得知传完→调度排上"的引擎侧反应。

**双模全貌**：传输（②）和调度反应（残差）双双按请求奇偶交替，疑似同一个周期-2 机制
（怀疑方向：decode 引擎等待期 dummy batch 循环的节奏干扰、staging 槽位或连接对状态）。
X2 轮 P_FIN−首token ≈ 149ms 恒定，也暗示存在 ~150ms 量级的周期性处理节拍。

## 工具链结论（给学弟妹的踩坑记录）

- nsys 在本机**采不到 GPU 活动**(CUPTI_ACTIVITY_KIND_MEMCPY/KERNEL 表为空，
  CPU sampling 也被禁）——GPU 侧拷贝只能靠代码级推断或 torch profiler 另试
- nsys osrt 时间戳基准是**会话相对值**，需加 TARGET_INFO_SESSION_START_TIME 换算 epoch
- osrt 的 returnValue 全 0、argumentsId 无效——拿不到 fd/字节数，只能用时序密度
- py-spy 在 ptrace_scope=1 下必须作为父进程启动（`py-spy record --subprocesses -- <cmd>`)
- py-spy svg 是全程聚合，测窗口行为应该用 `--format speedscope`（下次）

## 下一步

1. **传输层标尺实验**:`mooncake_tcp_bench.py --buf gpu`，引擎不在场时纯传输是否也双模
   → 区分"传输层自身问题" vs "引擎干扰传输"
2. **decode 等待期行为**:py-spy `--format speedscope` 只对 decode、只采测量窗口 120s，
   看 dummy batch 循环节奏是否就是残差 68/150ms 的来源
3. 原始档：`results/raw/NS1_*`（nsys-rep 太大不入库，sqlite 不入库，仅留日志与火焰图）
