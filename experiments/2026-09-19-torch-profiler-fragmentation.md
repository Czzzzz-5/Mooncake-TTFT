# 实验卡：TP1 轮 torch profiler 拆解 —— 双模真凶：传输描述符碎片化交替

日期：2026-09-19　操作人：wuzichun（测量命令本人执行）

## 设计

- 双引擎带 `--profiler-config '{"profiler":"torch",...}'` 重启（torch profiler 走 HTTP
  `/start_profile`/`/stop_profile` 触发；已验证本机可用，不受 nsys 的 GPU 权限限制）
- decode 引擎外套 py-spy speedscope 模式
- 测量：seed 123，4×16000 字符，max-tokens 10（同 X2/NS1 轮）

## 客户端结果

438.42 / 605.32 / 417.90 / 586.69 ms，avg 512.08ms。
（torch profiler 税比 nsys 更重：+184ms vs 裸跑 328ms。**绝对值作废，只用结构**。
profiler 税规律：nsys osrt 按 syscall 计税，torch profiler 按算子/API 计税——
结构分析用 profiler 轮，报数用裸跑轮，两者严格分开。)

## 核心发现：双模 = 传输描述符碎片化按请求奇偶交替

torch profiler GPU 时间线（首次直接观测到 GPU 拷贝实况）:

| 请求 | P侧 DtoH 拷贝次数 | D侧 HtoD 拷贝次数 | 拷贝窗口(P) | GPU 真正忙碌 |
|---|---|---|---|---|
| req0(快) | 364 | 365 | 55.7ms | 2.8ms |
| req1(慢) | **4228** | **4229** | 163.7ms | 5.0ms |
| req2(快) | 392 | 393 | 57.2ms | 2.8ms |
| req3(慢) | **4200** | **4201** | 159.9ms | 4.9ms |

四个请求 KV 总量几乎相同(~148MB/个)、块数几乎相同(162/162/161/160),
**但描述符（拷贝次数）按奇偶交替 364 vs 4228,差 12 倍**。
P 侧 took 日志镜像：62.7 / 169.0 / 62.1 / 164.3ms。D 侧拷贝簇完全镜像 P 侧。

**解读**：快档 ~364 个描述符（每 28 层约 13 段，每段 ~0.4MB,block 基本连续）;
慢档 ~4228 个描述符（每层 ~151 段≈逐块，每段 ~35KB,block 完全碎片化）。
碎片描述符 → 传输循环迭代次数 ×12、每次网络写变小 → TCP 频繁停等
（NS1 轮观测到的慢档 3 倍 syscall、发送方 60% 时间空等，全部对上）。

GPU 忙碌时间即使慢档也只有 5~13.7ms——**瓶颈从来不在 GPU 拷贝带宽**,
而在描述符数量驱动的 CPU/协议栈迭代开销。

## 修正/更新旧结论

1. v1 的"一次大拷贝"实际未发生：①段是按描述符的多次小异步拷贝（但都是异步+锁页,
   所以 v1 相对 legacy 的收益真实存在、不受影响）。
2. ③段实测修正：单机 50MB HtoD(锁页→显存)实测 ~10.9ms(torch profiler 空载实测),
   不是此前估计的 3ms；仍非瓶颈。
3. ②段"TCP 流控停滞"的机制落锤：不是链路问题，是描述符碎片化驱动的。

## 未解/待办

- **为什么 block 连续性按请求奇偶交替**：块数相同、拼法不同。待查 vLLM block
  allocator 的分配/释放顺序（P 侧还是 D 侧引入碎片，可用一条 DEBUG 日志打印
  连续段数确认）。
- py-spy speedscope 这轮无效：默认模式不采 idle 线程，等待态=无样本。
  下轮 py-spy 必须加 `--idle`。
- **定量化实验**：`mooncake_tcp_bench.py` 分别以 `--descs 364` 和 `--descs 4228`
  跑同样 115MB，直接测量碎片化的传输成本（无需引擎）。
- 修复方向候选：①描述符合并上限提高/跨块合并；②gather kernel 一次性把散块
  收拢进锁页槽再整发（4228 次拷贝→1 次 kernel+1 次传输）；这正好就是 v2 的形状。

## 归档

- trace: /tmp/ttft_exp/tp/*.pt.trace.json.gz（大文件不入库）
- py-spy: /tmp/ttft_exp/TP1_decode_cpu.json(speedscope,本轮因缺 --idle 信息量低)
- 日志：results/raw/TP1_*.log
