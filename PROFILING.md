# Profiler 工具链报告：nsys + torch profiler 怎么用来监测 PD 传输

日期：2026-09-19　整理：2026-09-22

## 为什么上 profiler

09-17 手工 epoch 打点出过定标事故。改用外挂式 profiler：
不改业务代码逻辑，直接看系统时间线。**纪律：结构分析用 profiler 轮，
报数用裸跑轮，两者严格分开**（profiler 税：nsys +90ms/req，torch +184ms/req，
绝对值一律作废）。

## nsys：系统级时间线（syscall / 传输结构）

```bash
nsys profile -o /tmp/ttft_exp/<轮次>_prefill -t cuda,nvtx,osrt \
  bash ~/pd-kv-transfer/scripts/serve_pd_mooncake.sh prefill   # decode 同理
# 跑完 ctrl+c 或 nsys shutdown，产 .nsys-rep；可转 sqlite 用 SQL 查询
```

- `-t osrt`：每次 send/recv syscall 的时刻和耗时——TCP 传输段（②段）直接可读，
  NS1 轮靠它测出 69/194ms 双模交替、慢档 syscall 数 3 倍、发送方 60% 空等
- `-t nvtx`：配合 `torch.cuda.nvtx.range_push/pop` 把传输调用框在时间线上
- `-t cuda`：理论上给 cudaMemcpy 时刻——**本机采不到**（CUPTI 表为空），
  GPU 侧只能靠 torch profiler

**本机踩坑**：osrt 时间戳是会话相对值，要加 `TARGET_INFO_SESSION_START_TIME`
换算 epoch；osrt 拿不到 fd/字节数，只能用时序密度；nsys-rep/sqlite 太大不入库。

## torch profiler：GPU 拷贝实况（本机唯一 GPU 侧眼睛）

```bash
# 起服务时加（serve 脚本已支持 TP_DIR 环境变量）：
TP_DIR=/tmp/ttft_exp/tp bash ~/pd-kv-transfer/scripts/serve_pd_mooncake.sh prefill
# 测量窗口内触发：
curl -X POST localhost:8100/start_profile && <跑测量> && curl -X POST localhost:8100/stop_profile
# 产 chrome trace（.pt.trace.json.gz），看 kernel 与 DtoH/HtoD 拷贝次数/窗口
```

TP1 轮靠它**首次直接观测到 GPU 拷贝实况**：快档 364 次 vs 慢档 4228 次
DtoH 拷贝，把双模真凶钉死在传输描述符碎片化。税比 nsys 重（按算子/API 计税）。

## py-spy：CPU 线程在等什么（辅助）

本机 `ptrace_scope=1`，不能事后 attach，必须让引擎做 py-spy 的子进程：
`py-spy record --subprocesses -- <启动命令>`。下轮必须加 `--idle`
（等待态=无样本，本轮因此信息量低）和 `--format speedscope`（按窗口看）。

## 三者分工（一句话版）

| 工具 | 回答什么 | 不能回答什么 |
|---|---|---|
| nsys (osrt) | TCP 传输何时发收、停等在哪 | GPU 拷贝（本机 CUPTI 空） |
| torch profiler | GPU 拷贝次数/窗口/算子构成 | 跨进程业务时序 |
| epoch XDBG 打点 | 跨进程业务语义（pull 到达等） | 细粒度系统行为 |

结论冲突时以 profiler 时间线为准。NS1/TP1 详细数据见对应实验卡。
