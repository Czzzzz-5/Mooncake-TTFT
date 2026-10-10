# T8：Mooncake TCP 并发模型调查与多 io 线程优化（10-09）

状态：**挂起 bug 已修复（10-10），A/B 出数：多 io 线程本负载中性，
瓶颈修正为 P 侧 gather 锁串行链（→ 下沉方案）**　操作人：wuzichun

## 1. 起因：T4.5 扫出来的瓶颈

定 K 实验（见 2026-09-30 任务卡 T4.5）把 D 侧 gather 槽开到 K=16 后，
降级率归零，但 TTFT p50 只从 1694ms（K=8）降到 1532ms，收益递减。
XDBG 相位拆解显示：35 路并发 gather 的传输相位（D_SEND_PULL→D_RESP_RECV）
mean 仍有 1321ms（安静时 16ms），聚合带宽 ~720MB/s（安静单流 4GB/s）。
结论：**Python/connector 侧已调优到头，瓶颈在 Mooncake 引擎内部**。

## 2. 源码调查结论（gpu-staging-v1 分支，含行号证据）

"单 TCP 排队"的说法修正，实际三因素：

1. **lanes 数 clamp 并发**：每 peer 默认 4 条 lane（连接），
   `MC_TCP_LANES_PER_PEER` 可调、上限 16（tcp_transport.h:286）；
   第 5 个起的 transfer 在 per-peer FIFO 排队（lane_impl.h:733-752）。
2. **全引擎单 io 线程**：accept + 全部连接读写 + 调度共 1 线程
   （tcp_transport.cpp:639-651）。
3. **chunk 严格串行无流水**：64KB/chunk，完成回调后才发下一个
   （session_impl.h:1037-1074）。
4. 附带发现：115MB 请求超 staging pool 默认 16MB 槽会**静默落回逐 chunk
   同步 cudaMemcpy**；本仓库 serve 脚本已设 SLOT_MB=192 规避。

## 3. 零代码验证：lanes/chunk 旋钮排除实验

serve 脚本默认改为 `MC_TCP_LANES_PER_PEER=16` + `MC_TCP_SLICE_SIZE=1MB`，
K=16 栈重跑 load_test（seed=567，与基准同 prompt）：

- 基准（4 lanes/64KB）：p50=1532ms，传输相位 mean 1321ms
- 16 lanes/1MB：p50=1638ms，传输相位 mean 1407ms

**无改善 → 排除 lanes 数与 chunk 大小，单 io 线程成为唯一嫌疑**
（聚合带宽被单核事件循环钉在 ~800MB/s）。

## 4. 多 io 线程补丁（已写、已编译、已部署）

设计：`MC_TCP_IO_THREADS=N`（默认 1=上游单线程行为）让共享 io_context
跑 N 个线程。改动 3 文件：

- `tcp_transport.h`：`thread_` → `threads_` 数组 + `io_threads_` 成员
- `tcp_transport.cpp`：env 解析（1–128）；install() 起 N 线程
  （0 号线程独享 doAccept，其余纯 run）；新增 `runWorker()`；
  启动 LOG 带线程数/lanes 数（验证生效用）
- `tcp_transport_lane_impl.h`：shutdown 处 join 全部线程

编译部署（2026-10-09）：build.sh 同款环境变量 `make -j32 engine` 增量编译
~2 分钟；拷入 venv 时踩坑——raw .so 依赖裸名 `libglog.so.1` 而 wheel 版
是改名 bundle，已在 `site-packages/mooncake/libglog.so.1` 建软链解决
（ engine.so.bak-20261009 为原 wheel 版备份）。

## 5. 挂起根因定位与修复（10-10 完成）

**根因（修正 10-09 的两条怀疑方向）：不是 session 状态机竞争，而是
io_context 空转死循环。** 证据链：

- 埋点（`MC_TCP_DEBUG=1` 门控，session_impl.h/lane_impl.h TCP_DBG）显示
  client 侧 ENQUEUE 之后再无 PUMP/LANE/连接事件；server 侧无任何连接到达；
  进度定时器从未触发——引擎事件循环整体停摆。
- 挂起进程两个 io 线程各 99.9% CPU（各攒 4 分钟 CPU 时间）= 忙等。
- `MC_TCP_PROTO=1`（v1 协议）同样挂 → 排除 v2 ack 竞态。

机制：`install()` 先起 runWorker 线程，thread 0 还没调 `doAccept()` 时
io_context 处于无 work 状态，runWorker 的 `run()` 立即返回；asio 语义里
`run()` 正常返回后 io_context 进入 stopped 态，**必须 `restart()` 才能再
跑，而 restart() 只在异常分支调用**——正常返回路径落入
`while(running_) run();` 空转，之后 post 的 handler 永远不会被执行。
N=1 时只有 worker() 一个线程、且 doAccept() 先于 run()，永远不触发。
两条" session 状态机竞态"方向在源码审查中均被排除（lane/group 状态机
全程持锁、pump 靠 pump_epoch 串行化、GpuStagingPool acquire/release
有锁、ServerSession 本就单链）。

**修复（mooncake 工作区，已编译已部署 venv）**：

1. `TcpContext` 增加 `executor_work_guard`（构造时 make_work_guard，
   shutdownConnectionLanes 在 stop() 前 reset()）——work  guards 保证
   run() 常驻 epoll_wait，空转路径不复存在；
2. strand 化保留并完成：ClientSession 半成品收尾（cancel 走
   post+兜底直调），ServerSession 四处 handler（sendStatus/readHeader/
   writeBody/readBody）补齐 strand——多线程下同一 session 的 handler
   链会真并发，这层保护仍然必要。

**冒烟（10-10，loopback bench）**：

- N=1 无回退：4MB push ~2.8GB/s（与修复前一致；中间一次 1.9GB/s 复跑
  消失，判为共享机噪声）
- N=2/4/16 传输恢复：修复前 30s `Sync batch data transfer timeout`
  ret=-1，修复后全部能传
- 闲时 CPU 0%（空转消失）
- 8 client 并行 ×16MB：每流 ~1.1-1.3GB/s，与单 client 基线持平
  （12.3ms/16MB）→ 聚合 ~8-9GB/s，突破单 io 线程 ~800MB/s 顶

## 5b. 正式 bench + vLLM A/B（10-10，wuzichun 代跑一轮，原始档 results/raw/T8_*）

**单流 bench（115MB，iters=8，cpu buf）**：

| N | push avg | pull avg |
|---|---|---|
| 1 | 3875 MB/s (29.9ms) | 3016 MB/s (38.1ms) |
| 16 | 1408 MB/s (81.9ms) | 1731 MB/s (66.9ms) |

单流随 N 增多反降（3875→1408），疑与多线程间 handler 弹跳/局部性有关；
**N 应按负载选，盲目开大伤单流**。

**8 并发聚合（8 client ×16MB）**：N=1 与 N=16 两档每流都 ~1.3-2.9GB/s
（8 peer 形状下 loopback 本就喂得饱）——**该形状不是瓶颈形状**。

**vLLM A/B（2P1D，GATHER=1，K=16，lanes=16/1MB，seed=567，32请求/并发16）**：

| 组 | MC_TCP_IO_THREADS | TTFT mean | p50 | p90 | p99 | fail |
|---|---|---|---|---|---|---|
| A | 1 | 1691 ms | 1567 ms | 2025 ms | 2128 ms | 0 |
| B | 16 | 1734 ms | 1645 ms | 2059 ms | 2185 ms | 0 |
| （10-09 基线） | 1 | 1572 ms | 1532 ms | — | 2103 ms | 0 |

A 组复现基线 ✅；B 组无收益（差值 ~4% 在共享机噪声内）。
传输相位 D_SEND_PULL→D_RESP_RECV：A mean 1441ms / B mean 1478ms
**对 io 线程数不敏感**。

**结论修正（重要）**：T4.5"单 io 线程是聚合带宽硬顶"的归因**不成立**
（至少在当前 lanes=16/1MB 配置下不成立）。A/B 唯一变量下传输相位不动，
~770MB/s 聚合顶（3.7GB/4.8s）另有其因。P 侧逐请求配对拆解（i-th 配对，
n=17-18/侧）：

- P_SEND_EXEC→P_SEND_DONE：**69-83ms**（115MB ≈1.6GB/s，同步等完成）
- P_READY→P_GATHER_DONE：**640-760ms**——gather 本体仅 ~3ms，此段是
  **`_gather_lock` 串行排队**（8 pull/P 挤一把锁，锁内 holder 占
  gather+cuda.synchronize+send ≈75-145ms）
- P_RECV_PULL→P_WAIT_WAKEUP：620-690ms（与上同源，排队前段）

**新瓶颈：P 侧 connector 的 `_gather_lock` + 同步 submitTransfer 串行链**
——正是学长说的"vLLM 掌管拷贝/发送就没法并行多 buffer"的实证，为
buffer/staging 下沉到 Mooncake 引擎（下沉方案）提供了最直接的动机。
注意 send 本体 115MB 仅 ~30ms（bench 单流 3875MB/s），锁内 70-83ms
还包含引擎内 chunk 串行，下沉时一并解。

## 6. 现场状态（10-10 快照）

- 引擎进程全杀，GPU 干净；bench server 无残留。
- Mooncake 工作区：上述 3 文件补丁 + session_impl.h 半成品 strand 修改，
  全部未 commit。
- venv engine.so = 修复后多线程版（work guard + 双端 strand）；
  **默认 MC_TCP_IO_THREADS 未设时行为与上游一致（=1），不影响现有实验**。
- 新增 `MC_TCP_DEBUG=1` 门控埋点（TCP_DBG），定位传输挂起用，默认关。

## 7. 接下来怎么干

1. ~~修完 strand 化~~ ✅ 10-10：挂起根因实为 io_context 空转（见 §5），
   work guard 修复 + 双端 strand 化完成，冒烟通过。
2. ~~正式 bench 出数~~ ✅ 10-10：见 §5b。**多 io 线程本负载无收益，瓶颈
   修正为 P 侧 `_gather_lock` + 同步 send 串行链**。
3. **下沉方案（新卡）**：把 gather/staging/buffer 所有权从 vLLM connector
   移到 Mooncake 引擎（学长方向 + §5b 实证动机）。第一步设计：引擎侧
   多 buffer + 异步流水提交，解开 P 侧串行链。
4. 之后回 T6（100k token 分片设计）和 T7（2P1D 出数）。

## 附：本轮 bench/测量命令

```bash
# 正式 bench（server/client 两侧都要设相同 MC_TCP_IO_THREADS；每个 N 档重启 server）
cd ~/pd-kv-transfer
MC_TCP_IO_THREADS=<N> nohup /home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python \
  scripts/mooncake_tcp_bench.py server --buf cpu --addr-file /tmp/bench_addr.txt &
sleep 8
MC_TCP_IO_THREADS=<N> timeout 120 /home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python \
  scripts/mooncake_tcp_bench.py client --buf cpu --dir push --sizes-mb 115 --iters 8 \
  --addr-file /tmp/bench_addr.txt

# 并发聚合（8 client 并行 ×16MB，server N=16）
for i in 1 2 3 4 5 6 7 8; do
  MC_TCP_IO_THREADS=16 timeout 120 /home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python \
    scripts/mooncake_tcp_bench.py client --buf cpu --dir push --sizes-mb 16 --iters 1 \
    --addr-file /tmp/bench_addr.txt > /tmp/bench_par_$i.log 2>&1 &
done
wait

# vLLM A/B（栈由 serve_pd_mooncake.sh 起，GATHER=1；A 组显式 MC_TCP_IO_THREADS=1）
MC_TCP_IO_THREADS=1 GATHER=1 bash ~/pd-kv-transfer/scripts/serve_pd_mooncake.sh prefill   # A 组
GATHER=1 bash ~/pd-kv-transfer/scripts/serve_pd_mooncake.sh prefill                        # B 组（默认16）
bash ~/pd-kv-transfer/scripts/run_load_test.sh 567
```
