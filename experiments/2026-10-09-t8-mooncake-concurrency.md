# T8：Mooncake TCP 并发模型调查与多 io 线程优化（10-09）

状态：**进行中（多线程补丁有 bug，修到一半，见 §5）**　操作人：wuzichun

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

## 5. 当前 bug：N≥2 传输挂起（修到一半）

- 现象：N=1 正常（bench 4MB push ~3.2GB/s）；N≥2 必挂起，30s 超时
  ret=-1（"Sync batch data transfer timeout"）。vLLM serving 同复现
  （P_SEND_EXEC → 30s → ret=-1）。
- 根因方向：单线程下靠事件循环隐式串行的两处，多线程后会真并发：
  1. `runGroupPump`（lane_impl.h:658-873）：asio::post 的 pump 两个线程
     可并行执行，lane 认领/会话启动若未全程持锁会竞争；
  2. ClientSession/ServerSession 状态机（session_impl.h）：全双工时
     读写 handler 链可在不同线程并发碰共享状态（字节计数、v2 ack 配对、
     terminal 判定）。
- 已做的一半修复（未编译未验证，在工作区）：给 ClientSession 加
  `asio::strand`（构造取自 socket executor、cancel 改走 strand 投递）——
  方向正确但未完成：ServerSession 未加、pump 并发问题未查、未验证。

## 6. 现场状态（10-10 快照）

- 引擎进程全杀，GPU 干净；bench server 无残留。
- Mooncake 工作区：上述 3 文件补丁 + session_impl.h 半成品 strand 修改，
  全部未 commit。
- venv engine.so = 带 bug 的多线程版；**默认 MC_TCP_IO_THREADS 未设时
  行为与上游一致（=1），不影响现有实验**。

## 7. 接下来怎么干

1. **修完 strand 化**：审 session_impl.h 现有半成品，给 ServerSession 补
   同款 strand；核查 runGroupPump 的 lane 认领是否全程持锁（不够就补）；
   原则是加 strand 不加粗粒度互斥锁（避免热路径重新串行化）。
2. **编译 + bench 验证**：N=1 不退化（~3.2GB/s）、N=2/4 能传、N=16
   恢复且单流带宽不崩；跑并发场景（多 client 并行 push 16MB）看聚合
   带宽是否突破 ~800MB/s。
3. **vLLM A/B**：serve 脚本开 MC_TCP_IO_THREADS=16，2P1D 栈重跑
   load_test seed=567，对基准 p50=1532ms；预期传输相位 1321ms 显著回落。
4. **收尾**：commit 到 gpu-staging-v1（多线程补丁 + strand 修复，含
   bench 数据），push fork；serve 脚本与实验卡同步。
5. 之后回 T6（100k token 分片设计，传输引擎行为已被本卡摸清）和
   T7（2P1D 出数）。

## 附：本轮 bench/测量命令

```bash
# bench（server/client 两侧都要设相同 MC_TCP_IO_THREADS）
cd ~/pd-kv-transfer
MC_TCP_IO_THREADS=16 nohup /home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python \
  scripts/mooncake_tcp_bench.py server --buf cpu --addr-file /tmp/bench_addr.txt &
sleep 6
MC_TCP_IO_THREADS=16 timeout 60 /home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python \
  scripts/mooncake_tcp_bench.py client --buf cpu --dir push --sizes-mb 4 --iters 4 \
  --addr-file /tmp/bench_addr.txt

# vLLM 压测（栈由 serve_pd_mooncake.sh 起，GATHER=1）
bash ~/pd-kv-transfer/scripts/run_load_test.sh 567
```
