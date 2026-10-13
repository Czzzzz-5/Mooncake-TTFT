# vLLM 侧改动（patches/）

`vllm-mooncake-gpu-staging.patch`：vLLM connector 侧**当前全部改动**的
单文件全量 patch（110KB，14 个代码文件，+1980 行），基于上游 commit
`ba07e4a48f`（vLLM main，2026-09 中旬），已验证 `git apply --check` 干净应用。

```bash
git apply patches/vllm-mooncake-gpu-staging.patch
```

## 演进故事（本地 vllm-repo 的 11 个 commit，按序）

1. XDBG epoch 打点（TTFT 解剖插桩，实验工具）
2. v2a gather 合批本体（`VLLM_MOONCAKE_GATHER`）
3. 完成通知延迟探针
4. **KV 污染修复**：D 侧每 pull 一槽轮转（2P1D 高并发下单 buffer 竞争
   导致 KV 静默污染，复现见 `scripts/repro_kv_pollution.py`）
5. 按角色分配槽（producer 固定 1 槽，省锁页内存）
6. M1：vLLM 侧多槽 gather + async batch submit（已退役，见 9）
7. **M2a：引擎侧 gather**——connector 只建段表，调
   `engine.submit_gather_write`，gather/发送全在 Mooncake 引擎
8. 引擎路径生效后 producer 不再分配 vLLM 锁页槽（省 8GB）
9. M1 路径整体删除（vLLM 侧多槽设计退役）
10. **M3：D 侧对称下沉**——接收槽 + scatter 也进引擎，
    vLLM 两侧锁页占用归零
11. **M2b：拆条吃满多 lane**——`submit_gather_write(split=K)`，
    单请求传输从 1 条 TCP lane 撒到 K 条

最终形态：**vLLM connector 只剩协议薄层**（段表/manifest 构建、
调度、XDBG 打点）；staging 槽、gather/scatter 拷贝、网络传输全部在
Mooncake fork（分支 `gpu-staging-v1`）。

注：旧 patches 0001-0005（M1 时代）已删除——0001 误含 66MB
safetensors 二进制，且内容已被上述全量 patch 覆盖。
