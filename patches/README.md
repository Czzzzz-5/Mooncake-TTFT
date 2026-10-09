# vLLM 侧改动 patch（对应本地 vllm-repo 的 5 个 commit）

基于 vllm 上游 commit `ba07e4a48f`（v0.x main，2026-09 中旬）。五个 patch 按序：

1. `0001` XDBG epoch 打点（TTFT 解剖用的插桩）
2. `0002` **v2a gather 合批本体**（mooncake connector，本项目的核心优化）
3. `0003` 完成通知延迟探针（V3 轮定位 148ms 用）
4. `0004` **KV 污染修复**：D 侧每 pull 一个槽的轮转（2P1D 高并发下
   单 buffer 竞争导致 KV 静默污染，复现见 repro_kv_pollution.py；
   无空闲槽则降级逐描述符路径，正确性不依赖 K 值）
5. `0005` 按角色分配槽：producer 固定 1 槽（省 (K-1)×512MiB 锁页内存；
   定 K 实验五档 1/2/4/8/16 → 16 并发选 K=16，见实验卡 T4.5）

查看：直接点开看 diff；应用：`git am <本目录>`。
注意：Czzzzz-5/vllm fork 在 GitHub 上未实际创建（remote 已配置但仓库不存在），
vLLM 侧改动以本目录 patch 为权威形态。
