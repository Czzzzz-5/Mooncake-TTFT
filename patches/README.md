# vLLM 侧改动 patch（对应本地 vllm-repo 的 3 个 commit）

基于 vllm 上游 commit `ba07e4a48f`（v0.x main，2026-09 中旬）。三个 patch 按序：

1. `0001` XDBG epoch 打点（TTFT 解剖用的插桩）
2. `0002` **v2a gather 合批本体**（mooncake connector，本项目的核心优化）
3. `0003` 完成通知延迟探针（V3 轮定位 148ms 用）

查看：直接点开看 diff；应用：`git am <本目录>`。
