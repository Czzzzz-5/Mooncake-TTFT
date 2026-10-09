# PD 分离 serving 模式启动脚本（MooncakeConnector TCP 版）
# 拓扑: client -> proxy(8000) -> prefill(8100, GPU1) --Mooncake TCP--> decode(8200, GPU2)
# 对应实验组：复刻 ttft_report_serving_fp32_baseline.md §6 的 bf16 组（KV 原格式 bf16 直传）
# Mooncake TCP 默认单 peer 排队上限 ~2048 个 transfer，长 prompt（~4500 描述符）会 ret=-1 失败，
# 必须放宽（2026-09-05 repro_desc_limit.py 实测验证，详见 MOONCAKE_INTEGRATION_LOG.md §8）
export MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535
export MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535
# GpuStagingPool 槽位：vLLM 将整请求 28 层合并为单次 ~115MB 传输，槽必须 ≥128MB
# （2026-09-15 教训：默认 16MB 导致 acquire 静默回退 legacy，优化全程未生效，见 ttft_report_serving_mooncake_bf16_gpustaging.md）
export MC_TCP_GPU_STAGING_SLOT_MB=192
export MC_TCP_GPU_STAGING_SLOTS=4
# v2a gather 合批（B 组）：GATHER=1 bash serve_pd_mooncake.sh ... 启用；
# 双端必须同时开（D 端用 buffer 基址交换，P 端没收到广告会自动回退 v1）
if [ -n "$GATHER" ]; then
  export VLLM_MOONCAKE_GATHER=1
  export VLLM_MOONCAKE_GATHER_BUF_MB=${GATHER_BUF_MB:-512}
  # 定 K 实验结论（2026-10-09，五档 1/2/4/8/16）：K 须 ≥ 峰值并发 pull 数，
  # 16 并发下 K=16 才零降级（降级 = 逐描述符老路慢 100 倍）
  export VLLM_MOONCAKE_GATHER_SLOTS=${GATHER_SLOTS:-16}
fi
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=DEBUG  # 真实性核验用：decode 侧逐请求 "pulling kv_caches ... finished"
MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM="/home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/vllm"
PYTHON="/home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python"
REPO="/home/wuzichun/llm_serve_demo/vllm-repo"
MOONCAKE_CFG='{"kv_connector":"MooncakeConnector","kv_role":"ROLE","kv_connector_extra_config":{"mooncake_protocol":"tcp","device_name":""}}'

# 可选:设 TP_DIR 则开 torch profiler(HTTP /start_profile 触发),例:TP_DIR=/tmp/ttft_exp/tp
PROF_ARGS=()
if [ -n "$TP_DIR" ]; then
  mkdir -p "$TP_DIR"
  PROF_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TP_DIR\"}")
fi

case "$1" in
  prefill)
    CUDA_VISIBLE_DEVICES=1 VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 "$VLLM" serve "$MODEL" \
      --port 8100 \
      --enforce-eager \
      --gpu-memory-utilization 0.85 \
      --max-num-batched-tokens 32768 \
      --no-enable-prefix-caching \
      --kv-transfer-config "${MOONCAKE_CFG/ROLE/kv_producer}" \
      "${PROF_ARGS[@]}"
    ;;
  prefill2)
    # 第二个 P 实例（2P1D 拓扑 / 并发污染复现用）：GPU0、端口 8101、bootstrap 8999
    CUDA_VISIBLE_DEVICES=0 VLLM_MOONCAKE_BOOTSTRAP_PORT=8999 "$VLLM" serve "$MODEL" \
      --port 8101 \
      --enforce-eager \
      --gpu-memory-utilization 0.85 \
      --max-num-batched-tokens 32768 \
      --no-enable-prefix-caching \
      --kv-transfer-config "${MOONCAKE_CFG/ROLE/kv_producer}" \
      "${PROF_ARGS[@]}"
    ;;
  decode)
    CUDA_VISIBLE_DEVICES=2 "$VLLM" serve "$MODEL" \
      --port 8200 \
      --enforce-eager \
      --gpu-memory-utilization 0.85 \
      --no-enable-prefix-caching \
      --kv-transfer-config "${MOONCAKE_CFG/ROLE/kv_consumer}" \
      "${PROF_ARGS[@]}"
    ;;
  proxy)
    exec "$PYTHON" "$REPO/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py" \
      --prefill "http://127.0.0.1:8100" 8998 \
      --decode "http://127.0.0.1:8200" \
      --port 8000
    ;;
  proxy2p)
    # 2P1D 拓扑的 proxy：双 prefill 轮询
    exec "$PYTHON" "$REPO/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py" \
      --prefill "http://127.0.0.1:8100" 8998 \
      --prefill "http://127.0.0.1:8101" 8999 \
      --decode "http://127.0.0.1:8200" \
      --port 8000
    ;;
  *)
    echo "usage: $0 {prefill|decode|proxy}"
    exit 1
    ;;
esac
