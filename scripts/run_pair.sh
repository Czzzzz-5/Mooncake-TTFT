#!/bin/bash
# A/B 测速一轮：起 P（GPU1，后台）→ 等 15s → 起 D（GPU2，前台跑完自动退）→ 杀 P
# 用法: ./run_pair.sh <模型> <tag>
#   模型: k3 / dsv3 / qwen7b
#   tag:  main 或 20261026（决定数据写进 realshape_<模型>_<tag>.jsonl 和 /tmp/<tag>_{P,D}_<模型>.log）
# 跑之前确认 switch_mooncake.sh 已切到对应版本！
set -e
M=$1; TAG=$2
if [ -z "$M" ] || [ -z "$TAG" ]; then echo "用法: $0 <k3|dsv3|qwen7b> <main|20261026>"; exit 1; fi

# main 不认识 MC_TCP_GPU_STAGING_SLOT_MB，会静默忽略，所以两侧命令统一
ENV="MC_TCP_PHASE_TIMING=1 BENCH_TAG=$TAG MC_TCP_GPU_STAGING_SLOT_MB=192 MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=65535 MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=65535"
PY=/home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python
BENCH=/home/wuzichun/llm_serve_demo/my_pd_demo/my_realshape_bench.py

echo "===== P ($M, tag=$TAG) 后台启动，日志 /tmp/${TAG}_P_${M}.log ====="
env CUDA_VISIBLE_DEVICES=1 $ENV $PY $BENCH p --model $M > /tmp/${TAG}_P_${M}.log 2>&1 &
P_PID=$!
sleep 15

echo "===== D ($M, tag=$TAG) 前台开跑，日志 /tmp/${TAG}_D_${M}.log ====="
set +e
env CUDA_VISIBLE_DEVICES=2 $ENV $PY $BENCH d --model $M > /tmp/${TAG}_D_${M}.log 2>&1
D_RC=$?
set -e

kill $P_PID 2>/dev/null || true
if [ $D_RC -eq 0 ]; then
  echo "===== $M 完成 ====="
  tail -2 /tmp/${TAG}_D_${M}.log | head -1
else
  echo "!!!!! $M 的 D 侧失败（exit=$D_RC），看 /tmp/${TAG}_D_${M}.log !!!!!"
fi
exit $D_RC
