#!/bin/bash
# 定 K 压测包装：参数已固定，避免长命令复制出错
exec /home/wuzichun/llm_serve_demo/vllm-repo/.venv/bin/python \
  /home/wuzichun/pd-kv-transfer/scripts/load_test.py \
  --requests 32 --concurrency 16 --target-chars 16000 --max-tokens 32 --seed "$@"
