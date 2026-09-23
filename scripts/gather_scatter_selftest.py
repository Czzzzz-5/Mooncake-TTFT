#!/usr/bin/env python
"""gather/scatter 布局逻辑的张量级自检（非实验数据，纯代码验证）。

构造一个假 KV cache：[num_blocks, block_len] 字节视图，随机打乱 block 顺序，
走 _gather_blocks_to_buffer -> _scatter_buffer_to_blocks 往返，
验证：1) buffer 布局 = 按名单顺序的块拼接；2) scatter 后目标 cache 与源一致。
"""
import sys

import torch

sys.path.insert(0, "/home/wuzichun/llm_serve_demo/vllm-repo")
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _gather_blocks_to_buffer,
    _scatter_buffer_to_blocks,
)

torch.manual_seed(0)
NUM_BLOCKS, BLOCK_LEN = 32, 4096
PAYLOAD_OFF, PAYLOAD_LEN = 512, 2048  # 模拟 region 偏移（非 0 才测得到算术）

# 模拟 [2, num_blocks, block_size*heads*dim] 的连续 cache：直接展平成字节
src_cache = torch.randn(NUM_BLOCKS * BLOCK_LEN // 4, dtype=torch.float32, device="cuda")
dst_cache = torch.zeros_like(src_cache)

# 随机散块名单（模拟碎片化分配）
src_ids = torch.randperm(NUM_BLOCKS)[:17].tolist()
dst_ids = torch.randperm(NUM_BLOCKS)[:17].tolist()

buf = torch.empty(len(src_ids) * PAYLOAD_LEN, dtype=torch.uint8, pin_memory=True)

_gather_blocks_to_buffer(buf, 0, src_cache, src_ids, BLOCK_LEN, PAYLOAD_OFF, PAYLOAD_LEN)
torch.cuda.synchronize()

# 校验 1：buffer 布局 = 按 src_ids 顺序的块载荷拼接
src_bytes = src_cache.view(torch.uint8).view(-1, BLOCK_LEN)
for i, bid in enumerate(src_ids):
    expect = src_bytes[bid, PAYLOAD_OFF : PAYLOAD_OFF + PAYLOAD_LEN].cpu()
    got = buf[i * PAYLOAD_LEN : (i + 1) * PAYLOAD_LEN]
    assert torch.equal(expect, got), f"layout mismatch at segment {i} (block {bid})"

# 校验 2：scatter 到另一组散块后内容一致
_scatter_buffer_to_blocks(buf, 0, dst_cache, dst_ids, BLOCK_LEN, PAYLOAD_OFF, PAYLOAD_LEN)
torch.cuda.synchronize()
dst_bytes = dst_cache.view(torch.uint8).view(-1, BLOCK_LEN)
for i, (sb, db) in enumerate(zip(src_ids, dst_ids)):
    s = src_bytes[sb, PAYLOAD_OFF : PAYLOAD_OFF + PAYLOAD_LEN]
    d = dst_bytes[db, PAYLOAD_OFF : PAYLOAD_OFF + PAYLOAD_LEN]
    assert torch.equal(s.cpu(), d.cpu()), f"roundtrip mismatch at segment {i}"

# 校验 3：未触及区域保持为零（没写越界）
mask = torch.zeros(NUM_BLOCKS * BLOCK_LEN, dtype=torch.bool)
for db in dst_ids:
    mask[db * BLOCK_LEN + PAYLOAD_OFF : db * BLOCK_LEN + PAYLOAD_OFF + PAYLOAD_LEN] = True
assert (dst_cache.view(torch.uint8)[~mask.to(dst_cache.device)] == 0).all()

print("GATHER/SCATTER ROUNDTRIP OK: layout, content, bounds all verified")
