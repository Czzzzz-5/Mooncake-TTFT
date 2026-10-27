#!/usr/bin/env python
"""真实模型 KV shape 跨 GPU 传输测速：P 侧造好 KV cache 并暴露显存，D 侧按
vLLM 真实布局拉回，扫 token 档位测速。

三个模型的 KV cache 布局（模仿 vLLM v1）：
  dsv3   : DeepSeek V3，61 层 MLA，每层 (num_blocks, 16, 576) bf16
           576 = kv_lora_rank(512) + qk_rope_head_dim(64)，70.3 KB/token
  k3     : Kimi K3，93 层中只有 24 层 Gated MLA 有 KV cache（69 层 KDA 是
           固定 state、不随 token 增长），每层 (num_blocks, 16, 576) bf16，
           27.6 KB/token
  qwen7b : Qwen2.5-7B（对照），28 层 GQA，每层 (2, num_blocks, 16, 4, 128)
           bf16（2 = K/V），56 KB/token

四组计时口径（每组独立实测起止点，互不推算）：
  段1 P侧 D2H   : mooncake C++ [XDBG] MC_PH1_GATHER（t0/t1 都是实测）
  段2 内存->内存 : C++ [XDBG] MC_PH2_NET_BEGIN（P 侧首字节交给 socket）
                  → MC_PH2_NET_END（D 侧末字节收齐），跨进程 epoch 时钟
  段3 D侧 H2D   : C++ [XDBG] MC_PH3_SCATTER（t0/t1 实测）
  总时间         : 本脚本 batch_transfer_sync_read 前后的实测（perf_counter
                  + epoch 双时钟，epoch 窗口供解析脚本对齐 C++ 打点）

用法（两个终端，先 P 后 D）：
  CUDA_VISIBLE_DEVICES=1 MC_TCP_PHASE_TIMING=1 ... python my_realshape_bench.py p --model dsv3 2> /tmp/realshape_P_dsv3.log
  CUDA_VISIBLE_DEVICES=2 MC_TCP_PHASE_TIMING=1 ... python my_realshape_bench.py d --model dsv3 2> /tmp/realshape_D_dsv3.log

显存：P/D 各占 BENCH_RESIDENT_GB（默认 26）GB 驻留 buffer；超过驻留上限的
token 档分轮搬同一 buffer 累加计时，jsonl 里记 rounds。
"""
import argparse
import json
import os
import sys
import time

import torch

# ---- 模型 KV 配置表 ----------------------------------------------------------
# (层数, 每层 region 形状函数)  MLA: 每 region (num_blocks, 16, 576)
# GQA: 每 region (2, num_blocks, 16, num_kv_heads, head_size)
BLOCK_SIZE = 16
MLA_DIM = 576  # kv_lora_rank 512 + qk_rope_head_dim 64

MODELS = {
    "dsv3": {"layers": 61, "kind": "mla"},
    "k3": {"layers": 24, "kind": "mla"},
    "qwen7b": {"layers": 28, "kind": "gqa", "kv_heads": 4, "head_size": 128},
}

TOKEN_LEVELS = [
    128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 393216, 524288, 786432, 1048576,
]
# 冒烟/调试用：BENCH_MAX_TOKENS=4096 只扫到该档
_max_tokens = int(os.environ.get("BENCH_MAX_TOKENS", "0"))
if _max_tokens > 0:
    TOKEN_LEVELS = [t for t in TOKEN_LEVELS if t <= _max_tokens]

RESIDENT_BYTES = int(os.environ.get("BENCH_RESIDENT_GB", "26")) << 30
MAX_DESC_BYTES = 128 << 20  # 单个描述符上限，须 < MC_TCP_GPU_STAGING_SLOT_MB


def layer_shape(model: str, num_blocks: int) -> tuple:
    cfg = MODELS[model]
    if cfg["kind"] == "mla":
        return (num_blocks, BLOCK_SIZE, MLA_DIM)
    return (2, num_blocks, BLOCK_SIZE, cfg["kv_heads"], cfg["head_size"])


def layer_block_bytes(model: str) -> int:
    """每层每 block 的字节数（qwen 的 K/V 两半分开算 region）。"""
    cfg = MODELS[model]
    if cfg["kind"] == "mla":
        return BLOCK_SIZE * MLA_DIM * 2
    return BLOCK_SIZE * cfg["kv_heads"] * cfg["head_size"] * 2


def regions_per_layer(model: str) -> int:
    return 1 if MODELS[model]["kind"] == "mla" else 2


def num_resident_blocks(model: str) -> int:
    per_block = layer_block_bytes(model) * regions_per_layer(model) * MODELS[model]["layers"]
    return RESIDENT_BYTES // per_block


def make_engine():
    from mooncake.engine import TransferEngine

    e = TransferEngine()
    ret = e.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "")
    assert ret == 0, f"engine init failed: {ret}"
    return e


def fill_pattern(t: torch.Tensor, layer_idx: int, kind: str):
    """block bid 的全部元素 = (layer*31 + bid) % 256，bf16 可精确表示（0-255 整数）。"""
    num_blocks = t.shape[0] if kind == "mla" else t.shape[1]
    vals = (torch.arange(num_blocks, dtype=torch.int32) + layer_idx * 31) % 256
    vals = vals.to(torch.bfloat16)
    if kind == "mla":
        t.copy_(vals.view(-1, 1, 1).expand_as(t))
    else:
        v = vals.view(1, -1, 1, 1, 1).expand_as(t)
        t.copy_(v)


def alloc_register(e, model: str, num_blocks: int, fill: bool):
    """分配各层 tensor、（P 侧）填图案、register，返回 (tensors, region_bases)。
    region_bases: 每层每个 region 的 (base_addr, block_bytes)。
    D 侧必须 fill=False：接收 buffer 不预填图案，否则传输没写进去也能
    “验证通过”（自我实现的假阳性）。"""
    cfg = MODELS[model]
    shape = layer_shape(model, num_blocks)
    block_bytes = layer_block_bytes(model)
    tensors = []
    region_bases = []
    for li in range(cfg["layers"]):
        t = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        if fill:
            fill_pattern(t, li, cfg["kind"])
        tensors.append(t)
    torch.cuda.synchronize()
    for t in tensors:
        size = t.numel() * t.element_size()
        ret = e.register_memory(t.data_ptr(), size)
        assert ret == 0, f"register_memory failed: {ret}"
        if MODELS[model]["kind"] == "mla":
            region_bases.append((t.data_ptr(), block_bytes))
        else:
            half = t.stride(0) * t.element_size()
            region_bases.append((t.data_ptr(), block_bytes))          # K
            region_bases.append((t.data_ptr() + half, block_bytes))   # V
    return tensors, region_bases


def build_descriptors(local_bases, remote_bases, block_lo, block_hi):
    """把 block 区间 [block_lo, block_hi) 按 region 展开成描述符，超
    MAX_DESC_BYTES 的切块。返回 (local_addrs, remote_addrs, lens)。"""
    local_addrs, remote_addrs, lens = [], [], []
    for (lb, bb), (rb, _) in zip(local_bases, remote_bases):
        lo, hi = block_lo, block_hi
        while lo < hi:
            max_blocks = MAX_DESC_BYTES // bb
            hi_chunk = min(hi, lo + max_blocks)
            n = hi_chunk - lo
            local_addrs.append(lb + lo * bb)
            remote_addrs.append(rb + lo * bb)
            lens.append(n * bb)
            lo = hi_chunk
    return local_addrs, remote_addrs, lens


def run_p(model: str, addr_file: str):
    e = make_engine()
    num_blocks = num_resident_blocks(model)
    cfg = MODELS[model]
    gb = RESIDENT_BYTES / (1 << 30)
    print(f"[P] model={model} layers={cfg['layers']} "
          f"resident_blocks={num_blocks} (~{num_blocks * BLOCK_SIZE} tokens, "
          f"预算 {gb:.0f}GB)", flush=True)
    tensors, region_bases = alloc_register(e, model, num_blocks, fill=True)
    real_gb = sum(t.numel() * t.element_size() for t in tensors) / (1 << 30)
    print(f"[P] 实际分配 {real_gb:.1f} GiB，{len(region_bases)} 个 region", flush=True)

    if os.path.exists(addr_file):
        os.remove(addr_file)
    payload = {
        "addr": f"127.0.0.1:{e.get_rpc_port()}",
        "model": model,
        "num_blocks": num_blocks,
        "region_bases": [b for b, _ in region_bases],
        "block_bytes": layer_block_bytes(model),
    }
    with open(addr_file, "w") as f:
        json.dump(payload, f)
    print(f"[P] 就绪: {payload['addr']}，地址表已写 {addr_file}", flush=True)
    while True:
        time.sleep(3600)


def run_d(model: str, addr_file: str, iters: int, out_file: str):
    e = make_engine()
    # P 填 26GB 图案 + register 要 ~30s；D 最多等 150s。
    # json.load 可能撞上 P 写文件的中途，失败就重试。
    info = None
    for _ in range(300):
        if os.path.exists(addr_file):
            try:
                info = json.load(open(addr_file))
                break
            except json.JSONDecodeError:
                pass
        time.sleep(0.5)
    assert info is not None, f"等了 150s 还没等到 {addr_file}，P 没起来？"
    assert info["model"] == model, f"地址文件是 {info['model']} 的，先清掉旧文件"
    target = info["addr"]
    num_blocks = info["num_blocks"]
    block_bytes = info["block_bytes"]
    remote_bases = [(b, block_bytes) for b in info["region_bases"]]
    print(f"[D] P 在 {target}，resident_blocks={num_blocks}", flush=True)

    tensors, local_bases = alloc_register(e, model, num_blocks, fill=False)
    assert len(local_bases) == len(remote_bases)

    cfg = MODELS[model]
    kb_per_token = (layer_block_bytes(model) * regions_per_layer(model)
                    * cfg["layers"]) / 1024
    print(f"[D] 每 token KV = {kb_per_token:.1f} KB，开始扫档", flush=True)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out = open(out_file, "a")

    for tokens in TOKEN_LEVELS:
        blocks_needed = -(-tokens // BLOCK_SIZE)  # ceil
        rounds = max(1, -(-blocks_needed // num_blocks))
        # 一次完整搬运（含分轮）的字节数，每轮复用驻留 buffer 的前段
        pass_bytes = 0
        for r in range(rounds):
            lo = r * num_blocks
            hi = min(blocks_needed, lo + num_blocks)
            _, _, ln = build_descriptors(local_bases, remote_bases, 0, hi - lo)
            pass_bytes += sum(ln)
        # 每档：iter0 为 warmup 不计数，随后 iters 次实测
        last_ms = 0.0
        for it in range(iters + 1):
            torch.cuda.synchronize()
            t0p = time.perf_counter_ns()
            t0e = time.time_ns()
            for r in range(rounds):
                lo = r * num_blocks
                hi = min(blocks_needed, lo + num_blocks)
                la, ra, ln = build_descriptors(
                    local_bases, remote_bases, 0, hi - lo)
                ret = e.batch_transfer_sync_read(target, la, ra, ln)
                assert ret == 0, f"transfer failed: ret={ret}"
            torch.cuda.synchronize()
            t1p = time.perf_counter_ns()
            t1e = time.time_ns()
            if it == 0:
                continue  # warmup
            last_ms = (t1p - t0p) / 1e6
            rec = {
                "model": model, "tokens": tokens, "iter": it,
                "rounds": rounds, "bytes": pass_bytes,
                "perf_ns": t1p - t0p, "epoch_t0": t0e, "epoch_t1": t1e,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()

        # 校验：抽首末两个 block（在驻留范围内），图案必须精确匹配
        for bid in {0, min(blocks_needed, num_blocks) - 1}:
            for li in (0, cfg["layers"] - 1):
                t = tensors[li]
                expect = float((li * 31 + bid) % 256)
                blk = t[bid] if cfg["kind"] == "mla" else t[:, bid]
                ok = bool((blk == expect).all().item())
                assert ok, f"数据不对! tokens={tokens} layer={li} block={bid}"

        print(f"[D] tokens={tokens:>8} blocks={blocks_needed:>7} "
              f"rounds={rounds} bytes={pass_bytes / 1e6:.0f}MB "
              f"last={last_ms:.1f}ms "
              f"bw={pass_bytes / (last_ms / 1e3) / 1e6:.0f}MB/s",
              flush=True)
    out.close()
    print(f"[D] 完成，结果写入 {out_file}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["p", "d"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--iters", type=int, default=3, help="每档实测次数（另含 1 次 warmup）")
    args = ap.parse_args()

    addr_file = f"/tmp/realshape_p_addr_{args.model}.json"
    # 结果目录锚定在脚本所在位置，不依赖 cwd（后台进程 cwd 可能是 $HOME）
    # BENCH_TAG 区分对照组（如 main / ours），默认保持原文件名兼容旧数据
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mooncake_data")
    tag = os.environ.get("BENCH_TAG", "20261019")
    out_file = os.path.join(data_dir,
                            f"realshape_{args.model}_{tag}.jsonl")

    if args.role == "p":
        run_p(args.model, addr_file)
    else:
        run_d(args.model, addr_file, args.iters, out_file)
