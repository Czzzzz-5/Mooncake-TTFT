#!/usr/bin/env python
"""锁页拷贝纯带宽 bench（v2 槽数设计输入：t_copy 段测定）。

测 GPU<->pinned 两个方向、多个片大小的纯拷贝带宽（无引擎、无网络），
与 mooncake_tcp_bench.py --buf cpu 的纯 TCP 数对比，得出 copy:send 时间比：
    v2 流水槽数 ≈ ceil(t_send / t_copy) + 1
片大小扫描的意义：太小会撞 per-copy 启动开销（带宽掉档），
太大则流水粒度粗——选两段都进入线性区的尺寸作为 v2 切片候选。

用法：
  CUDA_VISIBLE_DEVICES=0 python scripts/pinned_copy_bench.py [--iters 20]
"""
import argparse
import statistics
import time

import torch

SIZES_MB = [4, 8, 16, 32, 64, 115]


def bench_copy(src, dst, iters):
    for _ in range(3):  # warmup
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def report(name, size, ts):
    ms = [t * 1e3 for t in ts]
    gbps = [size / t / 1e9 for t in ts]
    print(f"{name} size={size / 1e6:7.1f} MB | avg {statistics.mean(ms):7.2f} ms "
          f"({statistics.mean(gbps):5.2f} GB/s) | median {statistics.median(ms):7.2f} "
          f"| min {min(ms):7.2f} | max {max(ms):7.2f} | n={len(ts)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need a CUDA device"
    print(f"[bench] device={torch.cuda.get_device_name(0)}", flush=True)

    for size_mb in SIZES_MB:
        n = int(size_mb * 1e6)
        gpu = torch.empty(n, dtype=torch.uint8, device="cuda")
        pin = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        report("D2H pinned", n, bench_copy(gpu, pin, args.iters))
        report("H2D pinned", n, bench_copy(pin, gpu, args.iters))
        del gpu, pin

    # 参照项：pageable（legacy 路径的内存类型），只测 115MB 一档
    n = int(115 * 1e6)
    gpu = torch.empty(n, dtype=torch.uint8, device="cuda")
    page = torch.empty(n, dtype=torch.uint8)
    report("D2H pageable(ref)", n, bench_copy(gpu, page, args.iters))
    report("H2D pageable(ref)", n, bench_copy(page, gpu, args.iters))


if __name__ == "__main__":
    main()
