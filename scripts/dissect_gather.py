#!/usr/bin/env python
"""解析 gather 轮的 XDBG 打点：按请求切分，输出各段耗时（ms）。"""
import re
import sys

paths = sys.argv[1:] or [
    "/tmp/ttft_exp/V2B_prefill.log",
    "/tmp/ttft_exp/V2B_decode.log",
]
evs = []
for p in paths:
    for line in open(p):
        m = re.search(
            r"\[XDBG\] (P_GATHER_DONE|P_SEND_EXEC|P_SEND_DONE|D_RESP_RECV|D_SCATTER_DONE)"
            r" t=(\d+)(?: bytes=(\d+))?",
            line,
        )
        if m:
            evs.append((int(m.group(2)), m.group(1), int(m.group(3) or 0)))
evs.sort()

reqs, cur = [], None
for t, name, b in evs:
    if name == "P_GATHER_DONE":
        cur = {"t0": t, "bytes": b}
        reqs.append(cur)
    elif cur is not None:
        cur[name] = t

print(f"{'req':>4} {'MB':>6} {'gather→send':>11} {'wire':>7} {'resp':>6} {'scatter':>7} {'total':>7}")
n = 0
for r in reqs:
    if r["bytes"] < 1e8 or "D_SCATTER_DONE" not in r:
        continue  # 跳过 warmup/smoke
    t0 = r["t0"]
    wire = (r["P_SEND_DONE"] - r["P_SEND_EXEC"]) / 1e6
    resp = (r["D_RESP_RECV"] - r["P_SEND_DONE"]) / 1e6
    scat = (r["D_SCATTER_DONE"] - r["D_RESP_RECV"]) / 1e6
    total = (r["D_SCATTER_DONE"] - t0) / 1e6
    g2s = (r["P_SEND_EXEC"] - t0) / 1e6
    print(f"{n:>4} {r['bytes']/1e6:>6.0f} {g2s:>11.1f} {wire:>7.1f} {resp:>6.1f} {scat:>7.1f} {total:>7.1f}")
    n += 1
