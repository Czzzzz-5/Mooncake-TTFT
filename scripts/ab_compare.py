#!/usr/bin/env python
"""A/B 对比：main vs ours（2026-10-26 同轮紧邻数据）。

数据源：mooncake_data/realshape_{model}_{tag}.jsonl（每档 3 次实测取平均）。
输出：每模型一张全档对比表（总时间、带宽、提速比），外加 128k 重点行汇总。
"""
import json
import os
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mooncake_data")
MODELS = ["k3", "dsv3", "qwen7b"]
TAGS = {"main": "main", "ours": "20261026"}


def load(model, tag):
    path = os.path.join(DATA, f"realshape_{model}_{tag}.jsonl")
    rows = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["tokens"]].append((r["perf_ns"], r["bytes"]))
    # 每档 3 次取平均
    return {t: (sum(p for p, _ in v) / len(v), v[0][1]) for t, v in rows.items()}


def fmt_size(tokens):
    if tokens >= 1024:
        return f"{tokens // 1024}k"
    return str(tokens)


for model in MODELS:
    a = load(model, TAGS["main"])
    b = load(model, TAGS["ours"])
    assert set(a) == set(b), f"{model} 两侧档位不一致"
    print(f"\n## {model}（main vs ours，2026-10-26 同轮）\n")
    print("| tokens | main 总ms | ours 总ms | main 带宽GB/s | ours 带宽GB/s | 提速比 |")
    print("|---|---|---|---|---|---|")
    for t in sorted(a):
        ma, by = a[t]
        mb, _ = b[t]
        print(f"| {fmt_size(t)} | {ma / 1e6:.1f} | {mb / 1e6:.1f} "
              f"| {by / ma:.2f} | {by / mb:.2f} | {ma / mb:.2f}x |")

# 128k 汇总
print("\n## 128k 档汇总\n")
print("| 模型 | main ms | ours ms | main GB/s | ours GB/s | 提速比 |")
print("|---|---|---|---|---|---|")
for model in MODELS:
    a = load(model, TAGS["main"])[131072]
    b = load(model, TAGS["ours"])[131072]
    print(f"| {model} | {a[0] / 1e6:.1f} | {b[0] / 1e6:.1f} "
          f"| {a[1] / a[0]:.2f} | {b[1] / b[0]:.2f} | {a[0] / b[0]:.2f}x |")
