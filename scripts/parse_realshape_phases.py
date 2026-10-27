#!/usr/bin/env python
"""解析真实 shape 测速的四段计时，输出趋势表。

输入：
  1. D 侧 jsonl（my_realshape_bench.py 落盘，含每次实测的 epoch 窗口和总耗时）
  2. P 侧 stderr 日志（含 [XDBG] MC_PH1_GATHER / MC_PH2_NET_BEGIN）
  3. D 侧 stderr 日志（含 [XDBG] MC_PH2_NET_END / MC_PH3_SCATTER）

四段口径（每段都有自己实测的起止点，没有任何一段是减出来的）：
  PH1 P侧D2H   : MC_PH1_GATHER 事件的 (t0,t1)，聚合 span = max(t1)-min(t0)
  PH2 内存->内存: min(MC_PH2_NET_BEGIN.t) → max(MC_PH2_NET_END.t)，两点分属
                 P/D 两进程，同一台机器共享 epoch 时钟，合法配对
  PH3 D侧H2D   : MC_PH3_SCATTER 事件的 (t0,t1)，聚合 span
  总时间        : jsonl 里的 perf_ns（Python 层实测）

注意：多 lane 并发时段与段之间可能有流水交叠，各段 span 之和 >= 总时间是
正常现象，不是错误；交叠程度本身就是结论的一部分。

用法：
  python parse_realshape_phases.py --model dsv3 \
      --jsonl my_pd_demo/mooncake_data/realshape_dsv3_20261019.jsonl \
      --p-log /tmp/realshape_P_dsv3.log --d-log /tmp/realshape_D_dsv3.log
"""
import argparse
import json
import re
from collections import defaultdict

EVENT_RE = re.compile(r"\[XDBG\] (MC_PH\w+)(.*)")
KV_RE = re.compile(r"(\w+)=([\dxa-fA-F]+)")


def parse_events(path):
    """从日志里抽出 [(name, {k: int}), ...]"""
    events = []
    with open(path, errors="replace") as f:
        for line in f:
            m = EVENT_RE.search(line)
            if not m:
                continue
            kv = {}
            for k, v in KV_RE.findall(m.group(2)):
                kv[k] = int(v, 16) if v.startswith("0x") else int(v)
            events.append((m.group(1), kv))
    return events


def in_window(t, t0, t1, slack_ns=100_000):
    # 打点严格落在 [epoch_t0, epoch_t1] 内（传输完成后 sync 才打结束点），
    # 只留 0.1ms 余量；余量大了会把相邻档位的串行事件串进来
    return t0 - slack_ns <= t <= t1 + slack_ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--p-log", required=True)
    ap.add_argument("--d-log", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.jsonl)]
    p_events = parse_events(args.p_log)
    d_events = parse_events(args.d_log)

    ph1 = [kv for name, kv in p_events if name == "MC_PH1_GATHER"]
    ph2_begin = [kv for name, kv in p_events if name == "MC_PH2_NET_BEGIN"]
    ph2_end = [kv for name, kv in d_events if name == "MC_PH2_NET_END"]
    ph3 = [kv for name, kv in d_events if name == "MC_PH3_SCATTER"]
    print(f"事件数: PH1={len(ph1)} PH2_BEGIN={len(ph2_begin)} "
          f"PH2_END={len(ph2_end)} PH3={len(ph3)}")

    # 按 (tokens, iter) 逐次实测切片
    rows = []
    for rec in records:
        t0, t1 = rec["epoch_t0"], rec["epoch_t1"]
        w1 = [(e["t0"], e["t1"]) for e in ph1
              if in_window(e["t0"], t0, t1) and in_window(e["t1"], t0, t1)]
        wb = [e["t"] for e in ph2_begin if in_window(e["t"], t0, t1)]
        we = [e["t"] for e in ph2_end if in_window(e["t"], t0, t1)]
        w3 = [(e["t0"], e["t1"]) for e in ph3
              if in_window(e["t0"], t0, t1) and in_window(e["t1"], t0, t1)]
        if not (w1 and wb and we and w3):
            print(f"⚠️  tokens={rec['tokens']} iter={rec['iter']} 窗口内事件不齐: "
                  f"PH1={len(w1)} PH2B={len(wb)} PH2E={len(we)} PH3={len(w3)}")
            continue
        rows.append({
            "tokens": rec["tokens"], "iter": rec["iter"],
            "rounds": rec["rounds"], "bytes": rec["bytes"],
            "ph1_span": max(x[1] for x in w1) - min(x[0] for x in w1),
            "ph2_span": max(we) - min(wb),
            "ph3_span": max(x[1] for x in w3) - min(x[0] for x in w3),
            "total": rec["perf_ns"],
        })

    # 每档按 iter 取平均（用户要求：每点位测 3 次取平均）
    by_level = defaultdict(list)
    for r in rows:
        by_level[r["tokens"]].append(r)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    lines = [
        f"| tokens | 数据量MB | rounds | 段1 P-D2H ms | 段2 网络 ms | "
        f"段3 D-H2D ms | 总时间 ms | 有效带宽 GB/s | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tokens in sorted(by_level):
        rs = by_level[tokens]
        ph1m = avg([r["ph1_span"] for r in rs]) / 1e6
        ph2m = avg([r["ph2_span"] for r in rs]) / 1e6
        ph3m = avg([r["ph3_span"] for r in rs]) / 1e6
        tot = avg([r["total"] for r in rs]) / 1e6
        mb = rs[0]["bytes"] / 1e6
        bw = rs[0]["bytes"] / (tot / 1e3) / 1e9
        lines.append(
            f"| {tokens} | {mb:.0f} | {rs[0]['rounds']} | {ph1m:.1f} | "
            f"{ph2m:.1f} | {ph3m:.1f} | {tot:.1f} | {bw:.2f} | {len(rs)} |")

    table = "\n".join(lines)
    n_iters = len({r["iter"] for r in rows})
    print(f"\n## {args.model}（每档 {n_iters} 次实测取平均）\n")
    print(table)

    out_md = args.jsonl.replace(".jsonl", "_table.md")
    with open(out_md, "w") as f:
        f.write(f"# {args.model} 分段趋势表\n\n{table}\n")
    print(f"\n表格已写入 {out_md}")


if __name__ == "__main__":
    main()
