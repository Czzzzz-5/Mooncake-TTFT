#!/usr/bin/env python
"""解析 main 分支 [MC-PHASE] 打点，按 jsonl 的 epoch 窗口聚合到 token 档。

main 的结构是每个 64KB 块"同步拷贝→收发→下一块"严格串行，没有流水线，
所以每个描述符内部 copy 与 io 互补。口径：
  PH1（P 侧显存->内存）= P 日志 side=server 的 copy_ms 求和
  PH3（D 侧内存->显存）= D 日志 side=client 的 copy_ms 求和
  PH2（网络）           = 两侧 io_ms 求和取大侧（两侧同一批字节的收发）
注意多 lane 并行时求和值 > 墙钟，占比解读用 copy/(copy+io) 的单 lane 结构比。
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime

LINE = re.compile(
    r"^I(\d{8}) (\d{2}):(\d{2}):(\d{2})\.(\d+) .*"
    r"\[MC-PHASE\] side=(\w+) op=(\w+) bytes=(\d+) "
    r"copy_ms=([\d.]+) io_ms=([\d.]+) total_ms=([\d.]+) status=(\w+)")


def to_epoch_ns(m):
    dt = datetime.strptime(m.group(1) + m.group(2) + m.group(3) + m.group(4),
                           "%Y%m%d%H%M%S")
    micros = int(m.group(5)[:6].ljust(6, "0"))
    return int(dt.timestamp() * 1e9) + micros * 1000


def load_windows(jsonl_path, tokens):
    wins = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r["tokens"] == tokens:
                wins.append((r["epoch_t0"], r["epoch_t1"], r["iter"]))
    return sorted(wins)


def bucket(log_path, wins):
    sums = defaultdict(lambda: [0.0, 0.0])  # iter -> [copy_ms, io_ms]
    with open(log_path, errors="replace") as f:
        for line in f:
            m = LINE.match(line)
            if not m:
                continue
            t = to_epoch_ns(m)
            for t0, t1, it in wins:
                if t0 <= t <= t1:
                    sums[it][0] += float(m.group(9))
                    sums[it][1] += float(m.group(10))
                    break
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--p-log", required=True)
    ap.add_argument("--d-log", required=True)
    ap.add_argument("--tokens", type=int, default=131072)
    args = ap.parse_args()

    wins = load_windows(args.jsonl, args.tokens)
    assert wins, "jsonl 里没有该档位"
    srv = bucket(args.p_log, wins)  # P 侧 server：copy=D2H(PH1)
    cli = bucket(args.d_log, wins)  # D 侧 client：copy=H2D(PH3)，io=网络

    print(f"model={args.model} tokens={args.tokens}（main，{len(wins)} 次实测求和）")
    print("| iter | PH1 P侧D2H ms(求和) | PH3 D侧H2D ms(求和) | PH2 网络 ms(求和) | copy结构占比 |")
    print("|---|---|---|---|---|")
    for t0, t1, it in wins:
        c1, _ = srv[it]
        c3, io = cli[it]
        share = (c1 + c3) / (c1 + c3 + io) if (c1 + c3 + io) else 0
        print(f"| {it} | {c1:.1f} | {c3:.1f} | {io:.1f} | {share:.1%} |")


if __name__ == "__main__":
    main()
