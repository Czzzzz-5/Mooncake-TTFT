# 并发压测：N 个互不相同的长 prompt，C 路并发，流式测每请求 TTFT
# 输出 mean/p50/p90/p99 + 吞吐；epoch 打点供 dissect_epoch.py 拼瀑布
import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from long_prompts import generate_prompts

PROXY = "http://localhost:8000"
_print_lock = threading.Lock()


def measure_one(idx: int, prompt: str, model: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time_ns()  # 客户端发送时刻（起点）
    t1 = None
    resp = requests.post(
        f"{PROXY}/v1/chat/completions", json=payload, stream=True, timeout=900
    )
    resp.raise_for_status()
    for raw in resp.iter_lines():
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for ch in chunk.get("choices", []):
            delta = (ch.get("delta") or {}).get("content")
            if delta and t1 is None:
                t1 = time.time_ns()
    t_end = time.time_ns()
    ttft_ms = (t1 - t0) / 1e6 if t1 else None
    with _print_lock:
        print(f"  [epoch] req={idx} t0={t0} t1={t1}")
    return {
        "idx": idx,
        "t0": t0,
        "ttft_ms": ttft_ms,
        "e2e_ms": (t_end - t0) / 1e6,
    }


def percentile(sorted_vals: list, q: float) -> float:
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--target-chars", type=int, default=16000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    if not args.skip_warmup:
        # 两条全尺寸 warmup：热 ~115MB KV 传输路径，避免冷启动税污染测量
        for wp in generate_prompts(
            num_prompts=2, target_chars=args.target_chars, seed=999
        ):
            measure_one(-1, wp, args.model, max_tokens=1)
        print("[warmup done]")

    prompts = generate_prompts(
        num_prompts=args.requests,
        target_chars=args.target_chars,
        seed=args.seed,
    )
    wall0 = time.time_ns()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(
            ex.map(
                lambda t: measure_one(t[0], t[1], args.model, args.max_tokens),
                enumerate(prompts),
            )
        )
    wall_s = (time.time_ns() - wall0) / 1e9

    ttfts = sorted(r["ttft_ms"] for r in results if r["ttft_ms"] is not None)
    n_fail = sum(1 for r in results if r["ttft_ms"] is None)
    print("=" * 40)
    for r in results:
        print(f"req {r['idx']}: TTFT={r['ttft_ms']:.2f} ms  e2e={r['e2e_ms']:.2f} ms")
    print("=" * 40)
    if ttfts:
        print(
            f"TTFT: n={len(ttfts)} (fail={n_fail}) | "
            f"mean={statistics.mean(ttfts):.2f} | "
            f"p50={percentile(ttfts, 0.50):.2f} | "
            f"p90={percentile(ttfts, 0.90):.2f} | "
            f"p99={percentile(ttfts, 0.99):.2f} | "
            f"min={ttfts[0]:.2f} | max={ttfts[-1]:.2f}"
        )
    print(f"吞吐: {len(results) / wall_s:.2f} req/s (wall={wall_s:.2f}s)")


if __name__ == "__main__":
    main()
