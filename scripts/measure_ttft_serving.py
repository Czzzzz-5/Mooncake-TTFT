# 客户端侧端到端 TTFT 测量（serving 模式，唯一权威口径）
# TTFT = 客户端发送请求的时刻 -> 客户端收到第一个 token 的时刻
import argparse
import json
import time

import requests

from long_prompts import generate_prompts

PROXY = "http://localhost:8000"


def measure_one(prompt: str, model: str, max_tokens: int = 10) -> float:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time_ns()  # 客户端发送时刻（起点）
    resp = requests.post(
        f"{PROXY}/v1/chat/completions", json=payload, stream=True, timeout=600
    )
    resp.raise_for_status()
    ttft_ms = None
    for raw in resp.iter_lines():
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="replace")
        else:
            line = raw
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        # 第一个含 content 的 chunk = 客户端收到第一个 token（终点）
        for ch in chunk.get("choices", []):
            delta = (ch.get("delta") or {}).get("content")
            if delta and ttft_ms is None:
                ttft_ms = (time.time_ns() - t0) / 1e6
    return ttft_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--target-chars", type=int, default=16000)
    ap.add_argument("--max-tokens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    # 第 1 条作为 warmup 丢弃（引擎首次请求有初始化开销）
    warmup = generate_prompts(num_prompts=1, target_chars=1000, seed=999)[0]
    measure_one(warmup, args.model, max_tokens=1)
    print("[warmup done]")

    prompts = generate_prompts(
        num_prompts=args.num_prompts,
        target_chars=args.target_chars,
        seed=args.seed,
    )
    results = []
    for i, p in enumerate(prompts):
        ms = measure_one(p, args.model, max_tokens=args.max_tokens)
        n_tokens_est = len(p) // 6
        results.append(ms)
        print(f"req {i}: prompt~{n_tokens_est} tokens -> TTFT = {ms:.2f} ms")

    avg = sum(results) / len(results)
    print("=" * 40)
    print(f"端到端 TTFT (客户端口径): avg={avg:.2f} ms, "
          f"min={min(results):.2f}, max={max(results):.2f}, n={len(results)}")


if __name__ == "__main__":
    main()
