#!/usr/bin/env python
"""KV 静默污染复现脚本 v2：N 个**不同**长 prompt 并发打（temperature=0），
随后各自串行跑参照。并发输出与参照不一致 = KV 污染实锤。
（v1 教训：N 个相同 prompt 的 KV 内容一样，buffer 混写也看不见。）

用法：
  python scripts/repro_kv_pollution.py --concurrency 8 --target-chars 16000 \
      --max-tokens 32 --seed 123
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import requests

from long_prompts import generate_prompts

PROXY = "http://localhost:8000"


def run_one(prompt: str, model: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    resp = requests.post(f"{PROXY}/v1/chat/completions", json=payload,
                         timeout=600)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--target-chars", type=int, default=16000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    # N 个不同 prompt（同长度）
    prompts = generate_prompts(num_prompts=args.concurrency,
                               target_chars=args.target_chars,
                               seed=args.seed)

    print(f"[压测] 并发 {args.concurrency} 次（各自不同 prompt）...", flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(ex.map(
            lambda p: run_one(p, args.model, args.max_tokens), prompts))

    print("[参照] 串行逐题重跑 ...", flush=True)
    refs = [run_one(p, args.model, args.max_tokens) for p in prompts]

    n_bad = 0
    for i, (out, ref) in enumerate(zip(outs, refs)):
        same = out == ref
        if not same:
            n_bad += 1
        print(f"[{i}] {'OK  ' if same else 'DIVER'}", flush=True)
        if not same:
            print(f"     并发: {out[:80]!r}", flush=True)
            print(f"     参照: {ref[:80]!r}", flush=True)
    print("=" * 40)
    print(f"结论: {args.concurrency} 次并发中 {n_bad} 次与参照不一致"
          + ("——KV 污染复现" if n_bad else "，未见污染"))


if __name__ == "__main__":
    main()
