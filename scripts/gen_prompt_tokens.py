"""Generate a token-exact prompt (default 100k tokens) for the 100k TTFT
experiment, calibrated with the Qwen tokenizer instead of char heuristics.

Usage: python gen_prompt_tokens.py [target_tokens] [out_file]
"""

import random
import sys

from transformers import AutoTokenizer

from long_prompts import _SENTENCE_POOL

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    out = sys.argv[2] if len(sys.argv) > 2 else None
    tok = AutoTokenizer.from_pretrained(MODEL)

    rng = random.Random(42)
    pool = _SENTENCE_POOL[:]
    rng.shuffle(pool)
    parts = []
    n = 0
    idx = 0
    while n < target:
        parts.append(pool[idx % len(pool)])
        idx += 1
        if len(parts) % 200 == 0:
            n = len(tok.encode(" ".join(parts)))
    text = " ".join(parts)
    n = len(tok.encode(text))
    # Trim sentence-wise to land exactly on target.
    while n > target and parts:
        parts.pop()
        text = " ".join(parts)
        n = len(tok.encode(text))
    # Pad back with a filler sentence fragment if we undershot.
    while n < target:
        text += " the"
        n = len(tok.encode(text))
    print(f"tokens={n} chars={len(text)} sentences={len(parts)}")
    if out:
        with open(out, "w") as f:
            f.write(text)
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
