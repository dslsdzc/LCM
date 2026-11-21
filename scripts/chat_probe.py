#!/usr/bin/env python3
"""Sample a trained cog checkpoint with chat-format prompts.

Development probe only: ``use_safety=False`` skips the gvalue (Three Laws)
check. The danger lattice still runs. Do not describe this as production-safe.

Example:
    JAX_PLATFORMS=cpu python scripts/chat_probe.py \
        --checkpoint /home/DslsDZC/LCM/checkpoints_chat_run1/step_0001000 \
        --prompt "你好"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--loop", action="store_true",
                   help="Use the full macro loop per token (slow on CPU)")
    args = p.parse_args()

    import lcm

    engine = lcm.LCMInferEngine(args.checkpoint)
    prompts = args.prompt or [
        "你好",
        "你叫什么名字？",
        "你会做什么？",
        "你知道宇宙的尽头是什么吗？",
        "再见",
    ]

    for text in prompts:
        prompt = f"user: {text} assistant:"
        engine._alerts.clear()
        engine._last_loop_ret = 0
        out = engine.generate_text(
            prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            use_safety=False,
            use_loop=args.loop,
        )
        ret = int(getattr(engine, "_last_loop_ret", 0))
        print(f"\n[USER] {text}")
        print(f"[BOT ] {out!r}  (ret={ret})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
