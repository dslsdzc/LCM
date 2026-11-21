#!/usr/bin/env python3
"""Minimal chat REPL for the Language LCM checkpoint.

This is a development demo, not the production safety path. It uses the
generative ``lang_lcm`` checkpoint (causal decoder), not the full cognitive
``cog_train`` readout. The model was trained on a tiny synthetic dialogue set,
so it knows the *shape* of a conversation, not general knowledge.

Usage:
    JAX_PLATFORMS=cpu python scripts/chat_lang.py \
        --checkpoint /home/DslsDZC/LCM/checkpoints_lang_chat2/lang_final.pkl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.config import LCMConfig  # noqa: E402
from train.lang_lcm import lang_lcm_generate  # noqa: E402
from train.train_lang_lcm import load_checkpoint  # noqa: E402


def _extract_reply(decoded: str, prompt: str) -> str:
    """Take only the first assistant turn, not the model's next user turn."""
    text = decoded
    # The decoded prompt can differ from the input by punctuation normalisation
    # (e.g. "？" → "?"), so do not rely on a prefix match. The prompt always
    # ends with the assistant marker; take the text after its last occurrence.
    matches = list(re.finditer(r"assistant\s*:", text))
    if matches:
        text = text[matches[0].end():]
    # The model often continues with "user:" after finishing the reply.
    text = re.split(r"\s*(?:user|assistant)\s*:", text, maxsplit=1)[0]
    return text.strip(" \t\n:：,，。") or "我还在学习，暂时不知道怎么回答。"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default=None,
                   help="One-shot prompt; omit for an interactive REPL")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (most stable on this tiny model)")
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    import jax
    from tokenizers import Tokenizer

    cfg = LCMConfig()
    params, _step = load_checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(REPO_ROOT / "data" / "tokenizer.json"))
    bos = tokenizer.token_to_id("[BOS]") or 2
    eos = tokenizer.token_to_id("[EOS]") or 3
    rng = jax.random.PRNGKey(42)
    max_len = int(cfg.max_seq_len)

    def reply(history: list[tuple[str, str]], user: str) -> str:
        lines = [f"{role}: {content}" for role, content in history]
        lines.append(f"user: {user}")
        lines.append("assistant:")
        prompt = " ".join(lines)
        ids = tokenizer.encode(prompt).ids
        # Keep the most recent context; the model was trained at N=64.
        if len(ids) > max_len - args.max_new:
            ids = ids[-(max_len - args.max_new):]
            prompt = tokenizer.decode(ids)
        tokens = lang_lcm_generate(
            params, ids, max_len=args.max_new, bos_id=bos, eos_id=eos,
            rng=rng, cfg=cfg, temperature=args.temperature,
            top_k=args.top_k)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        return _extract_reply(decoded, prompt)

    if args.prompt:
        print(f"user: {args.prompt}")
        print(f"assistant: {reply([], args.prompt)}")
        return 0

    history: list[tuple[str, str]] = []
    print("[CHAT] Language LCM demo. Type 'quit' to exit. /reset clears history.")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            break
        if user == "/reset":
            history.clear()
            print("[CHAT] history cleared")
            continue
        answer = reply(history, user)
        print(f"assistant: {answer}")
        history.append(("user", user))
        history.append(("assistant", answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
