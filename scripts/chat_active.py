#!/usr/bin/env python3
"""Sample the current-architecture active channel.

Pipeline (matches ``cog_train`` with a Qwen bridge):

    Qwen-tokenised prompt -> LCM encoder -> z
    z -> cognitive lattice loop -> z_q
    z_q -> z_proj -> frozen Qwen (first 4 decoder layers) -> logits
    logits -> token -> append -> repeat

This is a *development* script. Qwen is the temporary active-channel bridge,
not the final mouth. The checkpoint contains no Qwen tensors; the frozen Qwen
``.npz`` is loaded separately.

Usage:
    JAX_PLATFORMS=cpu python scripts/chat_active.py \
        --checkpoint /home/DslsDZC/LCM/checkpoints_cog_qwen_probe/step_000001 \
        --qwen /home/DslsDZC/LCM/checkpoints/qwen_model/qwen_params.npz \
        --prompt "你好" --max-new 8
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train.checkpoint import load_cog_state  # noqa: E402
from train.cog_loop import cog_loop_scan  # noqa: E402
from train.config import LCMConfig  # noqa: E402
from train.encoder import encoder_forward  # noqa: E402
from train.hyp import safe_unit  # noqa: E402
from train.qwen_lm import load_qwen_params, qwen_forward  # noqa: E402


DEF_QWEN = "/home/DslsDZC/LCM/checkpoints/qwen_model/qwen_params.npz"
DEF_TOK = "/home/DslsDZC/data/lcm/tokenizers/qwen2.5-0.5b/tokenizer.json"


def _cfg_from_checkpoint(ckpt: Path) -> LCMConfig:
    with open(ckpt / "config.json") as f:
        raw = json.load(f)
    fields = LCMConfig.__dataclass_fields__
    return LCMConfig(**{k: v for k, v in raw.items() if k in fields})


def _extract_reply(decoded: str) -> str:
    matches = list(re.finditer(r"assistant\s*:", decoded))
    text = decoded[matches[-1].end():] if matches else decoded
    text = re.split(r"\s*(?:user|assistant)\s*:", text, maxsplit=1)[0]
    return text.strip(" \t\n:：,，。") or "（active channel 还没有学会回答）"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--qwen", default=DEF_QWEN)
    p.add_argument("--tokenizer", default=DEF_TOK)
    p.add_argument("--prompt", default="你好")
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seq-cap", type=int, default=256)
    args = p.parse_args()

    from tokenizers import Tokenizer

    ckpt = Path(args.checkpoint)
    cfg = _cfg_from_checkpoint(ckpt)
    state = load_cog_state(str(ckpt), require_buffers=False)
    params = jax.tree_util.tree_map(jnp.asarray, state["params"])
    z_proj = params.pop("z_proj")
    qwen_params = load_qwen_params(args.qwen)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos = tokenizer.token_to_id("<|endoftext|>") or 151643
    im_end = tokenizer.token_to_id("<|im_end|>")

    prompt = f"user: {args.prompt} assistant:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    # Position 0 is overwritten by z_proj (same as training's injection slot).
    dummy = eos
    ids = [dummy] + prompt_ids
    seq_cap = max(int(args.seq_cap), len(ids) + int(args.max_new) + 1)
    ids = ids[:seq_cap]
    x = np.zeros((1, seq_cap), dtype=np.int32)
    x[0, :len(ids)] = ids
    pos = len(ids)

    z = encoder_forward(params["encoder"], jnp.asarray(x[:, 1:len(ids)]),
                        cfg.n_heads)
    z = safe_unit(z)
    z_qs, _, _ = cog_loop_scan(
        z, params, cfg, max_steps=cfg.max_inference_steps,
        rng=jax.random.PRNGKey(0), return_stats=False)
    z_q = z_qs[-1]

    out_ids = []
    for _ in range(int(args.max_new)):
        logits = qwen_forward(
            qwen_params, jnp.asarray(x), z_q=z_q, z_proj=z_proj,
            n_layers=4)
        next_logits = np.asarray(logits[0, pos - 1, :], dtype=np.float32)
        if args.temperature <= 0.0:
            next_id = int(np.argmax(next_logits))
        else:
            scaled = next_logits / max(float(args.temperature), 1e-8)
            if args.top_k > 0:
                k = min(int(args.top_k), scaled.size)
                cutoff = np.sort(scaled)[-k]
                scaled = np.where(scaled < cutoff, -np.inf, scaled)
            scaled = scaled - scaled.max()
            probs = np.exp(scaled)
            probs /= probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))
        if next_id == eos or (im_end is not None and next_id == im_end):
            break
        out_ids.append(next_id)
        x[0, pos] = next_id
        pos += 1

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    print(f"user: {args.prompt}")
    print(f"assistant: {_extract_reply(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
