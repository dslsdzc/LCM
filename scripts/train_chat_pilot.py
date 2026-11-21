#!/usr/bin/env python3
"""CPU dialogue pilot: run the full cognitive training path on a tiny corpus.

This is the same ``train_cog`` used by the main training flow (encoder,
six lattices, fusion, self lattice, decoder), with a small batch/sequence so it
fits a CPU-only 4-core box. It is a behavioural pilot, not a quality run.

Example:
    JAX_PLATFORMS=cpu python scripts/train_chat_pilot.py \
        --output /home/DslsDZC/LCM/checkpoints_chat_run1 \
        --steps 1000 --save-every 250 --log-every 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.cog_train import train_cog  # noqa: E402
from train.config import LCMConfig  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="/home/DslsDZC/LCM/checkpoints_chat_run1")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", default=None)
    p.add_argument("--data", default="/home/DslsDZC/LCM/data_chat/dialogue.dat")
    p.add_argument("--shape", default="/home/DslsDZC/LCM/data_chat/dialogue_shape.json")
    p.add_argument("--spans", default="/home/DslsDZC/LCM/data_chat/dialogue_docs.npy")
    args = p.parse_args()

    cfg = LCMConfig()
    t0 = time.time()
    train_cog(
        cfg,
        output_dir=args.output,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch,
        seq_len=args.seq,
        log_every=args.log_every,
        save_every=args.save_every,
        data_path=args.data,
        shape_path=args.shape,
        spans_path=args.spans,
        resume=args.resume,
        joint=True,
        self_awareness=True,
        overfit_one_batch=False,
    )
    print(f"[chat-pilot] done in {time.time() - t0:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
