"""Create compact semantic embeddings from frozen Qwen."""
import argparse
import os

import jax
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qwen', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--dim', type=int, default=64)
    args = ap.parse_args()
    src = np.load(args.qwen)['model.embed_tokens.weight']
    proj = np.asarray(
        jax.random.normal(jax.random.PRNGKey(91), (src.shape[1], args.dim))
        / np.sqrt(args.dim), dtype=np.float32)
    out = np.empty((src.shape[0], args.dim), dtype=np.float32)
    for start in range(0, len(src), 2048):
        chunk = np.asarray(src[start:start + 2048], dtype=np.float32)
        out[start:start + len(chunk)] = chunk @ proj
    out /= np.sqrt(np.mean(out * out, axis=1, keepdims=True) + 1e-6)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, out)
    print(args.output, out.shape)


if __name__ == '__main__':
    main()
