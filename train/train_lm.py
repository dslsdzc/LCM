"""Stage 1: Decoder-only LM pretraining.

Trains the gen_head as a standalone autoregressive language model, WITHOUT
the encoder, codebook lattices, or fusion. The decoder learns basic vocabulary,
grammar, and sentence coherence from pure text.

Architecture:
  tokens → w_embed → [start; x_0..x_{N-1}] → causal linear attn + GLU → logits

Key differences from full LCM training:
  - No VQ loss, no contrast loss, no orth loss, no value loss
  - Only cross-entropy LM loss
  - Only gen_head params + a learned start vector are trained
  - ~1M trainable parameters (vs ~15M in full model)

Checkpoints are saved in a format compatible with the full model's
gen_head params, so they can be loaded back via:
    state['params']['gen_head'] = lm_state['gen_head']

Usage:
    python -m train.train_lm --checkpoint checkpoints/step_19000 \\
        --lr 3e-4 --steps 50000 --save-every 5000
"""
import argparse
import json
import os
import pickle
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from train.config import LCMConfig
from train.data import WikiDataIter, TextLineIter, MMAP_PATH, MMAP_SHAPE_PATH, TOKENIZER_PATH


# ─── Forward pass ──────────────────────────────────────────────────────────

def decoder_forward(params, x, training=True):
    """Decoder-only causal LM forward.

    Args:
        params: Dict with 'gen_head' (8 weight matrices) and 'w_start' (d,).
        x: Input token IDs (B, N).

    Returns:
        logits: (B, N, V) next-token predictions.
    """
    B, N = x.shape
    g = params['gen_head']
    d = g['w_q'].shape[-1]

    # Embed tokens, prepend learned start vector
    target_emb = g['w_embed'][x]                        # (B, N, d)
    start = params['w_start'][None, None, :]            # (1, 1, d)
    start = jnp.broadcast_to(start, (B, 1, d))
    inputs = jnp.concatenate([start, target_emb], axis=1)  # (B, N+1, d)

    # Causal linear attention: φ(x) = ELU(x) + 1
    Q = jax.nn.elu(inputs @ g['w_q']) + 1.0             # (B, N+1, d)
    K = jax.nn.elu(inputs @ g['w_k']) + 1.0
    V = inputs @ g['w_v']

    # Causal cumulative sum (O(d²) per step, linear in sequence length)
    kv = K[:, :, :, None] @ V[:, :, None, :]            # (B, N+1, d, d)
    kv_cs = jnp.cumsum(kv, axis=1)
    k_cs = jnp.cumsum(K, axis=1)

    attn = jnp.einsum('bnd,bnde->bne', Q, kv_cs) / (
        jnp.einsum('bnd,bnd->bn', Q, k_cs)[:, :, None] + 1e-8)
    attn_out = attn @ g['w_o']                           # (B, N+1, d)

    # GLU
    gate = jax.nn.sigmoid(attn_out @ g['w_1'])
    up = attn_out @ g['w_2']
    glu_out = gate * up                                   # (B, N+1, 4d)

    full_logits = glu_out @ g['w_3']                      # (B, N+1, V)
    return full_logits[:, 1:, :]  # (B, N, V) — predict next token at each position


# ─── Loss ──────────────────────────────────────────────────────────────────

def lm_loss(logits, targets):
    """Cross-entropy LM loss."""
    B, N, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, V), targets.reshape(-1)).mean()


# ─── Training step ─────────────────────────────────────────────────────────

def make_train_step():
    """Create jitted training step function."""

    @partial(jax.jit, static_argnums=(3,))
    def train_step(params, opt_state, batch, lr):
        inputs, targets = batch

        def loss_fn(p):
            logits = decoder_forward(p, inputs, training=True)
            return lm_loss(logits, targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = opt_state_optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return train_step


# ─── Init params ──────────────────────────────────────────────────────────

def init_lm_params(rng, d, vocab_size, from_full_checkpoint=None):
    """Initialize decoder-only LM params.

    Args:
        rng: JAX PRNG key.
        d: Model dimension.
        vocab_size: Vocabulary size.
        from_full_checkpoint: Optional path to full LCM checkpoint .pkl.
            If provided, loads gen_head weights from it as starting point.

    Returns:
        params: {'gen_head': {...}, 'w_start': (d,)}
    """
    from train.fusion import init_gen_head_params

    keys = jax.random.split(rng, 4)
    params = {}

    if from_full_checkpoint:
        print(f"[LM] Loading gen_head from {from_full_checkpoint}...")
        with open(from_full_checkpoint, 'rb') as f:
            ckpt = pickle.load(f)
        full_params = ckpt['params'] if 'params' in ckpt else ckpt
        # Extract gen_head if it exists in the checkpoint
        if 'gen_head' in full_params:
            params['gen_head'] = full_params['gen_head']
            # Convert JAX arrays if needed
            params['gen_head'] = jax.tree_util.tree_map(
                lambda x: jnp.array(x) if not isinstance(x, (jax.Array,)) else x,
                params['gen_head'])
            print(f"[LM] Loaded gen_head from checkpoint (step {ckpt.get('step', '?')})")
        else:
            print("[LM] No gen_head found in checkpoint, initializing fresh")
            params['gen_head'] = init_gen_head_params(keys[0], d, vocab_size)
    else:
        params['gen_head'] = init_gen_head_params(keys[0], d, vocab_size)

    # Learned start vector (replaces z_q in the full model)
    params['w_start'] = jax.random.normal(keys[1], (d,)) * (d ** -0.5)

    return params


# ─── Checkpoint save/load ─────────────────────────────────────────────────

def save_lm_checkpoint(params, opt_state, step, path):
    """Save LM checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    ckpt = {
        'gen_head': jax.tree_util.tree_map(lambda x: jax.device_get(x), params['gen_head']),
        'w_start': jax.device_get(params['w_start']),
        'opt_state': jax.tree_util.tree_map(
            lambda x: jax.device_get(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
            opt_state) if opt_state else None,
        'step': step,
        'd_model': params['gen_head']['w_q'].shape[-1],
    }
    with open(path, 'wb') as f:
        pickle.dump(ckpt, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[CKPT] Saved LM checkpoint {path} ({size_mb:.1f} MB, step {step})")
    return path


def load_lm_checkpoint(path, rng):
    """Load LM checkpoint and rebuild params."""
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)

    params = {
        'gen_head': jax.tree_util.tree_map(
            lambda x: jnp.array(x) if not isinstance(x, (jax.Array,)) else x,
            ckpt['gen_head']),
        'w_start': jnp.array(ckpt['w_start']),
    }
    print(f"[LM] Loaded checkpoint {path} (step {ckpt.get('step', '?')})")
    return params, ckpt.get('step', 0)


def merge_into_full_checkpoint(lm_path, full_path, output_path):
    """Merge LM gen_head weights into a full model checkpoint.

    Args:
        lm_path: Path to LM checkpoint .pkl.
        full_path: Path to full LCM checkpoint .pkl.
        output_path: Output path for merged checkpoint.
    """
    with open(lm_path, 'rb') as f:
        lm_ckpt = pickle.load(f)
    with open(full_path, 'rb') as f:
        full_ckpt = pickle.load(f)

    gen_head_np = jax.tree_util.tree_map(
        lambda x: np.array(x) if hasattr(x, 'numpy') else x, lm_ckpt['gen_head'])
    full_ckpt['params']['gen_head'] = gen_head_np
    full_ckpt['step'] = full_ckpt.get('step', 0)

    with open(output_path, 'wb') as f:
        pickle.dump(full_ckpt, f)
    print(f"[MERGE] gen_head from {lm_path} merged into {output_path}")


# ─── Main training loop ──────────────────────────────────────────────────

def train_lm(cfg, output_dir, steps=50000, lr=3e-4, batch_size=16,
             seq_len=512, log_every=100, save_every=5000,
             from_ckpt=None, from_full_ckpt=None,
             data_path=None, shape_path=None):
    """Run Stage 1: decoder-only LM pretraining.

    Args:
        cfg: LCMConfig.
        output_dir: Directory for LM checkpoints.
        steps: Total training steps.
        lr: Learning rate.
        batch_size: Batch size.
        seq_len: Sequence length.
        log_every: Logging interval.
        save_every: Checkpoint save interval.
        from_ckpt: Path to LM checkpoint to resume from.
        from_full_ckpt: Path to full model checkpoint to initialize gen_head from.
        data_path: Path to .dat mmap file (default: MMAP_PATH from data.py).
        shape_path: Path to shape JSON (default: derived from data_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(42)

    # Init params
    rng, init_rng = jax.random.split(rng)
    step_offset = 0

    if from_ckpt:
        params, step_offset = load_lm_checkpoint(from_ckpt, init_rng)
        print(f"[LM] Resuming from step {step_offset}")
    else:
        params = init_lm_params(init_rng, cfg.d_model, cfg.vocab_size,
                                from_full_checkpoint=from_full_ckpt)

    # Optimizer
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=cfg.adam_beta1,
                     b2=cfg.adam_beta2, eps=cfg.adam_eps,
                     weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(params)
    if from_ckpt:
        with open(from_ckpt, 'rb') as f:
            ckpt = pickle.load(f)
        if ckpt.get('opt_state'):
            opt_state = jax.tree_util.tree_map(
                lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
                ckpt['opt_state'])

    # Data iterator — auto-detect text/jsonl vs binary .dat
    if data_path and (data_path.endswith('.txt') or data_path.endswith('.jsonl')):
        print(f"[LM] Online tokenization from: {data_path}")
        data_iter = TextLineIter(text_path=data_path, tokenizer_path=TOKENIZER_PATH,
                                  B=batch_size, N=seq_len)
    else:
        _mp = data_path or MMAP_PATH
        _sp = shape_path or (data_path.replace('.dat', '_shape.json') if data_path else MMAP_SHAPE_PATH)
        data_iter = WikiDataIter(mmap_path=_mp,
                                  shape_path=_sp,
                                  B=batch_size, N=seq_len)

    # JIT-compiled step
    @jax.jit
    def jitted_step(p, opt, batch, lr_val):
        inputs, targets = batch

        def loss_fn(pp):
            logits = decoder_forward(pp, inputs, training=True)
            return lm_loss(logits, targets)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_opt = optimizer.update(grads, opt, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt, loss

    # Training loop
    print(f"[LM] Starting Stage 1 training: {steps} steps, lr={lr}, "
          f"B={batch_size}, N={seq_len}")
    print(f"[LM] Trainable params: gen_head (8 matrices) + w_start ({cfg.d_model}d)")
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params)
                   if hasattr(p, 'size'))
    print(f"[LM] Total params: {n_params:,}")
    print()

    from tqdm import tqdm
    from train.monitor import MetricsRecorder
    recorder = MetricsRecorder(save_dir=output_dir, window=50)

    total_steps = steps + step_offset
    running_loss = 0.0
    start_time = time.time()
    pbar = tqdm(total=steps, desc="   stage 1 training", unit="step",
                initial=step_offset)

    for global_step in range(step_offset, total_steps):
        batch = next(data_iter)
        rng, step_rng = jax.random.split(rng)

        current_lr = schedule(global_step - step_offset)

        params, opt_state, loss_val = jitted_step(
            params, opt_state, batch, current_lr)

        running_loss += float(loss_val)

        if global_step % 50 == 0:
            recorder.record(global_step, loss=float(loss_val), lr=float(current_lr))

        # Logging
        if (global_step - step_offset) % log_every == 0 and global_step >= step_offset and global_step > 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - start_time
            tokens_per_sec = batch_size * seq_len * log_every / elapsed
            lr_val = float(current_lr)
            tqdm.write(f"  step {global_step:>6d} | loss={avg_loss:.4f} | "
                       f"lr={lr_val:.2e} | {tokens_per_sec:.0f} tok/s")
            running_loss = 0.0
            start_time = time.time()

        # Checkpoint save
        if save_every > 0 and (global_step + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"lm_step_{global_step + 1}.pkl")
            save_lm_checkpoint(params, opt_state, global_step + 1, ckpt_path)
            recorder.save()

        pbar.update(1)

    pbar.close()

    recorder.save()
    final_path = os.path.join(output_dir, "lm_final.pkl")
    save_lm_checkpoint(params, opt_state, total_steps, final_path)
    print(f"[LM] Stage 1 complete! Final checkpoint: {final_path}")
    return params


# ─── Quick sanity check ───────────────────────────────────────────────────

def sanity_check():
    """Quick test: init model, run one forward/backward, verify loss decreases."""
    cfg = LCMConfig()
    rng = jax.random.PRNGKey(0)
    params = init_lm_params(rng, cfg.d_model, cfg.vocab_size)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params)
                   if hasattr(p, 'size'))
    print(f"[SANITY] params={n_params:,}")

    # Fake batch
    x = jnp.zeros((2, 32), dtype=jnp.int32)
    logits = decoder_forward(params, x, training=True)
    print(f"[SANITY] logits shape: {logits.shape} (expected: 2, 32, 30000)")

    targets = jnp.ones((2, 32), dtype=jnp.int32)
    loss_val = lm_loss(logits, targets)
    print(f"[SANITY] initial loss: {float(loss_val):.4f} (expected ~ln(30000)≈10.3)")

    # One gradient step
    opt = optax.adamw(3e-4)
    opt_state = opt.init(params)
    grads = jax.grad(lambda p: lm_loss(decoder_forward(p, x), targets))(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    loss2 = lm_loss(decoder_forward(params_new, x), targets)
    print(f"[SANITY] after 1 step: {float(loss2):.4f} (should be lower)")
    print("[SANITY] OK!")
    return True


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Decoder-only LM Pretraining")
    parser.add_argument("--output-dir", default="checkpoints/lm_stage1",
                        help="Output directory for LM checkpoints")
    parser.add_argument("--steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Logging interval")
    parser.add_argument("--save-every", type=int, default=5000,
                        help="Checkpoint save interval")
    parser.add_argument("--from-ckpt", default=None,
                        help="Resume from LM checkpoint")
    parser.add_argument("--from-full-ckpt", default=None,
                        help="Initialize gen_head from full LCM checkpoint")
    parser.add_argument("--sanity", action="store_true",
                        help="Run sanity check and exit")
    args = parser.parse_args()

    if args.sanity:
        sanity_check()
        return

    cfg = LCMConfig()
    # Override d_model if needed (to match existing checkpoints)
    train_lm(
        cfg=cfg,
        output_dir=args.output_dir,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        log_every=args.log_every,
        save_every=args.save_every,
        from_ckpt=args.from_ckpt,
        from_full_ckpt=args.from_full_ckpt,
    )


if __name__ == "__main__":
    main()
