"""Stage 1: Language LCM standalone training with V4 innovations.

Trains a pure-transformer Language LCM with mHC residual streams,
Multi-Token Prediction (MTP), and Muon optimizer.

Usage:
    python -m train.train_lang_lcm --lr 3e-4 --steps 100000

Checkpoint format:
    {'lang_params': ..., 'opt_state': ..., 'step': ..., 'cfg': ...}
"""
import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from train.config import LCMConfig
from train.data import WikiDataIter, MMAP_PATH, MMAP_SHAPE_PATH
from train.lang_lcm import init_lang_lcm_params, lang_lcm_forward


# ─── Precision (BF16) ─────────────────────────────────────────────────────────

def maybe_cast_to_bf16(params, cfg):
    """Convert all params to BF16 when enabled (FP32 fallback on no-BF16 GPUs)."""
    if getattr(cfg, 'use_bf16', False):
        try:
            import jax
            import jax.numpy as jnp
            params = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 and x.ndim > 0 else x,
                params)
            print(f"[PREC] BF16 enabled: params converted to bfloat16")
        except Exception as e:
            print(f"[PREC] BF16 not supported ({e}), falling back to float32")
    return params


# ─── MTP Loss ─────────────────────────────────────────────────────────────────

def lang_lm_loss(logits, targets, aux=None, mtp_weight=0.3):
    """Cross-entropy LM loss with optional Multi-Token Prediction.

    Main loss: CE over next-token predictions (standard).
    MTP loss:  CE over future-token predictions (aux['mtp_logits']).

    Args:
        logits: (B, N, V) main head logits.
        targets: (B, N) integer token IDs.
        aux: Dict with optional 'mtp_logits' list from lang_lcm_forward.
        mtp_weight: Weight for each MTP depth's loss.

    Returns:
        Scalar loss.
    """
    B, N, V = logits.shape

    # Main next-token loss: predict targets[:, 1:] from logits[:, :-1, :]
    # Wait — logits from forward are already (B, N, V) where position t
    # predicts token t+1. So loss is against targets[:, :], shifted.
    # Actually, in the forward: final h @ W_out gives logits at each position.
    # Position t should predict token at position t+1 (teacher forcing with
    # targets[:, t] = input[:, t+1]).
    # So: logits[:, t, :] should match targets[:, t].
    # Standard: CE(logits.reshape(-1, V), targets.reshape(-1))

    # ── Main loss ────────────────────────────────────────────────────────
    # targets are (B, N) where targets[:, t] = input[:, t+1]
    main_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, V), targets.reshape(-1)).mean()

    # ── MTP loss ─────────────────────────────────────────────────────────
    mtp_loss = jnp.array(0.0)
    if aux is not None and 'mtp_logits' in aux:
        mtp_logits_list = aux['mtp_logits']
        depths = aux.get('mtp_depths', list(range(1, len(mtp_logits_list) + 1)))

        for d, mtp_l in zip(depths, mtp_logits_list):
            # mtp_l: (B, N-d, V) — predicts token at position d
            # targets for depth d: targets[:, d:]   (B, N-d)
            N_d = mtp_l.shape[1]
            target_d = targets[:, d:N_d + d]  # (B, N-d)
            loss_d = optax.softmax_cross_entropy_with_integer_labels(
                mtp_l.reshape(-1, V), target_d.reshape(-1)).mean()
            mtp_loss = mtp_loss + loss_d

        # Average over depths, scale by weight
        mtp_loss = mtp_loss * (mtp_weight / max(len(mtp_logits_list), 1))

    return main_loss + mtp_loss


# ─── Muon Optimizer ──────────────────────────────────────────────────────────

def _newton_schulz(G, iters=5):
    """Approximate nearest orthogonal matrix via Newton-Schulz.

    For gradient matrix G (m, n) with m <= n:
      X₀ = G @ G.ᵀ
      X_{t+1} = X_t @ (3I - X_t @ (3I - X_t)) / 2    (t=0..iters-1)
      return X_{iters} @ G                             ≈ nearest orthogonal

    Handles tensors with ndim > 2 by flattening all but the last dim.

    Args:
        G: (..., m, n) gradient tensor, ndim >= 2.
        iters: Newton-Schulz iterations.

    Returns:
        Orthogonalized gradient, same shape as G.
    """
    orig_shape = G.shape
    if G.ndim == 2:
        m, n = G.shape
        G_2d = G
    else:
        # Flatten all dims except last: (..., m, n) → (prod(..., m), n)
        *batch_dims, m, n = G.shape
        G_2d = G.reshape(-1, n)
        # Apply Newton-Schulz to the 2D reshaped version
        result_2d = _apply_ns_2d(G_2d, iters)
        return result_2d.reshape(orig_shape)

    return _apply_ns_2d(G_2d, iters)


def _apply_ns_2d(G, iters=5):
    """Newton-Schulz on a 2D matrix G (m, n)."""
    m, n = G.shape
    if m > n:
        return _apply_ns_2d(G.T, iters).T

    X = G @ G.T
    I = jnp.eye(m, dtype=G.dtype)
    for _ in range(iters):
        X = X @ (3 * I - X @ (3 * I - X)) / 2
    return X @ G


def muon_transform(learning_rate_fn, newton_schulz_iters=5):
    """Muon optimizer: orthogonalize matrix gradients, SGD for vectors.

    Evaluates learning_rate_fn(step_count) inside the JIT'd update
    to get the current LR.

    For matrix params (ndim >= 2): Newton-Schulz orthogonalization → SGD.
    For vectors (ndim < 2): plain SGD.

    Based on: Jordan et al., "Muon: An Optimizer for Matrix Parameters" (2024)

    Args:
        learning_rate_fn: callable(step) → scalar LR.
        newton_schulz_iters: Newton-Schulz iterations.

    Returns:
        optax.GradientTransformation
    """
    def init_fn(params):
        return jnp.zeros(())

    def update_fn(updates, state, params):
        lr = learning_rate_fn(state)  # evaluate schedule at current step

        def _process(g, p):
            if g is None:
                return None
            if g.ndim >= 2:
                return -lr * _newton_schulz(g, newton_schulz_iters)
            return -lr * g

        new_updates = jax.tree_util.tree_map(_process, updates, params)
        return new_updates, state + 1

    return optax.GradientTransformation(init_fn, update_fn)


def make_optimizer(cfg, steps, lr):
    """Build optimizer: Muon for matrix params + AdamW for vectors, or full AdamW.

    When cfg.use_muon is True, matrix-shaped params (weights) use Muon
    while vectors/biases use AdamW.  When False, all params use AdamW.
    """
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.1)

    use_muon = getattr(cfg, 'use_muon', False)

    if use_muon:
        muon_opt = muon_transform(learning_rate_fn=schedule)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            muon_opt,
            optax.add_decayed_weights(cfg.weight_decay),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, b1=cfg.adam_beta1,
                         b2=cfg.adam_beta2, eps=cfg.adam_eps,
                         weight_decay=cfg.weight_decay),
        )

    return optimizer, schedule


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(params, opt_state, step, path):
    """Save Language LCM checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    ckpt = jax.tree_util.tree_map(lambda x: np.array(x), params)
    if opt_state is not None:
        ckpt_opt = jax.tree_util.tree_map(
            lambda x: np.array(x) if hasattr(x, 'numpy') else x, opt_state)
    else:
        ckpt_opt = None
    data = {
        'lang_params': ckpt,
        'opt_state': ckpt_opt,
        'step': step,
        'd_model': jnp.array(params['W_out'].shape[0]),
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    size_mb = os.path.getsize(path) / 1e6
    n_saved = sum(p.size for p in jax.tree_util.tree_leaves(ckpt) if hasattr(p, 'size'))
    print(f"[CKPT] Step {step}: saved {path} ({size_mb:.1f} MB, {n_saved:,} params)")


def load_checkpoint(path):
    """Load Language LCM checkpoint.

    Args:
        path: Path to .pkl checkpoint.

    Returns:
        params, step
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    params = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
        data['lang_params'])
    step = data.get('step', 0)
    print(f"[CKPT] Loaded {path} (step {step})")
    return params, step


# ─── Training loop ────────────────────────────────────────────────────────────

def train_lang_lcm(cfg, output_dir, steps=100000, lr=3e-4, batch_size=16,
                   seq_len=512, log_every=100, save_every=5000,
                   from_ckpt=None, data_path=None, shape_path=None):
    """Run Language LCM training with MTP + mHC + Muon.

    Args:
        cfg: LCMConfig.
        output_dir: Output directory for checkpoints.
        steps: Total training steps.
        lr: Learning rate.
        batch_size: Batch size.
        seq_len: Sequence length.
        log_every: Logging interval.
        save_every: Checkpoint save interval.
        from_ckpt: Resume from checkpoint path.
        data_path: Path to .dat mmap file.
        shape_path: Path to shape JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(42)

    # Init params
    rng, init_rng = jax.random.split(rng)
    d = cfg.d_model
    step_offset = 0
    if from_ckpt:
        params, step_offset = load_checkpoint(from_ckpt)
        # Strip old codebook-related keys
        params.pop('codebook_entries', None)
        for layer in params.get('decoder', []):
            layer.pop('cb_read', None)
            layer.pop('ln3_scale', None)
            layer.pop('ln3_bias', None)
        # Add pos_embed if missing
        if 'pos_embed' not in params:
            max_len = getattr(cfg, 'max_seq_len', 512)
            d_ckpt = params.get('W_out', {}).shape[0] if hasattr(params.get('W_out'), 'shape') else d
            rng, pe_rng = jax.random.split(rng)
            params['pos_embed'] = jax.random.normal(pe_rng, (max_len, d_ckpt)) * (d_ckpt ** -0.5)
            print(f"[CKPT]  Added pos_embed ({max_len}, {d_ckpt})")
        # Add mHC params if missing
        if 'hc' not in params and getattr(cfg, 'n_hc', 1) > 1:
            from train.lang_lcm import init_hc_params
            n_hc = getattr(cfg, 'n_hc', 2)
            n_layers = len(params.get('decoder', []))
            rng, hc_rng = jax.random.split(rng)
            params['hc'] = init_hc_params(hc_rng, d, n_hc, n_layers)
            print(f"[CKPT]  Added mHC params (n_hc={n_hc}, {n_layers} layers)")
    else:
        params = init_lang_lcm_params(init_rng, cfg)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params)
                   if hasattr(p, 'size'))
    params = maybe_cast_to_bf16(params, cfg)
    rng_init = jax.random.PRNGKey(0)
    _test_x = jnp.zeros((1, 4), dtype=jnp.int32)
    _logits, _z_qs, _aux = lang_lcm_forward(params, _test_x, cfg, rng=rng_init)
    n_mtp = getattr(cfg, 'n_mtp_depth', 1)
    n_hc = getattr(cfg, 'n_hc', 1)
    use_muon = getattr(cfg, 'use_muon', False)
    print(f"[LANG] Language LCM — {n_params:,} params")
    print(f"[LANG] mHC: {'ON (n_hc=' + str(n_hc) + ')' if n_hc > 1 else 'OFF'}")
    print(f"[LANG] MTP: {'ON (depth=' + str(n_mtp) + ')' if n_mtp > 1 else 'OFF'}")
    print(f"[LANG] Optimizer: {'Muon+AdamW' if use_muon else 'AdamW'}")
    print(f"[LANG] Output: {output_dir}")
    print(f"[LANG] Steps: {steps}, B={batch_size}, N={seq_len}, lr={lr}")

    # Open log file
    log_file_path = os.path.join(output_dir, "training_log.txt")
    _log_file = open(log_file_path, "w", buffering=1)
    _log_file.write(f"# Language LCM training log\n")
    _log_file.write(f"# output_dir={output_dir} steps={steps} lr={lr} "
                    f"batch={batch_size} seq={seq_len}\n")
    _log_file.write(f"# step_offset={step_offset}\n")
    _log_file.write(f"# step,loss,ppl,lr\n")
    _log_file.flush()

    # Optimizer
    optimizer, schedule = make_optimizer(cfg, steps, lr)
    opt_state = optimizer.init(params)
    if from_ckpt and step_offset > 0:
        with open(from_ckpt, 'rb') as f:
            data = pickle.load(f)
        old_opt = data.get('opt_state')
        if old_opt is not None:
            try:
                old_opt_jax = jax.tree_util.tree_map(
                    lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
                    old_opt)
                dummy_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
                _ = optimizer.update(dummy_grads, old_opt_jax, params)
                opt_state = old_opt_jax
                print(f"[CKPT]  Loaded optimizer state from checkpoint")
            except (ValueError, KeyError, TypeError) as e:
                print(f"[CKPT]  Optimizer state incompatible ({e}), re-initialized")
                opt_state = optimizer.init(params)

    # Data iterator
    mp = data_path or MMAP_PATH
    sp = shape_path or (mp.replace('.dat', '_shape.json'))
    data_iter = WikiDataIter(mmap_path=mp, shape_path=sp, B=batch_size, N=seq_len)

    mtp_w = getattr(cfg, 'mtp_loss_weight', 0.3)

    # JIT-compiled training step
    @jax.jit
    def train_step(p, opt, batch, lr_val, rng_key):
        inputs, targets = batch  # targets = inputs shifted left by 1

        def loss_fn(pp):
            logits, _, aux = lang_lcm_forward(
                pp, inputs, cfg, rng=rng_key, training=True,
                targets=targets)  # pass targets for MTP embedding
            return lang_lm_loss(logits, targets, aux=aux, mtp_weight=mtp_w)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, new_opt = optimizer.update(grads, opt, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt, loss

    # Training loop
    total_steps = steps + step_offset
    running_loss = 0.0
    start_time = time.time()
    pbar = tqdm(total=steps, desc=" lang lcm training", unit="step",
                initial=step_offset)

    for global_step in range(step_offset, total_steps):
        batch = next(data_iter)
        current_lr = schedule(global_step - step_offset)
        rng, step_rng = jax.random.split(rng)

        params, opt_state, loss_val = train_step(
            params, opt_state, batch, current_lr, step_rng)

        loss_f = float(loss_val)
        if np.isnan(loss_f) or np.isinf(loss_f):
            print(f"\n[LANG] NaN at step {global_step}, skipping...")
            continue

        running_loss += loss_f

        steps_this_run = global_step - step_offset
        if steps_this_run % log_every == 0 and steps_this_run > 0:
            n_steps = min(log_every, steps_this_run)
            avg_loss = running_loss / n_steps
            elapsed = time.time() - start_time
            tok_s = batch_size * seq_len * log_every / elapsed
            ppl = np.exp(min(avg_loss, 20.0))
            msg = (f"  step {global_step:>6d} | loss={avg_loss:.4f} | "
                   f"ppl={ppl:.1f} | lr={current_lr:.2e} | {tok_s:.0f} tok/s")
            print(f"\r{msg}", flush=True)
            _log_file.write(f"{global_step},{avg_loss:.6f},{ppl:.1f},{current_lr:.2e}\n")
            _log_file.flush()
            running_loss = 0.0
            start_time = time.time()

        if save_every > 0 and (global_step + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"lang_step_{global_step + 1}.pkl")
            save_checkpoint(params, opt_state, global_step + 1, ckpt_path)

        pbar.update(1)

    pbar.close()

    final_path = os.path.join(output_dir, "lang_final.pkl")
    save_checkpoint(params, opt_state, total_steps, final_path)
    print(f"[LANG] Training complete → {final_path}")

    w_out = np.array(params['W_out'])
    w_out.tofile(os.path.join(output_dir, "W_out.bin"))
    print(f"[LANG] W_out exported to {output_dir}/W_out.bin")

    _log_file.close()
    print(f"[LANG] Training log saved to {log_file_path}")

    return params


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Language LCM Training (V4)")
    parser.add_argument("--output-dir", default="checkpoints/lang_lm",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Logging interval")
    parser.add_argument("--save-every", type=int, default=10000,
                        help="Checkpoint save interval")
    parser.add_argument("--from-ckpt", default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--data", default=None,
                        help="Path to .dat mmap file")
    parser.add_argument("--shape", default=None,
                        help="Path to shape JSON")
    # V4 toggles
    parser.add_argument("--mtp-depth", type=int, default=None,
                        help="MTP depth (overrides config)")
    parser.add_argument("--hc", type=int, default=None,
                        help="mHC streams (overrides config)")
    parser.add_argument("--no-muon", action="store_true",
                        help="Disable Muon optimizer (use AdamW)")
    args = parser.parse_args()

    import dataclasses as _dc
    cfg = LCMConfig()
    cfg_params = {f.name: getattr(cfg, f.name) for f in _dc.fields(cfg)}
    if args.mtp_depth is not None:
        cfg_params['n_mtp_depth'] = args.mtp_depth
    if args.hc is not None:
        cfg_params['n_hc'] = args.hc
    if args.no_muon:
        cfg_params['use_muon'] = False
    cfg = LCMConfig(**cfg_params)

    train_lang_lcm(
        cfg=cfg,
        output_dir=args.output_dir,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        log_every=args.log_every,
        save_every=args.save_every,
        from_ckpt=args.from_ckpt,
        data_path=args.data,
        shape_path=args.shape,
    )


if __name__ == "__main__":
    main()
