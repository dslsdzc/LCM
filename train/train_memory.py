"""Stage 2: 训练记忆 (Train Memory) — Encoder + Codebook Training.

Loads a trained gen_head (decoder) from Stage 1 and completely freezes it.
Trains the encoder and all 6 codebook lattices so they learn to store and
retrieve conceptual knowledge. The gen_head serves as a fixed receiver.

Losses:
  - VQ commitment loss (encoder → codebook alignment)
  - Contrastive NCE loss (codebook discrimination)
  - Manifold orthogonality loss (tangent space regularization)
  - No LM loss (gen_head is frozen)

Usage:
    python -m train.train_memory --data data/zhwiki_tokens.dat \\
        --lm-checkpoint checkpoints/lm_stage1/lm_final.pkl \\
        --steps 20000 --lr 1e-4 --save-every 1000
"""
import argparse
import json
import os
import pickle
import signal
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from train.config import LCMConfig
from train.data import WikiDataIter
from train.model import init_all_params, forward, split_trainable_frozen
from train.losses import compute_vq_loss
from train.lattices import manifold_orth_loss


# ─── Memory-only training step ──────────────────────────────────────────

def make_memory_step(cfg, gvalue_C_pos, gvalue_C_neg):
    """Create a jitted training step that only trains encoder + codebooks.

    gen_head is completely frozen via stop_gradient — no gradients flow to
    it, and LM loss is not computed.

    Only VQ, contrastive, and orthogonality losses are used.
    """
    d = cfg.d_model
    V = cfg.vocab_size
    n_heads = cfg.n_heads

    @jax.jit
    def memory_step(params, ema_state, feature_bank, opt_state, batch, step, rng,
                    self_state=None):
        inputs, targets = batch
        B, N = inputs.shape
        rng, *subkeys = jax.random.split(rng, 5)

        # ── Loss function for gradient computation ────────────────────
        def loss_fn(p):
            z, z_q, logits, aux, _ = forward(
                p, None, inputs, cfg, training=True, rng=subkeys[0],
                self_state=self_state)

            # Freeze gen_head: stop gradients from flowing through it
            p['gen_head'] = lax.stop_gradient(p['gen_head'])
            logits = lax.stop_gradient(logits)

            loss_vq_total = 0.0
            vq_components = compute_vq_loss(p, aux, z, cfg)
            for name, val in vq_components.items():
                loss_vq_total = loss_vq_total + val

            from train.lattices import contrast_info_nce_loss
            loss_contrast = cfg.lambda_contrast * contrast_info_nce_loss(
                p['contrast'], z, tau=0.5)

            if cfg.lambda_orth > 0:
                loss_orth = cfg.lambda_orth * jnp.mean(
                    jnp.sum(p['manifold']['T'] ** 2, axis=(-2, -1)))
            else:
                loss_orth = 0.0

            # Self lattice regularization (if self_state is active)
            loss_self = jnp.array(0.0)
            if self_state is not None and 'self' in p:
                from train.self_lattice import self_lattice_reg_loss
                loss_self = self_lattice_reg_loss(p['self'], aux.get('self_state', self_state))

            components = {
                'vq': loss_vq_total,
                'contrast': loss_contrast,
                'orth': loss_orth,
                'self': loss_self,
            }
            return loss_vq_total + loss_contrast + loss_orth + loss_self, (z, z_q, logits, aux, components)

        (total, (z, z_q, logits, aux, components)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Zero out gen_head gradients (double safety)
        grads['gen_head'] = jax.tree_util.tree_map(jnp.zeros_like, grads['gen_head'])

        updates, new_opt = _OPTIMIZER.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Restore original gen_head (optimizer shouldn't have changed it)
        new_params['gen_head'] = params['gen_head']

        # ── EMA updates (codebook-specific) ───────────────────────────
        new_params, new_ema = _jitted_ema(new_params, ema_state, z)

        # ── Feature bank ──────────────────────────────────────────────
        new_fb = _jitted_feature_bank(feature_bank, z, step)

        return new_params, new_ema, new_fb, new_opt, components, aux

    return memory_step


# ─── EMA + Feature bank helpers (from train.py) ─────────────────────────

@jax.jit
def _jitted_ema(params, ema_state, z):
    """EMA updates for sparsely-managed codebooks.

    按最近码索引做每码本更新：z 只更新其最近码的 N/m 累计量，
    而不是把 batch 总和广播到每个码本行（否则所有码收敛到同一质心）。
    """
    # Sparse
    C_s = params['sparse']['C']
    M_s = C_s.shape[0]
    dists = jnp.sum((z[:, None, :] - C_s[None, :, :]) ** 2, axis=-1)  # (B, M)
    nearest = jnp.argmin(dists, axis=-1)  # (B,)
    onehot = jax.nn.one_hot(nearest, M_s, dtype=jnp.float32)  # (B, M)
    counts = onehot.sum(axis=0)  # (M,)
    sums = onehot.T @ z  # (M, d)

    N_s, m_s = ema_state['sparse']['N'], ema_state['sparse']['m']
    N_s_new = 0.99 * N_s + 0.01 * counts
    m_s_new = 0.99 * m_s + 0.01 * sums
    C_s_new = m_s_new / jnp.clip(N_s_new, 1.0)[:, None]
    lam = 1e-4
    C_s_new = jnp.sign(C_s_new) * jnp.clip(jnp.abs(C_s_new) - lam, 0)
    params['sparse']['C'] = C_s_new

    # Manifold（同上，更新 C 后用 exp_map 回到 Poincaré 球）
    C_m = params['manifold']['C']
    M_m = C_m.shape[0]
    dists_m = jnp.sum((z[:, None, :] - C_m[None, :, :]) ** 2, axis=-1)  # (B, M)
    nearest_m = jnp.argmin(dists_m, axis=-1)  # (B,)
    onehot_m = jax.nn.one_hot(nearest_m, M_m, dtype=jnp.float32)  # (B, M)
    counts_m = onehot_m.sum(axis=0)  # (M,)
    sums_m = onehot_m.T @ z  # (M, d)

    N_m, m_m = ema_state['manifold']['N'], ema_state['manifold']['m']
    N_m_new = 0.99 * N_m + 0.01 * counts_m
    m_m_new = 0.99 * m_m + 0.01 * sums_m
    from train.hyp import exp_map
    C_m_new = exp_map(m_m_new / jnp.clip(N_m_new, 1.0)[:, None])
    params['manifold']['C'] = C_m_new

    new_ema = {
        'sparse': {'N': N_s_new, 'm': m_s_new},
        'manifold': {'N': N_m_new, 'm': m_m_new},
        'binding': ema_state.get('binding', {}),
    }
    return params, new_ema


@jax.jit
def _jitted_feature_bank(feature_bank, z, step):
    """FIFO feature bank update."""
    B = z.shape[0]
    bank = feature_bank['bank']
    ptr = feature_bank['ptr']
    last_used = feature_bank['last_used']
    cap = 4096
    for i in range(B):
        bank = bank.at[ptr % cap].set(z[i])
        ptr = ptr + 1
    return {'bank': bank, 'ptr': ptr, 'last_used': last_used}


# ─── Main training function ─────────────────────────────────────────────

def train_memory(cfg, data_path, shape_path, output_dir,
                 lm_checkpoint=None, steps=20000, lr=1e-4,
                 batch_size=16, seq_len=512,
                 log_every=100, save_every=1000,
                 resume_from=None):
    """Train encoder + codebooks with frozen gen_head.

    Args:
        cfg: LCMConfig.
        data_path: Path to tokenized .dat mmap file.
        shape_path: Path to shape JSON.
        output_dir: Checkpoint save directory.
        lm_checkpoint: Path to Stage 1 LM checkpoint .pkl.
        steps: Number of training steps.
        lr: Learning rate.
        batch_size: Batch size.
        seq_len: Sequence length.
        log_every: Logging interval.
        save_every: Checkpoint save interval.
        resume_from: Optional checkpoint directory to resume from.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(42)

    # ── Init parameters ───────────────────────────────────────────────
    rng, init_rng = jax.random.split(rng)
    if resume_from:
        print(f"[MEMORY] Resuming from {resume_from}")
        from train.checkpoint import load_checkpoint as bin_load
        params, gvalue, opt_state, start_step = bin_load(
            resume_from, cfg=None, rng=rng, load_opt=True)
        import json as _json
        ckpt_cfg_path = os.path.join(resume_from, "config.json")
        if os.path.exists(ckpt_cfg_path):
            with open(ckpt_cfg_path) as f:
                for k, v in _json.load(f).items():
                    if hasattr(cfg, k):
                        object.__setattr__(cfg, k, v)
        # Build EMA state from scratch (not saved in .bin format)
        ema_state = {
            'sparse': {'N': jnp.zeros(cfg.M_sparse), 'm': jnp.zeros((cfg.M_sparse, cfg.d_model))},
            'manifold': {'N': jnp.zeros(cfg.M_man), 'm': jnp.zeros((cfg.M_man, cfg.d_model))},
            'binding': {},
        }
        feature_bank = {
            'bank': jnp.zeros((cfg.bank_capacity, cfg.d_model)),
            'ptr': jnp.array(0),
            'last_used': jnp.zeros(cfg.bank_capacity, dtype=jnp.int32),
        }
        self_state = None
    else:
        print("[MEMORY] Initializing full model parameters...")
        params, gvalue, self_state = init_all_params(cfg, init_rng)
        start_step = 0
        ema_state = {
            'sparse': {'N': jnp.zeros(cfg.M_sparse), 'm': jnp.zeros((cfg.M_sparse, cfg.d_model))},
            'manifold': {'N': jnp.zeros(cfg.M_man), 'm': jnp.zeros((cfg.M_man, cfg.d_model))},
            'binding': {},
        }
        feature_bank = {
            'bank': jnp.zeros((cfg.bank_capacity, cfg.d_model)),
            'ptr': jnp.array(0),
            'last_used': jnp.zeros(cfg.bank_capacity, dtype=jnp.int32),
        }

    # ── Load trained gen_head from Stage 1 ────────────────────────────
    if lm_checkpoint:
        print(f"[MEMORY] Loading trained gen_head from {lm_checkpoint}")
        with open(lm_checkpoint, 'rb') as f:
            lm_ckpt = pickle.load(f)
        gen_head = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if not isinstance(x, (jax.Array,)) else x,
            lm_ckpt['gen_head'])
        params['gen_head'] = gen_head
        print(f"[MEMORY] gen_head loaded (step {lm_ckpt.get('step', '?')}), "
              f"now frozen — only encoder + codebooks will be trained")
    else:
        print("[MEMORY] WARNING: No LM checkpoint loaded! gen_head is randomly initialized.")
        print("  The decoder must be trained first (Stage 1) for meaningful results.")

    # ── Freeze gen_head permanently ───────────────────────────────────
    params_gh_frozen = params['gen_head']
    params['gen_head'] = lax.stop_gradient(params_gh_frozen)
    print(f"[MEMORY] gen_head frozen. Trainable params: "
          f"{sum(p.size for k, v in params.items() if k != 'gen_head' for p in jax.tree_util.tree_leaves(v) if hasattr(p, 'size')):,}")

    # ── Optimizer (only for non-gen_head) ─────────────────────────────
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.1)
    global _OPTIMIZER
    _OPTIMIZER = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=cfg.adam_beta1,
                     b2=cfg.adam_beta2, eps=cfg.adam_eps,
                     weight_decay=cfg.weight_decay),
    )
    opt_state = _OPTIMIZER.init(params)

    # ── Create training step ──────────────────────────────────────────
    gvalue_C_pos = gvalue.C_pos if gvalue is not None else None
    gvalue_C_neg = gvalue.C_neg if gvalue is not None else None
    memory_step = make_memory_step(cfg, gvalue_C_pos, gvalue_C_neg)

    # ── Data ──────────────────────────────────────────────────────────
    data_iter = WikiDataIter(data_path, shape_path, B=batch_size, N=seq_len)

    # ── Graceful shutdown ─────────────────────────────────────────────
    def _handler(sig, frame):
        _save_ckpt(params, gvalue, opt_state, cfg, output_dir, step, self_state=self_state)
        print(f"\nSaved interrupt checkpoint → {output_dir}/")
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nStage 2: {steps} steps, lr={lr}, B={batch_size}, N={seq_len}")
    print(f"  gen_head frozen | losses: VQ + contrast + orth")
    print(f"  checkpoints → {output_dir}/ (every {save_every})")

    from tqdm import tqdm
    from train.monitor import MetricsRecorder
    recorder = MetricsRecorder(save_dir=output_dir, window=50)
    running_loss = {'vq': 0.0, 'contrast': 0.0, 'orth': 0.0, 'self': 0.0}
    start_time = time.time()
    pbar = tqdm(total=steps - start_step, desc="   stage 2 training", unit="step",
                initial=start_step)

    for step in range(start_step, steps):
        batch = next(data_iter)
        rng, step_rng = jax.random.split(rng)

        current_lr = schedule(step - start_step)

        params, ema_state, feature_bank, opt_state, comps, aux = memory_step(
            params, ema_state, feature_bank, opt_state, batch, step, step_rng,
            self_state=self_state)

        # Update self state from forward pass
        if self_state is not None and aux.get('self_state') is not None:
            self_state = aux['self_state']

        # Restore frozen gen_head (memory_step zeros its gradients, but restore to be safe)
        params['gen_head'] = lax.stop_gradient(params_gh_frozen)

        for k in running_loss:
            running_loss[k] += float(comps.get(k, 0.0))

        if step % 50 == 0:
            recorder.record(
                step,
                vq=float(comps.get('vq', 0.0)),
                contrast=float(comps.get('contrast', 0.0)),
                orth=float(comps.get('orth', 0.0)),
                self_loss=float(comps.get('self', 0.0)),
                lr=float(current_lr),
            )

        # Logging
        if step % log_every == 0 and step > 0:
            elapsed = time.time() - start_time
            tok_s = batch_size * seq_len * log_every / elapsed
            avg_vq = running_loss['vq'] / log_every
            avg_ct = running_loss['contrast'] / log_every
            avg_orth = running_loss['orth'] / log_every
            avg_self = running_loss['self'] / log_every
            parts = [f"  step {step:>6d} | vq={avg_vq:.4f} | contrast={avg_ct:.4f} | "
                     f"orth={avg_orth:.4f}"]
            if avg_self > 0:
                parts.append(f"self={avg_self:.6f}")
            parts.append(f"| {tok_s:.0f} tok/s")
            tqdm.write("".join(parts))
            running_loss = {'vq': 0.0, 'contrast': 0.0, 'orth': 0.0, 'self': 0.0}
            start_time = time.time()

        # Save checkpoint
        if save_every > 0 and step > 0 and step % save_every == 0:
            _save_ckpt(params, gvalue, opt_state, cfg, output_dir, step, self_state=self_state)
            recorder.save()

        pbar.update(1)

    pbar.close()
    recorder.save()

    # Final save
    _save_ckpt(params, gvalue, opt_state, cfg, output_dir, steps)
    print(f"[MEMORY] Training complete! Final checkpoint in {output_dir}/")


def _save_ckpt(params, gvalue, opt_state, cfg, output_dir, step, self_state=None):
    """Save full-format checkpoint with memory parameters."""
    from train.checkpoint import save_checkpoint as bin_save
    state = {
        'params': params,
        'gvalue': gvalue,
        'opt_state': opt_state,
        'self_state': self_state,
    }
    bin_save(state, cfg, output_dir=output_dir, step=step)


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: 训练记忆 (Train Memory) — Encoder + Codebook Training")
    parser.add_argument("--data", required=True, help="Path to tokenized .dat file")
    parser.add_argument("--shape", default=None, help="Path to shape JSON")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--lm-checkpoint", default=None,
                        help="Stage 1 LM checkpoint (.pkl) with trained gen_head")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    args = parser.parse_args()

    cfg = LCMConfig()
    shape_path = args.shape or (os.path.splitext(args.data)[0] + "_shape.json")

    train_memory(
        cfg=cfg,
        data_path=args.data,
        shape_path=shape_path,
        output_dir=args.output_dir,
        lm_checkpoint=args.lm_checkpoint,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        log_every=args.log_every,
        save_every=args.save_every,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
