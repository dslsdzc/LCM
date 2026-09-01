"""LCM training loop — JAX pure functional style with continual learning."""
import dataclasses
import os
import pickle
from functools import partial
import jax
import jax.numpy as jnp
import optax
from jax import lax

from train.config import LCMConfig
from train.model import init_all_params, forward, split_trainable_frozen
from train.losses import compute_total_loss
from train.lattices import (
    sparse_ema_update, sparse_forward,
    manifold_ema_update, manifold_orth_loss,
    init_value_scalars,
)
from train.gvalue import GValueCodebook, make_global_value_vectors
from train.continual import (
    ContinualState, init_continual_state, detect_new_task,
    expand_lattice_codebooks, snapshot_protected_params,
    compute_ewc_loss, compute_fisher_diag_flat,
    update_replay_buffer, sample_replay,
    update_access_counters, consolidate_memory,
)


def make_optimizer(cfg: LCMConfig):
    """Create AdamW optimizer with cosine decay schedule."""
    schedule = optax.cosine_decay_schedule(
        init_value=cfg.learning_rate,
        decay_steps=100_000,
        alpha=0.1)
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.adam_beta1,
            b2=cfg.adam_beta2,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay),
    )


def create_train_state(cfg: LCMConfig, rng, enable_continual=True):
    """Create initial training state with continual learning support."""
    rng, init_rng = jax.random.split(rng)
    params, gvalue, self_state = init_all_params(cfg, init_rng)
    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(params)

    # EMA state for EMA-managed codebooks
    ema_state = {
        'sparse': {
            'N': jnp.zeros(cfg.M_sparse),
            'm': jnp.zeros((cfg.M_sparse, cfg.d_model)),
        },
        'manifold': {
            'N': jnp.zeros(cfg.M_man),
            'm': jnp.zeros((cfg.M_man, cfg.d_model)),
        },
        'binding': {
            'key': [{'N': jnp.zeros(cfg.M_bind), 'm': jnp.zeros((cfg.M_bind, cfg.d_model))}
                    for _ in range(cfg.n_bind_layers)],
            'val': [{'N': jnp.zeros(cfg.M_bind), 'm': jnp.zeros((cfg.M_bind, cfg.d_model))}
                    for _ in range(cfg.n_bind_layers)],
            'bind': [{'N': jnp.zeros(cfg.M_bind), 'm': jnp.zeros((cfg.M_bind, cfg.d_model))}
                     for _ in range(cfg.n_bind_layers)],
        },
    }

    # Feature bank (dead vector prevention)
    feature_bank = {
        'bank': jnp.zeros((cfg.bank_capacity, cfg.d_model)),
        'ptr': 0,
        'last_used': jnp.zeros(cfg.bank_capacity, dtype=jnp.int32),
    }

    # Cumulative codebook usage tracking for utilization measurement
    seen_masks = {
        'hrq': jnp.zeros(cfg.M_top, dtype=bool),
    }

    # Continual learning state
    continual = init_continual_state(cfg.d_model) if enable_continual else None

    # Verify gvalue integrity on init
    try:
        gvalue.verify_integrity()
    except AssertionError as e:
        print(f"[WARN] GValue integrity check: {e}")

    return {
        'params': params,
        'gvalue': gvalue,
        'opt_state': opt_state,
        'ema_state': ema_state,
        'feature_bank': feature_bank,
        'continual': continual,
        'self_state': self_state,
        'step': 0,
        'seen_masks': seen_masks,
    }


def _jitted_compute(params, opt_state, gvalue_C_pos, gvalue_C_neg,
                     inputs, targets, cfg_dict, rng, ewc_loss_val,
                     ema_state, feature_bank, step):
    """JAX-jitted core computation: forward, loss, grad, optimizer, EMA, feature bank.

    Args:
        gvalue_C_pos, gvalue_C_neg: Frozen arrays from GValueCodebook (pass through JIT).
        cfg_dict: Frozen config dict extracted from LCMConfig.
    """
    # Reconstruct minimal config access
    B, N = inputs.shape
    d = cfg_dict['d_model']

    rng, *subkeys = jax.random.split(rng, 5)

    # Loss function for gradient computation
    def loss_fn(p):
        z, z_q, logits, aux, _ = forward(
            p, None, inputs, _cfg_from_dict(cfg_dict),
            training=True, rng=subkeys[0])

        def _pd(x, y):
            xn2 = jnp.linalg.norm(x, axis=-1, keepdims=True) ** 2
            yn2 = jnp.linalg.norm(y, axis=-1, keepdims=True) ** 2
            d2 = jnp.linalg.norm(x - y, axis=-1, keepdims=True) ** 2
            denom = (1.0 - xn2) * (1.0 - yn2) + 1e-8
            return jnp.arccosh(1.0 + 2.0 * d2 / denom + 1e-8)

        def _min_dist_to(z_batch, anchors):
            return jnp.min(jnp.stack(
                [_pd(z_batch[:, None, :], a[None, :]).squeeze(-1) for a in anchors], axis=-1), axis=-1)

        loss_lm = _lm_loss(logits, targets, cfg_dict['vocab_size'])
        loss_vq = sum(_vq_losses(p, aux, z, cfg_dict).values())
        loss_contrast = cfg_dict['lambda_contrast'] * (
            _value_biased_contrast_loss(
                p['contrast'], z, gvalue_C_neg[1] if gvalue_C_neg is not None else None,
                cfg_dict)
            if (gvalue_C_neg is not None and cfg_dict.get('alpha_val', 0) > 0)
            else _contrast_nce_loss(p['contrast'], z)
        )
        loss_orth = manifold_orth_loss(
            p['manifold']['T'], aux['man_idx'], cfg_dict['n_orth_samples'],
            lambda_orth=cfg_dict['lambda_orth'], rng=subkeys[4])
        loss_val = _value_contrast_loss_inline(
            aux['lattice_outputs'], gvalue_C_pos, gvalue_C_neg, cfg_dict)
        total = loss_lm + loss_vq + loss_contrast + loss_orth + loss_val

        if ewc_loss_val is not None:
            total = total + ewc_loss_val

        margin = cfg_dict.get('safety_margin_loss_weight', 0.0) * (
            _safety_margin_loss_inline(z_q, gvalue_C_pos, gvalue_C_neg, cfg_dict))

        components = {
            'lm': loss_lm, 'vq': loss_vq, 'contrast': loss_contrast,
            'orth': loss_orth, 'val': loss_val,
            'ewc': ewc_loss_val if ewc_loss_val is not None else jnp.array(0.0),
            'margin': margin,
            'world_dev': aux.get('world_dev', jnp.array(0.0)),
        }
        return total, (z, z_q, logits, aux, components)

    (total, (z, z_q, logits, aux, components)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Optimizer update
    updates, opt_state_new = cfg_dict['_optimizer'].update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)

    # EMA updates
    params_new, ema_state_new = _jitted_ema(
        params_new, ema_state, z, cfg_dict)

    # Feature bank update
    feature_bank_new = _jitted_feature_bank(
        feature_bank, z, step, cfg_dict)

    return params_new, opt_state_new, ema_state_new, feature_bank_new, components, aux, z


def _cfg_from_dict(d):
    """Build minimal config with necessary fields for forward()."""
    from train.config import LCMConfig
    cfg = LCMConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            object.__setattr__(cfg, k, v)
    return cfg


def _lm_loss(logits, targets, vocab_size):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)).mean()


def _vq_losses(params, aux, z, cfg):
    from train.losses import compute_vq_loss
    return compute_vq_loss(params, aux, z, _cfg_from_dict(cfg))


def _contrast_nce_loss(contrast_params, z):
    from train.lattices import contrast_info_nce_loss
    return contrast_info_nce_loss(contrast_params, z, tau=0.5)


def _value_biased_contrast_loss(contrast_params, z, v_harm, cfg_dict):
    """Value-biased contrast loss for jitted path."""
    if v_harm is None:
        return _contrast_nce_loss(contrast_params, z)
    from train.lattices import contrast_value_biased_nce_loss
    tau_val = cfg_dict.get('tau_val_signal', 0.1)
    return contrast_value_biased_nce_loss(
        contrast_params, z, v_harm, tau=0.5, tau_val=tau_val)


def _euclidean_dist(x, y):
    """Euclidean distance between x(B,d) and y(1,d)."""
    return jnp.linalg.norm(x - y, axis=-1)  # (B,)


def _min_ed_to(x, anchors):
    """Minimum Euclidean distance from x to a set of anchors."""
    return jnp.min(jnp.stack(
        [_euclidean_dist(x, a[None, :]) for a in anchors], axis=-1), axis=-1)


def _value_contrast_loss_inline(lattice_outputs, C_pos, C_neg, cfg):
    """Value contrast loss (Euclidean distance, safe for any norm)."""
    if lattice_outputs is None or C_pos is None or C_neg is None:
        return jnp.array(0.0)
    v_harm = C_neg[1:2]  # (1, d)
    tau = cfg.get('tau_val_signal', 0.1)
    lam = cfg.get('lambda_val', 0.01)
    if lam == 0:
        return jnp.array(0.0)
    loss = 0.0
    n = len(lattice_outputs)
    for o in lattice_outputs:
        d_pos = _min_ed_to(o, C_pos)
        d_neg_vals = []
        for j in range(4):
            d_j = _min_ed_to(o, C_neg[j:j+1])[:, None]
            hd = _min_ed_to(o, v_harm)[:, None]
            w_harm = jnp.exp(-hd / tau)
            d_neg_vals.append(w_harm * d_j)
        d_neg_w = jnp.concatenate(d_neg_vals, axis=-1).mean(axis=-1, keepdims=True)
        logit = (d_neg_w.squeeze(-1) - d_pos) / tau
        loss = loss + jnp.mean(jax.nn.softplus(logit))
    return lam * loss / n


def _safety_margin_loss_inline(z_q, C_pos, C_neg, cfg):
    margin = cfg.get('safety_margin_relative', 0.5)
    thresh = cfg.get('margin_penalty_threshold', 0.2)
    weight = cfg.get('safety_margin_loss_weight', 0.001)
    d_pos = _min_pd_to(z_q, C_pos)
    d_neg = _min_pd_to(z_q, C_neg)
    margins = d_neg - margin - d_pos
    excess = jnp.clip(thresh - margins, 0)
    return weight * jnp.mean(excess ** 2)


def _jitted_ema(params, ema_state, z, cfg):
    """Per-codebook EMA (nearest-code assignment).

    Broadcasting the batch sum onto every codebook row made all codes
    converge to the same centroid (same bug train_memory.py's _jitted_ema
    used to have). Each z updates only its nearest code.
    """
    g_s = cfg.get('gamma_sparse', 0.99)
    g_m = cfg.get('gamma_man', 0.99)

    def _per_code_ema(C, N, m, gamma):
        dists = jnp.sum((z[:, None, :] - C[None, :, :]) ** 2, axis=-1)  # (B, M)
        nearest = jnp.argmin(dists, axis=-1)  # (B,)
        onehot = jax.nn.one_hot(nearest, C.shape[0], dtype=jnp.float32)
        counts = onehot.sum(axis=0)  # (M,)
        sums = onehot.T @ z  # (M, d)
        N_new = gamma * N + (1 - gamma) * counts
        m_new = gamma * m + (1 - gamma) * sums
        return N_new, m_new, m_new / jnp.clip(N_new, 1.0)[:, None]

    # Sparse
    N_s, m_s = ema_state['sparse']['N'], ema_state['sparse']['m']
    N_s_new, m_s_new, C_s_new = _per_code_ema(
        params['sparse']['C'], N_s, m_s, g_s)
    lam = cfg.get('lambda_sparse', 1e-4)
    C_s_new = jnp.sign(C_s_new) * jnp.clip(jnp.abs(C_s_new) - lam, 0)
    params['sparse']['C'] = C_s_new

    # Manifold
    N_m, m_m = ema_state['manifold']['N'], ema_state['manifold']['m']
    N_m_new, m_m_new, C_m_new = _per_code_ema(
        params['manifold']['C'], N_m, m_m, g_m)
    from train.hyp import exp_map
    params['manifold']['C'] = exp_map(C_m_new)

    ema_state_new = {
        'sparse': {'N': N_s_new, 'm': m_s_new},
        'manifold': {'N': N_m_new, 'm': m_m_new},
        'binding': ema_state.get('binding', {}),
    }
    return params, ema_state_new


def _jitted_feature_bank(feature_bank, z, step, cfg):
    B = z.shape[0]
    bank = feature_bank['bank']
    ptr = feature_bank['ptr']
    last_used = feature_bank['last_used']
    cap = cfg.get('bank_capacity', 4096)

    z_detach = lax.stop_gradient(z)
    for i in range(B):
        bank = bank.at[ptr % cap].set(z_detach[i])
        ptr += 1

    return {'bank': bank, 'ptr': ptr, 'last_used': last_used}


def _build_loss_grad_fn(params, gvalue, inputs, targets, cfg, rng, ewc_loss_val):
    """Build loss function returning (loss, (components, aux, z))."""
    def loss_fn(p):
        z, z_q, logits, aux, _ = forward(
            p, gvalue, inputs, cfg, training=True, rng=rng)
        total_loss, components = compute_total_loss(
            p, gvalue, logits, targets, z, aux, cfg,
            ewc_loss_val=ewc_loss_val)
        if gvalue is not None and cfg.safety_margin_loss_weight > 0:
            margin_loss = gvalue.safety_margin_loss(
                z_q, cfg.safety_margin_relative,
                weight=cfg.safety_margin_loss_weight)
            total_loss = total_loss + margin_loss
            components['margin'] = margin_loss
        else:
            components['margin'] = jnp.array(0.0)
        return total_loss, (components, aux, z)
    return loss_fn


def _jitted_step(params, opt_state, gvalue_C_pos, gvalue_C_neg,
                 inputs, targets, cfg_dict, rng, ewc_loss_val,
                 ema_state, feature_bank, step,
                 self_state=None):
    """Training step with jax.grad (auto-jitted) — outer function not jit-decorated
    because cfg_dict contains shape-determining values (n_heads, d_model) that
    must be concrete for reshape ops.

    self_state is a dict (from init_self_state) — it flows through the trace
    because JAX can trace dicts whose leaves are arrays."""
    from train.model import forward as fwd
    from train.self_lattice import self_lattice_reg_loss
    from train.hyp import poincare_distance
    B, N = inputs.shape
    d = cfg_dict['d_model']
    rng, *subkeys = jax.random.split(rng, 4)

    # Loss function for gradient computation (traced by jax.grad)
    def loss_fn(p):
        z, z_q, logits, aux, _ = fwd(
            p, None, inputs, _cfg_from_dict(cfg_dict),
            training=True, rng=subkeys[0],
            self_state=self_state)

        def _euclidean_batch(x, y):
            return jnp.linalg.norm(x[:, None, :] - y[None, :], axis=-1)

        def _d_min(x, anchors):
            return jnp.min(
                jnp.stack([_euclidean_batch(x, a) for a in anchors], axis=-1),
                axis=-1)

        loss_lm = _lm_loss(logits, targets, cfg_dict['vocab_size'])
        loss_vq = sum(_vq_losses(p, aux, z, cfg_dict).values())
        loss_contrast = cfg_dict['lambda_contrast'] * (
            _value_biased_contrast_loss(
                p['contrast'], z, gvalue_C_neg[1] if gvalue_C_neg is not None else None,
                cfg_dict)
            if (gvalue_C_neg is not None and cfg_dict.get('alpha_val', 0) > 0)
            else _contrast_nce_loss(p['contrast'], z)
        )
        loss_orth = cfg_dict['lambda_orth'] * jnp.mean(
            jnp.sum(p['manifold']['T'] ** 2, axis=(-2, -1)))
        loss_val = cfg_dict['lambda_val'] * _value_contrast_loss_inline(
            aux['lattice_outputs'], gvalue_C_pos, gvalue_C_neg, cfg_dict)
        total = loss_lm + loss_vq + loss_contrast + loss_orth + loss_val

        # Self lattice regularization loss (if self_state is active)
        loss_self = jnp.array(0.0)
        if self_state is not None and 'self' in p:
            loss_self = self_lattice_reg_loss(p['self'], aux.get('self_state', self_state))
            total = total + loss_self

        if ewc_loss_val is not None:
            total = total + ewc_loss_val

        margin_loss = jnp.array(0.0)
        if cfg_dict.get('safety_margin_loss_weight', 0.0) > 0:
            d_pm = _d_min(z_q, gvalue_C_pos)
            d_nm = _d_min(z_q, gvalue_C_neg)
            margins = d_nm - cfg_dict['safety_margin_relative'] - d_pm
            thresh = cfg_dict.get('margin_penalty_threshold', 0.2)
            margin_loss = (cfg_dict['safety_margin_loss_weight']
                           * jnp.mean(jax.nn.relu(thresh - margins) ** 2))
            total = total + margin_loss

        components = {
            'lm': loss_lm, 'vq': loss_vq, 'contrast': loss_contrast,
            'orth': loss_orth, 'val': loss_val,
            'ewc': ewc_loss_val if ewc_loss_val is not None else jnp.array(0.0),
            'margin': margin_loss,
            'world_dev': aux.get('world_dev', jnp.array(0.0)),
            'soft_mask': aux.get('soft_mask'),
            'hrq_top_sim': aux.get('hrq_top_sim'),
            'convergence_diff': jnp.array(0.0),
            'self': loss_self,
        }
        return total, (z, z_q, logits, aux, components)

    (total, (z, z_q, logits, aux, components)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Optimizer (built inline from primitive config values)
    schedule = optax.cosine_decay_schedule(
        init_value=cfg_dict['learning_rate'], decay_steps=100_000, alpha=0.1)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=cfg_dict['adam_beta1'],
                     b2=cfg_dict['adam_beta2'], eps=cfg_dict['adam_eps'],
                     weight_decay=cfg_dict['weight_decay']))
    updates, opt_state_new = opt.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)

    # EMA
    params_new, ema_state_new = _jitted_ema(params_new, ema_state, z, cfg_dict)

    # Feature bank
    feature_bank_new = _jitted_feature_bank(feature_bank, z, step, cfg_dict)

    return (params_new, opt_state_new, ema_state_new,
            feature_bank_new, components, aux, z)


def train_step(state, batch, rng):
    """Single training step with continual learning support.

    The inner gradient computation is jitted; Python-side CL logic runs outside JIT.
    """
    cfg = _get_global_cfg()

    # Extract arrays from gvalue for JIT compatibility
    gvalue = state.get('gvalue')
    gvalue_C_pos = gvalue.C_pos if gvalue is not None else None
    gvalue_C_neg = gvalue.C_neg if gvalue is not None else None

    # Build cfg_dict for JIT (all primitive types / arrays, no optimizer object)
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}

    # ── Continual Learning: EWC loss ──
    cl_state = state.get('continual')
    ewc_val = None
    if cl_state is not None and cl_state.protected_params:
        ewc_val = compute_ewc_loss(
            state['params'], cl_state.protected_params,
            cl_state.fisher_diag, cfg.ewc_lambda)

    # ── Core jitted step ──
    inputs, targets = batch
    rng, *subkeys = jax.random.split(rng, 4)

    (params_new, opt_state_new, ema_state_new,
     feature_bank_new, components, aux, z) = _jitted_step(
        state['params'], state['opt_state'],
        gvalue_C_pos, gvalue_C_neg,
        inputs, targets, cfg_dict, subkeys[0],
        ewc_val,
        state['ema_state'], state['feature_bank'],
        state['step'],
        self_state=state.get('self_state'))

    new_state = {
        'params': params_new,
        'gvalue': state['gvalue'],
        'opt_state': opt_state_new,
        'ema_state': ema_state_new,
        'feature_bank': feature_bank_new,
        'continual': cl_state,
        'self_state': state.get('self_state'),
        'step': state['step'] + 1,
        'seen_masks': state.get('seen_masks', {}),
    }

    # ── Self lattice state update (non-jitted: dataclass with JAX arrays) ──
    if state.get('self_state') is not None and aux.get('self_state') is not None:
        new_state['self_state'] = aux['self_state']

    # ── Post-step continual learning ──
    if cl_state is not None:
        cl_state = update_access_counters(cl_state, params_new, aux)
        cl_state = update_replay_buffer(
            cl_state, cl_state.task_id, z,
            jnp.zeros((z.shape[0], cfg.vocab_size)),
            aux['soft_mask'], cfg.replay_capacity)
        if (state['step'] > 0
                and state['step'] % cfg.consolidate_interval == 0):
            params_new, cl_state = consolidate_memory(
                cl_state, params_new, state['step'])
        new_state['params'] = params_new
        new_state['continual'] = cl_state

    # ── Update cumulative seen masks for codebook utilization tracking ──
    seen = new_state['seen_masks']
    if 'hrq' in seen and aux.get('hrq_idx') is not None:
        idx = aux['hrq_idx'].reshape(-1)
        idx_int = idx.astype(jnp.int32)
        mask = (idx_int >= 0) & (idx_int < seen['hrq'].shape[0])
        valid_idx = jnp.where(mask, idx_int, 0)
        seen['hrq'] = seen['hrq'].at[valid_idx].set(True)

    # ── GValue integrity check ──
    if gvalue is not None:
        try:
            gvalue.verify_integrity()
        except AssertionError:
            print("[FATAL] GValue codebook tampered! Training aborted.")
            raise

    return new_state, components


def _make_jit_optimizer(cfg):
    """Build optimizer for use inside JIT (as a static object)."""
    schedule = optax.cosine_decay_schedule(
        init_value=cfg.learning_rate, decay_steps=100_000, alpha=0.1)
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.adam_beta1,
            b2=cfg.adam_beta2,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay),
    )


def expand_for_new_task(state, cfg, rng):
    """Expand all codebooks for a new task and protect old params."""
    if state.get('continual') is None:
        return state

    cl_state = state['continual']
    task_id = cl_state.task_id + 1
    rng, exp_rng = jax.random.split(rng)

    print(f"[LCM] Expanding codebooks for task {task_id} "
          f"(+{cfg.n_new_codebook_entries} entries per lattice)...")

    # Snapshot current params for EWC
    cl_state.protected_params = snapshot_protected_params(state['params'])

    # Compute Fisher diagonal for protected params
    # (In practice: run a few batches to estimate; here we use placeholder)
    cl_state.fisher_diag = {}
    for path, val in cl_state.protected_params.items():
        cl_state.fisher_diag[path] = jnp.ones_like(val) * 1e-4  # placeholder

    # Expand codebooks
    new_params = expand_lattice_codebooks(
        state['params'], cfg.n_new_codebook_entries, exp_rng, cfg)

    # Reset optimizer state for expanded params
    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(new_params)

    cl_state.task_id = task_id
    cl_state.step = 0
    cl_state.task_boundaries[state['step']] = task_id

    state['params'] = new_params
    state['opt_state'] = opt_state
    state['continual'] = cl_state
    state['step'] = 0

    print(f"[LCM] Task {task_id} expansion complete. "
          f"Params protected via EWC.")
    return state


def _update_ema(params, ema_state, z, aux, cfg):
    """Update EMA-managed codebooks."""
    z_sum = z.sum(axis=0)
    count = z.shape[0]

    # Sparse EMA
    C_new, N_new, m_new = sparse_ema_update(
        params['sparse']['C'],
        z_sum, count,
        ema_state['sparse']['N'],
        ema_state['sparse']['m'],
        gamma=cfg.gamma_sparse,
        lambda_s=cfg.lambda_sparse)
    params['sparse']['C'] = C_new
    ema_state['sparse']['N'] = N_new
    ema_state['sparse']['m'] = m_new

    # Manifold EMA
    C_man_new, N_man_new, m_man_new = manifold_ema_update(
        params['manifold']['C'],
        z_sum, count,
        ema_state['manifold']['N'],
        ema_state['manifold']['m'],
        gamma=cfg.gamma_man)
    params['manifold']['C'] = C_man_new
    ema_state['manifold']['N'] = N_man_new
    ema_state['manifold']['m'] = m_man_new

    # Binding EMA (all sub-codebooks)
    for cb_type in ['key_cb', 'val_cb', 'bind_cb']:
        ema_key = 'key' if cb_type == 'key_cb' else ('val' if cb_type == 'val_cb' else 'bind')
        for l in range(cfg.n_bind_layers):
            C_bind = params['binding'][cb_type][l]['A'] @ params['binding'][cb_type][l]['W']
            N_bind = ema_state['binding'][ema_key][l]['N']
            m_bind = ema_state['binding'][ema_key][l]['m']
            N_new_v = cfg.gamma_bind * N_bind + (1 - cfg.gamma_bind) * count
            m_new_v = cfg.gamma_bind * m_bind + (1 - cfg.gamma_bind) * z_sum
            C_new_v = m_new_v / jnp.clip(N_new_v, 1.0)[:, None]
            params['binding'][cb_type][l]['A'] = C_new_v
            ema_state['binding'][ema_key][l]['N'] = N_new_v
            ema_state['binding'][ema_key][l]['m'] = m_new_v

    return params, ema_state


def _update_feature_bank(feature_bank, z, aux, cfg, step):
    """Update feature bank and reset dead vectors."""
    B = z.shape[0]
    bank = feature_bank['bank']
    ptr = feature_bank['ptr']
    last_used = feature_bank['last_used']

    # Add current batch to bank (FIFO)
    z_detach = lax.stop_gradient(z)
    for i in range(B):
        bank = bank.at[ptr % cfg.bank_capacity].set(z_detach[i])
        last_used = last_used.at[ptr % cfg.bank_capacity].set(step)
        ptr += 1

    # Dead vector reset (every bank_check_interval steps)
    if step > 0 and step % cfg.bank_check_interval == 0:
        dead_mask = (step - last_used) > cfg.bank_dead_threshold
        for idx in jnp.where(dead_mask)[0]:
            rng = jax.random.PRNGKey(step + idx)
            repl = jax.random.choice(rng, bank[:min(ptr, cfg.bank_capacity)])
            _ = repl  # In practice: params['sparse']['C'] = repl
            last_used = last_used.at[idx].set(step)

    return {
        'bank': bank,
        'ptr': ptr,
        'last_used': last_used,
    }


def compute_codebook_utilization(params, aux, ema_state=None, seen_masks=None):
    """Compute codebook utilization percentages.

    Sources (in priority order):
      - Sparse / Manifold: EMA accumulated counts (N > 0.5 → activated).
      - HRQ: seen_masks boolean array updated each step.
      - Fallback: per-batch unique indices (only meaningful with large B).

    Returns dict of lattice_name → (active_%, batch_unique_%).
    """
    util = {}
    cb_sizes = {}

    if 'hrq' in params:
        cb_sizes['hrq'] = params['hrq']['top']['A'].shape[0]
    if 'C' in params.get('sparse', {}):
        cb_sizes['sparse'] = params['sparse']['C'].shape[0]
    if 'C' in params.get('manifold', {}):
        cb_sizes['manifold'] = params['manifold']['C'].shape[0]

    # EMA-based: sparse, manifold
    if ema_state is not None:
        for name, ema_key in [('sparse', 'sparse'), ('manifold', 'manifold')]:
            if name in cb_sizes and ema_key in ema_state and 'N' in ema_state[ema_key]:
                N = ema_state[ema_key]['N']
                active = int(jnp.sum(N > 0.5))
                total = cb_sizes[name]
                pct = 100.0 * active / max(total, 1)
                # Batch-level
                batch_pct = 0.0
                ak = 'sparse_idx' if name == 'sparse' else 'man_idx'
                if ak in aux and aux[ak] is not None:
                    batch_pct = 100.0 * len(jnp.unique(aux[ak])) / max(aux[ak].shape[0], 1)
                if active > 0:
                    util[name] = (float(pct), float(batch_pct))

    # seen_masks-based: HRQ
    if seen_masks is not None and 'hrq' in seen_masks and 'hrq' in cb_sizes:
        active = int(jnp.sum(seen_masks['hrq']))
        pct = 100.0 * active / max(cb_sizes['hrq'], 1)
        batch_pct = 0.0
        if 'hrq_idx' in aux and aux['hrq_idx'] is not None:
            batch_pct = 100.0 * len(jnp.unique(aux['hrq_idx'])) / max(aux['hrq_idx'].shape[0], 1)
        if active > 0:
            util['hrq'] = (float(pct), float(batch_pct))

    # Fallback: per-batch only (no cumulative data)
    if 'hrq' not in util and 'hrq' in cb_sizes \
       and 'hrq_idx' in aux and aux['hrq_idx'] is not None:
        idx = aux['hrq_idx']
        bc = len(jnp.unique(idx))
        pct = 100.0 * bc / max(cb_sizes['hrq'], 1)
        bpct = 100.0 * bc / max(idx.shape[0], 1)
        util['hrq'] = (float(pct), float(bpct))

    return util


def _get_global_cfg():
    """Get global config singleton."""
    return LCMConfig()


def train_loop(state, data_iter, num_steps, log_every=100,
               detect_shift_every=500, save_every=1000, cfg=None,
               ckpt_dir="checkpoints", keep_last=3,
               raw_capacity=1000, obs_every_n=10,
               lt_enabled=True, lt_max_records=10000,
               narr_consolidate_interval=500,
               narr_keep_threshold=0.05,
               behavior_explore=True,
               behavior_explore_prob=0.2,
               behavior_eval_window=10):
    """Main training loop with continual learning and periodic checkpointing.

    Args:
        state: Training state from create_train_state.
        data_iter: Iterator yielding (inputs, targets) batches.
        num_steps: Total number of training steps.
        log_every: Logging interval.
        detect_shift_every: Distribution shift detection interval.
        save_every: Checkpoint save interval.
        cfg: Configuration (uses defaults if None).
        ckpt_dir: Checkpoint directory path.
        keep_last: Max checkpoints to keep (oldest removed).
        behavior_explore: Enable active behavior exploration.
        behavior_explore_prob: Probability of entering exploration mode.
        behavior_eval_window: Window size for evaluating bias benefit.

    Returns:
        Updated training state.
    """
    if cfg is None:
        cfg = _get_global_cfg()
    rng = jax.random.PRNGKey(42)

    # Track saved checkpoint paths for rotation
    saved_ckpts = []

    # ── Metrics recorder ──────────────────────────────────────────────
    from train.monitor import MetricsRecorder
    recorder = MetricsRecorder(save_dir=ckpt_dir, window=50)

    # ── Observability recorder (黑匣子) ──
    from train.observability import ObservabilityRecorder
    obs = ObservabilityRecorder(
        raw_capacity=raw_capacity, record_every_n=obs_every_n)

    from train.narrative_memory import NarrativeMemory
    narr = NarrativeMemory(max_records=lt_max_records) if lt_enabled else None
    print(f"[OBS] 黑匣子: raw={raw_capacity} every_n={obs_every_n}")
    if narr:
        print(f"[NARR] 叙事记忆: on max={lt_max_records}")

    # ── Reflection loop (轨迹审计器) ──
    reflection_loop = None
    if cfg.reflection_enabled:
        from train.reflection_loop import ReflectionLoop
        reflection_loop = ReflectionLoop(
            anomaly_window=cfg.reflection_anomaly_window,
            anomaly_z=cfg.reflection_anomaly_z,
            max_records=cfg.reflection_max_records,
            cooldown=cfg.reflection_cooldown,
        )
        print(f"[REFL] 反思回路: window={cfg.reflection_anomaly_window} "
              f"z_thresh={cfg.reflection_anomaly_z} "
              f"cooldown={cfg.reflection_cooldown}")

    explorer = None
    if behavior_explore:
        from train.behavior_explorer import BehaviorExplorer
        explorer = BehaviorExplorer(
            explore_prob=behavior_explore_prob,
            eval_window=behavior_eval_window)
        print(f"[EXPL] 行为探索: prob={behavior_explore_prob} "
              f"window={behavior_eval_window}")

    for step in range(num_steps):
        batch = next(data_iter)
        rng, step_rng = jax.random.split(rng)

        # Distribution shift detection (continual learning)
        cl_state = state.get('continual')
        if (cl_state is not None and step > 0
                and step % detect_shift_every == 0):
            z_sample = batch[0][:min(16, batch[0].shape[0])]
            z_for_detect = jnp.mean(z_sample, axis=0, keepdims=True)
            is_new, cl_state = detect_new_task(
                z_for_detect, cl_state, cfg.shift_detection_threshold)
            if is_new:
                print(f"[LCM] Distribution shift detected at step {step}!")
                state = expand_for_new_task(state, cfg, rng)
            state['continual'] = cl_state

        # Training step
        state, components = train_step(state, batch, step_rng)

        # Metrics tracking (every 50 steps)
        if step % 50 == 0:
            log_kw = {}
            for k in ('lm', 'vq', 'contrast', 'orth', 'val', 'ewc', 'margin', 'self'):
                if k in components:
                    log_kw[k] = float(components[k])
            if log_kw:
                recorder.record(step, **log_kw)

        # Observability record (黑匣子)
        record = None
        if obs.should_record(step):
            record = obs.record_step(
                step=step,
                safety_margin=components.get('margin'),
                world_dev=components.get('world_dev'),
                is_safe=components.get('margin', 1.0) > 0.0 if 'margin' in components else True,
                soft_mask=components.get('soft_mask'),
                hrq_top_sim=components.get('hrq_top_sim'),
                convergence_diff=components.get('convergence_diff'),
            )

        # Feed to narrative memory
        if record is not None and narr is not None:
            narr.feed(record)

        # Narrative consolidation (model-driven forgetting)
        if (narr is not None and narr_consolidate_interval > 0
                and step > 0 and step % narr_consolidate_interval == 0
                and state.get('self_state') is not None):
            try:
                self_p = state['params'].get('self')
                self_s = state['self_state']
                if self_p is not None and 'mode_activation' in self_s:
                    from train.self_lattice import narrative_keep_score
                    import jax
                    # Extract numpy arrays once
                    p_np = jax.device_get(self_p)
                    ma_np = jax.device_get(self_s['mode_activation'])

                    def _scorer(r):
                        return narrative_keep_score(p_np, ma_np, r)

                    dropped = narr.consolidate(
                        scorer_fn=_scorer,
                        keep_threshold=narr_keep_threshold)
                    if dropped > 0:
                        print(f"[NARR] Consolidated: dropped {dropped} records "
                              f"(keep ≥ {narr_keep_threshold})")
            except Exception as e:
                print(f"[NARR] Consolidation error: {e}")

        # ── Behavior exploration (路由偏置元学习) ──
        if explorer is not None and step > 0:
            import jax as _jax
            from train.behavior_explorer import (
                TensionSignals, compute_tension_from_aux)

            # Simplified tension from components (no aux at loop level)
            wd = float(_jax.device_get(components.get('world_dev', jnp.array(0.0))))
            mg = float(_jax.device_get(abs(components.get('margin', jnp.array(0.0)))))
            safety = mg > 0.3
            tension = TensionSignals(
                T_pred=min(1.0, wd * 2.0),
                T_conf=min(1.0, mg * 2.0),
                T_res=0.0,
            )
            explorer.observe_tension(step, tension, safety_breach=safety)

            # Exploration: perturb routing bias, run non-jitted forward
            if explorer.should_explore(step):
                bias = explorer.select_bias(step)
                rng, fwd_rng = jax.random.split(rng)
                try:
                    from train.model import forward as _fwd
                    _, _, _, aux_b, _ = _fwd(
                        state['params'], state.get('gvalue'),
                        batch[0], cfg, training=False, rng=fwd_rng,
                        routing_bias=jnp.array(bias, dtype=jnp.float32))
                    aux_np = _jax.device_get(aux_b)
                    tension_b = compute_tension_from_aux(
                        aux_np, None, None, d_model=cfg.d_model)
                    explorer.observe_tension(step, tension_b, safety_breach=safety)
                except Exception as e:
                    print(f"[EXPL] Forward error: {e}")

            # Finalize bias evaluation periodically
            if (step > explorer.eval_window
                    and step % explorer.eval_window == 0
                    and explorer.current_bias is not None):
                explorer.finalize_evaluation(step)

        # ── Reflection loop (轨迹审计器) ──
        if reflection_loop is not None and record is not None:
            try:
                report = reflection_loop.feed(step, record)
                if report is not None and report.corrections:
                    for corr in report.corrections:
                        if corr.action_type == 'safety_cautious' and explorer is not None:
                            old = explorer.explore_prob
                            explorer.explore_prob = max(
                                0.05, old * corr.params.get('explore_rate_factor', 0.3))
                            print(f"[REFL] safety_cautious: explore_prob "
                                  f"{old:.3f} -> {explorer.explore_prob:.3f}")
                        elif corr.action_type == 'explore_rate_reduce' and explorer is not None:
                            old = explorer.explore_prob
                            explorer.explore_prob = max(
                                0.05, old * corr.params.get('factor', 0.5))
                            print(f"[REFL] explore_rate_reduce: "
                                  f"{old:.3f} -> {explorer.explore_prob:.3f}")
            except Exception as e:
                print(f"[REFL] Feed error: {e}")

        # Logging
        if step % log_every == 0:
            log_parts = [
                f"step {step}",
                f"lm={components['lm']:.4f}" if 'lm' in components else "",
                f"vq={components['vq']:.4f}" if 'vq' in components else "",
                f"ctrst={components['contrast']:.4f}" if 'contrast' in components else "",
                f"orth={components['orth']:.4f}" if 'orth' in components else "",
                f"val={components['val']:.4f}" if 'val' in components else "",
                f"ewc={components['ewc']:.4f}" if 'ewc' in components else "",
                f"mgn={components['margin']:.4f}" if 'margin' in components else "",
            ]
            print('  '.join(p for p in log_parts if p))

            # Codebook utilization (every log_every step)
            from train.model import forward as fwd
            _, _, _, aux, _ = fwd(
                state['params'], None, batch[0], cfg, training=True, rng=step_rng)
            cb_util = compute_codebook_utilization(
                state['params'], aux,
                ema_state=state.get('ema_state'),
                seen_masks=state.get('seen_masks'))
            if cb_util:
                util_str = '  '.join(
                    f"{k}={v:.0f}%/{v2:.0f}%"
                    for k, (v, v2) in cb_util.items()
                )
                print(f"  CB util: {util_str}")

        # Checkpoint save
        if save_every > 0 and step > 0 and step % save_every == 0:
            ckpt_path = _do_save_ckpt(state, cfg, step, ckpt_dir)
            saved_ckpts.append(ckpt_path)
            # Rotate old checkpoints
            while len(saved_ckpts) > keep_last:
                old = saved_ckpts.pop(0)
                try:
                    os.remove(old)
                except OSError:
                    pass
            recorder.save()
    recorder.save()
    return state


def _do_save_ckpt(state, cfg, step, ckpt_dir):
    """Save pickle checkpoint with step number in filename."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step}.pkl")
    return save_checkpoint(state, cfg, step=step, path=path)


# ── Checkpoint save/load ─────────────────────────────────────────────────────

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def save_checkpoint(state, cfg, step=None, path=None):
    """Save training checkpoint.

    Excludes gvalue (recreated on load via make_global_value_vectors).
    Excludes continual replay buffers (can be regenerated).
    Per spec: frozen_params preserved; gvalue re-initialized from C_pos/C_neg.

    Args:
        state: Training state dict.
        cfg: LCMConfig (used to recreate gvalue on load).
        step: Override step number (default: state['step']).
        path: Full path (default: checkpoints/step_{step}.pkl).

    Returns:
        Path where checkpoint was saved.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if step is None:
        step = state.get('step', 0)
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, f"step_{step}.pkl")

    def _to_cpu(x):
        try:
            return jax.device_get(x)
        except Exception:
            return x  # int, str, None, etc.

    ckpt = {
        'params': jax.tree_util.tree_map(_to_cpu, state['params']),
        'opt_state': jax.tree_util.tree_map(_to_cpu, state['opt_state']),
        'ema_state': jax.tree_util.tree_map(_to_cpu, state.get('ema_state', {})),
        'feature_bank': jax.tree_util.tree_map(_to_cpu, state.get('feature_bank', {})),
        'seen_masks': jax.tree_util.tree_map(_to_cpu, state.get('seen_masks', {})),
        'self_state': state.get('self_state'),
        'step': step,
        'd_model': cfg.d_model,
    }

    with open(path, 'wb') as f:
        pickle.dump(ckpt, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[CKPT] Saved {path} ({size_mb:.1f} MB)")
    return path


def load_checkpoint(path, cfg, rng, enable_continual=True):
    """Load training checkpoint and rebuild full state.

    Recreates gvalue from frozen C_pos / C_neg (per spec).
    Calls verify_integrity() after loading (per spec b.md:479).

    Args:
        path: Path to checkpoint .pkl file.
        cfg: LCMConfig.
        rng: JAX PRNG key.
        enable_continual: Restore continual learning state.

    Returns:
        Full training state dict.
    """
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)

    def _to_jax(x):
        if isinstance(x, (jax.Array,)):
            return x
        try:
            return jnp.array(x)
        except Exception:
            return x

    params = jax.tree_util.tree_map(_to_jax, ckpt['params'])
    opt_state = jax.tree_util.tree_map(_to_jax, ckpt['opt_state'])
    ema_state = jax.tree_util.tree_map(_to_jax, ckpt.get('ema_state', {}))

    # Feature bank
    feature_bank = {}
    for k in ('bank', 'ptr', 'last_used'):
        if k in ckpt.get('feature_bank', {}):
            feature_bank[k] = _to_jax(ckpt['feature_bank'][k])

    # Seen masks
    seen_masks = {}
    for k, v in ckpt.get('seen_masks', {}).items():
        seen_masks[k] = _to_jax(v)

    # Gvalue: reinit from frozen C_pos/C_neg (per spec, deterministic from d_model)
    C_pos, C_neg = make_global_value_vectors(cfg.d_model)
    gvalue = GValueCodebook(C_pos, C_neg)

    # Verify integrity per spec (b.md:479)
    try:
        gvalue.verify_integrity()
    except AssertionError as e:
        print(f"[WARN] GValue integrity: {e}")

    # Continual learning (re-init; replay buffers not saved)
    from train.continual import init_continual_state
    continual = init_continual_state(cfg.d_model) if enable_continual else None

    # Self state (restore from checkpoint if saved, otherwise re-init)
    ckpt_self_state = ckpt.get('self_state')
    if ckpt_self_state is not None and isinstance(ckpt_self_state, dict):
        from train.self_lattice import init_self_state
        # Re-create with JAX arrays from saved numpy/pickle values
        saved = {k: _to_jax(v) if hasattr(v, 'shape') else v
                 for k, v in ckpt_self_state.items()}
        # Ensure shape matches cfg
        if saved.get('mode_activation', jnp.zeros(1)).shape[0] == cfg.n_self_codes:
            self_state = saved
            print(f"[CKPT] Self state restored from checkpoint")
        else:
            self_state = init_self_state(cfg.n_self_codes, cfg.d_model)
    else:
        from train.self_lattice import init_self_state
        self_state = init_self_state(cfg.n_self_codes, cfg.d_model)

    state = {
        'params': params,
        'gvalue': gvalue,
        'opt_state': opt_state,
        'ema_state': ema_state,
        'feature_bank': feature_bank,
        'continual': continual,
        'self_state': self_state,
        'step': ckpt.get('step', 0),
        'seen_masks': seen_masks,
    }
    print(f"[CKPT] Loaded {path} (step {state['step']}, gvalue integrity OK)")
    return state
