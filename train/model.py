"""LCM — Full model assembly.

Connects encoder, all six specialized lattices, routing gate, fusion,
and generation head into a single forward pass.
"""
import jax
import jax.numpy as jnp
from jax import lax
import optax

from train.config import LCMConfig
from train.encoder import init_encoder_params, encoder_forward
from train.hyp import safe_unit
from train.lattices import (
    init_route_params, routing_gate, route_commit_loss,
    init_hrq_params, hrq_forward,
    init_sparse_params, sparse_forward,
    init_lowrank_params, lowrank_forward,
    init_manifold_params, manifold_forward,
    init_binding_params, binding_forward,
    init_contrast_params, contrast_forward, contrast_info_nce_loss,
    init_value_scalars, init_danger_params,
)
from train.gvalue import GValueCodebook, make_global_value_vectors
from train.cognitive_step import six_lattice_step
from train.fusion import init_fusion_params, init_gen_head_params, gen_head_forward
from train.self_lattice import (
    init_self_params, init_self_state, self_lattice_forward,
    reset_session_state, SelfState,
)


def init_all_params(cfg: LCMConfig, rng):
    """Initialize all model parameters."""
    keys = jax.random.split(rng, 16)
    d = cfg.d_model

    params = {}

    # Encoder
    params['encoder'] = init_encoder_params(
        keys[0], d, cfg.d_ff, cfg.n_heads, cfg.n_encoder_layers,
        cfg.vocab_size, cfg.max_seq_len)

    # Routing gate
    params['route'] = init_route_params(keys[1], cfg.n_lattices, d)

    # Lattices
    params['hrq'] = init_hrq_params(keys[2], d, cfg.M_top, cfg.M_fine, cfg.n_hrq_layers)
    params['sparse'] = init_sparse_params(keys[3], d, cfg.M_sparse)
    params['lowrank'] = init_lowrank_params(keys[4], d, cfg.M_lr, cfg.ranks)
    params['manifold'] = init_manifold_params(keys[5], d, cfg.M_man, cfg.t_dim)
    params['binding'] = init_binding_params(keys[6], d, cfg.M_bind, cfg.n_bind_layers, cfg.r_max)
    params['contrast'] = init_contrast_params(keys[7], d, cfg.M_contrast, cfg.n_contrast_layers)

    # Fusion
    params['fusion'] = init_fusion_params(keys[8], cfg.n_lattices, d)
    params['gen_head'] = init_gen_head_params(keys[9], d, cfg.vocab_size)

    # Danger codebook (frozen, saved with SHA-256, never trained)
    params['danger'] = init_danger_params(keys[12], cfg.M_danger, d)

    # Local value scalars
    lattice_sizes = [
        ('hrq', cfg.M_top),
        ('sparse', cfg.M_sparse),
        ('lowrank', cfg.M_lr),
        ('manifold', cfg.M_man),
        ('binding', cfg.M_bind),
        ('contrast', cfg.M_contrast),
    ]
    params['value_scalars'] = init_value_scalars(keys[10], lattice_sizes)

    # Shared low-rank base V (stored in lowrank params, used by binding)
    # Already in params['lowrank']

    # Self lattice params (initialized but managed separately)
    params['self'] = init_self_params(keys[11], d, cfg.n_self_codes)
    self_state = init_self_state(cfg.n_self_codes, d)

    # Global value lattice — not in params (excluded from optimizer)
    C_pos, C_neg = make_global_value_vectors(d)
    gvalue = GValueCodebook(C_pos, C_neg)

    return params, gvalue, self_state


def forward(params, gvalue, x, cfg: LCMConfig, training=True, rng=None,
            self_state=None, routing_bias=None):
    """Full model forward pass.

    Args:
        params: All model parameters.
        gvalue: Global value codebook (frozen).
        x: Input tokens (B, N).
        cfg: Configuration.
        training: Whether in training mode.
        rng: JAX PRNG key (for Gumbel-Softmax).
        self_state: Optional SelfState for self lattice.
        routing_bias: Optional (6,) bias added to routing logits before softmax.
            Used by BehaviorExplorer for active bias exploration (see e.md §六).

    Returns:
        z: Bottleneck vector (B, d).
        z_q: Quantized memory output (B, d).
        logits: Output logits (B, N, V).
        aux: Auxiliary outputs for loss computation.
    """
    B, N = x.shape
    d = cfg.d_model

    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Encoder
    z = encoder_forward(params['encoder'], x, cfg.n_heads)  # (B, d)
    # Normalize to unit sphere so encoder magnitude doesn't explode through
    # lattice forwards and commitment losses. Without this, N=512 with random
    # init can produce z-norms that overflow downstream softmax/log ops.
    z = safe_unit(z)

    # Routing gate with optional bias injection
    route_params = params['route']
    if routing_bias is not None:
        route_params = dict(route_params, bias=routing_bias)
    soft_mask, z_route, route_idx = routing_gate(
        route_params, z, cfg.tau_route,
        hard=not training, rng=rng)

    # Self lattice (internal state machine). Runs first so its output can be
    # fused as the 7th element. Its output is INDEPENDENT of z — self exists
    # regardless of external input; z only enters the world-self divergence
    # (diagnostic).
    self_state_out = None
    world_dev = jnp.array(0.0)
    self_output = None
    if self_state is not None and 'self' in params:
        self_output, self_state_out, world_dev = self_lattice_forward(
            params['self'], self_state, z=z, rng=rng, training=training)

    # Routing + the six lattice forwards + fusion, from the single canonical
    # implementation in train/cognitive_step.py — the same one the cognitive
    # loop runs. Everything outside this call belongs to this function: the
    # encoder above, self above, the generation head below.
    z_q, lat = six_lattice_step(
        z, params, cfg, training=training, rng=rng, gvalue=gvalue,
        self_output=self_output,
        self_bias_weight=cfg.alpha_self if self_output is not None else None)
    lattice_outputs = lat['lattice_outputs']

    # Safety check on fused output (log only, no interrupt during training)
    if gvalue is not None:
        is_safe, margins, violated_law = gvalue.check_safety_batch(
            z_q, cfg.safety_margin_relative)
        min_margin = margins.min()
    else:
        min_margin = jnp.array(1.0)

    # Generation head with causal linear attention + GLU (teacher-forced)
    logits = gen_head_forward(params['gen_head'], z_q, x, training=training)

    aux = {
        'z_route': lat['z_route'],
        'route_idx': lat['route_idx'],
        'soft_mask': lat['soft_mask'],
        'lattice_outputs': lattice_outputs,
        'man_idx': lat['man_idx'],
        'hrq_idx': lat['hrq_idx'],
        'hrq_top_sim': lat['hrq_top_sim'],
        'sparse_idx': lat['sparse_idx'],
        # Per-layer query vectors of the binding residual chains — the space
        # each binding codebook actually quantises, and therefore the space its
        # EMA must accumulate.
        'binding_residuals': lat['binding_residuals'],
        'value_signals': None if gvalue is None else
            gvalue.compute_value_signal_batch(lattice_outputs, cfg.tau_val_signal),
        'safety_margin': min_margin,
        'self_state': self_state_out,
        'world_dev': world_dev,
    }

    return z, z_q, logits, aux, self_state_out


def get_frozen_param_names():
    """Return names of parameters excluded from *all* updates.

    These are the permanently-frozen safety layers. They are never trained and
    never touched by the optimizer, so neither their gradient nor AdamW's
    decoupled weight decay may modify them.

    Deliberately NOT listed here (see README §"Gradient and EMA Hybrid
    Management"): ``sparse/C``, ``manifold/C`` and ``binding/*_cb``. Those are
    *EMA + gradient* codebooks — they receive gradient like any other
    parameter **and** are additionally updated by the per-code EMA pass. An
    earlier revision of this list marked them frozen, which contradicted the
    README and the EMA design; it was never applied anywhere, so nothing
    depended on it. ``gvalue`` is absent from ``params`` altogether (it lives
    on ``TrainingState`` as a ``GValueCodebook``) and needs no entry.
    """
    return [
        'danger/C',      # Danger lattice — frozen, never trained
    ]


def restore_frozen_params(params_new, params_old, frozen_prefixes):
    """Undo any change to frozen leaves after an optimizer update.

    AdamW applies decoupled weight decay as ``-lr·wd·θ`` regardless of the
    gradient, so a parameter that is never trained still shrinks every step
    unless something restores it. The optimizer tree keeps every leaf (the
    optimizer state shape is unchanged), so existing checkpoints stay loadable.

    Note ``_flatten_dict`` treats a Python list as a single leaf, so a pattern
    cannot address *inside* a list (``hrq/fine/0/W`` will not match). Freeze
    whole subtrees, or extend the flattener first.
    """
    if not frozen_prefixes:
        return params_new
    flat_old = _flatten_dict(params_old)
    flat_new = _flatten_dict(params_new)
    out = {}
    for path, val in flat_new.items():
        key = '/'.join(path)
        if any(_match_prefix(key, p) for p in frozen_prefixes):
            val = flat_old[path]
        _set_in_dict(out, path, val)
    return out


def _flatten_dict(d, prefix=()):
    items = []
    for k, v in d.items():
        path = prefix + (k,)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, path).items())
        else:
            items.append((path, v))
    return dict(items)


def _set_in_dict(d, path, val):
    for p in path[:-1]:
        d = d.setdefault(p, {})
    d[path[-1]] = val


def _match_prefix(key, pattern):
    parts = pattern.rstrip('*').rstrip('/').split('/')
    key_parts = key.split('/')
    if pattern.endswith('*'):
        return key_parts[:len(parts)] == parts
    return key_parts == parts


def generate(state, prompt_ids, rng, cfg, max_new_tokens=50, bos_id=1, eos_id=2):
    """Simple autoregressive generation for verification.

    Args:
        state: Training state dict.
        prompt_ids: (1, N) prompt token IDs.
        rng: JAX PRNG key.
        cfg: LCMConfig.
        max_new_tokens: Max tokens to generate.
        bos_id: BOS token ID.
        eos_id: EOS token ID.

    Returns:
        output_ids: List of generated token IDs.
    """
    import jax
    import jax.numpy as jnp

    B, N = prompt_ids.shape
    generated = list(prompt_ids[0].tolist())

    for step in range(max_new_tokens):
        # Pad or truncate input to N
        if len(generated) > N:
            input_ids = jnp.array([generated[-N:]], dtype=jnp.int32)
        else:
            input_ids = jnp.array([generated], dtype=jnp.int32)

        # Forward pass
        rng, fwd_rng = jax.random.split(rng)
        z, z_q, logits, aux, self_state_out = forward(
            state['params'], state['gvalue'], input_ids, cfg,
            training=False, rng=fwd_rng,
            self_state=state.get('self_state'))

        # Get last-token logits
        last_logits = logits[0, -1, :]  # (V,)

        # Sample (temperature=1.0)
        rng, sample_rng = jax.random.split(rng)
        next_token = jax.random.categorical(sample_rng, last_logits)
        next_id = int(next_token)

        generated.append(next_id)

        # Stop on EOS
        if next_id == eos_id:
            break

    return generated
