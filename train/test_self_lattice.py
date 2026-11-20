"""Self lattice R15: codebook nearest-neighbour + cosine familiarity.

``C_self[0]`` is the frozen identity anchor. ``C_self[1:]`` are EMA modes.
The forward returns the nearest codebook row (STE) and a familiarity scalar in
``[0, alpha_self]``; fusion adds ``self_bias * z_self`` *before* LayerNorm.

Run from repo root:
    JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
        train/test_self_lattice.py -v
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from train.ema_state import ema_update_self, init_ema_state
from train.fusion import fuse_lattices_with_aux, init_fusion_params, layer_norm
from train.self_lattice import (
    init_self_params, init_self_state, self_lattice_forward,
    self_lattice_reg_loss, update_self_state,
)

D = 8
M = 5
ALPHA = 0.05


def _codebook():
    """Rows: e0, e1, e2, -e0, e3. Directions are easy to reason about."""
    C = jnp.zeros((M, D))
    C = C.at[0].set(jnp.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
    C = C.at[1].set(jnp.array([0, 1.0, 0, 0, 0, 0, 0, 0]))
    C = C.at[2].set(jnp.array([0, 0, 1.0, 0, 0, 0, 0, 0]))
    C = C.at[3].set(jnp.array([-1.0, 0, 0, 0, 0, 0, 0, 0]))
    C = C.at[4].set(jnp.array([0, 0, 0, 1.0, 0, 0, 0, 0]))
    return C


def _params():
    return {'C_self': _codebook(), 'bias_self': jnp.array(ALPHA)}


def _tiny_cfg(**overrides):
    from train.config import LCMConfig
    base = dict(
        d_model=D, d_ff=16, n_heads=2, n_encoder_layers=1,
        vocab_size=32, max_seq_len=16,
        M_top=4, M_fine=4, n_hrq_layers=1,
        M_sparse=4, M_lr=4, n_lr_layers=1, r_max=2, ranks=(2,),
        M_man=4, t_dim=2,
        M_bind=4, n_bind_layers=1,
        M_contrast=4, n_contrast_layers=1,
        n_self_codes=M, n_lattices=6,
        max_inference_steps=2, n_orth_samples=4,
    )
    base.update(overrides)
    return dataclasses.replace(LCMConfig(), **base)


# ─── Shapes / initialisation ────────────────────────────────────────────────


def test_init_shapes_and_bias():
    p = init_self_params(jax.random.PRNGKey(0), D, M_self=M)
    assert p['C_self'].shape == (M, D)
    assert p['bias_self'].shape == ()
    # Row 0 is the frozen core; the remaining 4 rows are modes.
    assert 'core' not in p and 'modes' not in p


def test_init_modes_from_encoder_samples():
    """When early bottleneck samples are supplied, they seed the EMA modes."""
    samples = jnp.arange(3 * D, dtype=jnp.float32).reshape(3, D) * 0.01
    p = init_self_params(jax.random.PRNGKey(0), D, M_self=M,
                         init_samples=samples)
    np.testing.assert_allclose(np.asarray(p['C_self'][1:4]),
                               np.asarray(samples), atol=1e-7)
    assert np.all(np.isfinite(np.asarray(p['C_self'][4])))


def test_init_cog_params_seeds_modes_from_encoder():
    """R15 init path uses the freshly initialised encoder's bottlenecks."""
    from train.cog_train import init_cog_params

    cfg = _tiny_cfg()
    params, _ = init_cog_params(cfg, jax.random.PRNGKey(0))
    modes = params['self']['C_self'][1:]
    norms = np.asarray(jnp.linalg.norm(modes, axis=-1))
    # safe_unit normalises the encoder outputs; random fallback rows would not
    # all be unit-length.
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


# ─── Forward: nearest row + familiarity scalar ──────────────────────────────


def test_z_self_is_a_codebook_row_not_a_mode_blend():
    p = _params()
    z = jnp.array([
        [0.20, 0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.00, 0.00, 0.00, 0.30, 0.10, 0.0, 0.0, 0.0],
    ])
    z_self, _ = self_lattice_forward(p, z, training=False)

    dist2 = jnp.sum((z[:, None, :] - p['C_self'][None, :, :]) ** 2, axis=-1)
    idx = jnp.argmin(dist2, axis=-1)
    np.testing.assert_allclose(np.asarray(z_self),
                               np.asarray(p['C_self'][idx]), atol=1e-7)
    # Exactly one codebook row per sample: not the softmax/activation blend
    # the old mode machine returned.
    blend = jnp.sum(jax.nn.softmax(-dist2, axis=-1)[..., None]
                    * p['C_self'][None, :, :], axis=1)
    assert not np.allclose(np.asarray(z_self[0]), np.asarray(blend[0]))


def test_selected_codebook_row_is_stop_gradient():
    """Ruling 37: EMA owns the codebook; the forward must not train it."""
    p = _params()
    z = jnp.array([[0.20, 0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def loss(C):
        return jnp.sum(self_lattice_forward(
            {'C_self': C, 'bias_self': p['bias_self']}, z)[0])

    grad_C = jax.grad(loss)(p['C_self'])
    np.testing.assert_allclose(np.asarray(grad_C), 0.0, atol=1e-7)


def test_self_bias_varies_with_z_and_stays_in_range():
    p = _params()
    alpha = float(p['bias_self'])
    familiar = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    unfamiliar = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

    _, b_familiar = self_lattice_forward(p, familiar, training=False)
    _, b_unfamiliar = self_lattice_forward(p, unfamiliar, training=False)
    assert b_familiar.shape == (1,)
    assert float(b_familiar[0]) > float(b_unfamiliar[0])
    assert np.all(np.asarray(b_unfamiliar) >= 0.0)
    assert np.all(np.asarray(b_familiar) <= alpha + 1e-6)
    assert np.all(np.asarray(b_unfamiliar) <= alpha + 1e-6)


def test_negative_cosine_clips_self_bias_to_zero():
    """A state opposite to its nearest prototype is not "familiar"."""
    # Row 0 is e0; row 1 is far away. z=-e0 is nearest to row 0 with cosine -1.
    C = jnp.zeros((2, D))
    C = C.at[0, 0].set(1.0)
    C = C.at[1, 1].set(100.0)
    p = {'C_self': C, 'bias_self': jnp.array(ALPHA)}
    z = jnp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    _, bias, aux = self_lattice_forward(
        p, z, training=False, return_aux=True)
    assert int(aux['idx'][0]) == 0
    assert float(aux['cosine'][0]) < 0.0
    assert float(bias[0]) == 0.0


def test_self_bias_never_exceeds_configured_alpha():
    cfg = _tiny_cfg()
    p = init_self_params(jax.random.PRNGKey(0), D, M_self=M,
                         alpha_self=cfg.alpha_self)
    np.testing.assert_allclose(float(p['bias_self']), cfg.alpha_self,
                               atol=1e-7)
    z = jnp.stack([_codebook()[0], _codebook()[1], _codebook()[2]])
    _, bias = self_lattice_forward(p, z, training=False)
    assert np.all(np.asarray(bias) >= 0.0)
    assert np.all(np.asarray(bias) <= cfg.alpha_self + 1e-6)


# ─── EMA: core frozen, modes move ───────────────────────────────────────────


def test_ema_freezes_core_row_and_updates_modes():
    C = _codebook() * 2.0
    state = {'N': jnp.zeros(M), 'm': jnp.zeros((M, D)),
             'last_used': jnp.array([7, 0, 0, 0, 0], dtype=jnp.int32)}
    z = jnp.array([[5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    idx = jnp.array([0, 2])  # core is explicitly selected by the first row

    C_new, state_new = ema_update_self(C, state, z, idx, gamma=0.99, step=3)

    np.testing.assert_array_equal(np.asarray(C_new[0]), np.asarray(C[0]))
    assert not np.array_equal(np.asarray(C_new[2]), np.asarray(C[2]))
    # The frozen row does not accumulate EMA statistics and its usage clock
    # does not move either.
    assert float(state_new['N'][0]) == 0.0
    assert float(jnp.linalg.norm(state_new['m'][0])) == 0.0
    assert int(state_new['last_used'][0]) == 7
    assert float(state_new['N'][2]) > 0.0
    assert int(state_new['last_used'][2]) == 3


def test_core_stays_bit_identical_through_a_training_step():
    """The optimizer/EMA must not write C_self[0]."""
    from train.cog_train import init_cog_params, make_train_step
    from train.param_groups import partition_params

    cfg = _tiny_cfg()
    params, self_state = init_cog_params(cfg, jax.random.PRNGKey(0))
    # Put the core far from the encoder's unit-sphere outputs so every sample
    # selects a mode row: the test is about the core staying frozen while modes
    # demonstrably move.
    C0 = params['self']['C_self']
    C0 = C0.at[0].set(jnp.full((D,), 100.0))
    params['self'] = dict(params['self'], C_self=C0)
    core_before = np.asarray(params['self']['C_self'][0])
    modes_before = np.asarray(params['self']['C_self'][1:])

    ema = init_ema_state(params, cfg)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0, weight_decay=0.5))
    bias_before = np.asarray(params['self']['bias_self'])
    trainable, _, _ = partition_params(params)
    opt_state = optimizer.init(trainable)
    train_step = make_train_step(cfg, optimizer, joint=False)

    tokens = jnp.arange(16, dtype=jnp.int32).reshape(4, 4) % cfg.vocab_size
    batch = (tokens, jnp.roll(tokens, -1, axis=1))
    params2, _, ema2, loss, _ = train_step(
        params, opt_state, ema, batch, 1e-3, jax.random.PRNGKey(7),
        self_state=self_state, step=0)

    assert np.isfinite(float(loss))
    np.testing.assert_array_equal(
        np.asarray(params2['self']['C_self'][0]), core_before)
    # Ruling 34: the alpha mirror is frozen, not a learned scalar.
    np.testing.assert_array_equal(
        np.asarray(params2['self']['bias_self']), bias_before)
    assert not np.array_equal(
        np.asarray(params2['self']['C_self'][1:]), modes_before)
    assert float(np.sum(np.asarray(ema2['self']['N'][1:]))) > 0.0


def test_stage1_weight_decay_does_not_move_core():
    """Ruling 36: Stage-1's AdamW must not touch EMA-owned C_self.

    Exercises the exact Stage-1 mechanism: AdamW with wd=0.5 moves every leaf,
    ``restore_frozen_params(get_frozen_param_names())`` puts back C_self and
    the alpha mirror, and ``train._jitted_ema`` then updates only the modes.
    """
    from train.model import get_frozen_param_names, restore_frozen_params
    from train.train import _jitted_ema

    C = _codebook()
    C = C.at[0].set(jnp.full((D,), 100.0))
    params = {
        'self': {'C_self': C, 'bias_self': jnp.array(ALPHA)},
        'sparse': {'C': jnp.ones((4, D)) * 0.5},
        'manifold': {'C': jnp.ones((4, D)) * 0.25},
    }
    ema = {
        'self': {'N': jnp.zeros(M), 'm': jnp.zeros((M, D))},
        'sparse': {'N': jnp.zeros(4), 'm': jnp.zeros((4, D))},
        'manifold': {'N': jnp.zeros(4), 'm': jnp.zeros((4, D))},
    }
    core_before = np.asarray(C[0])
    modes_before = np.asarray(C[1:])
    bias_before = np.asarray(params['self']['bias_self'])

    opt = optax.adamw(learning_rate=1.0, weight_decay=0.5)
    opt_state = opt.init(params)
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    updates, _ = opt.update(grads, opt_state, params)
    decayed = optax.apply_updates(params, updates)
    assert not np.array_equal(
        np.asarray(decayed['self']['C_self']), np.asarray(C)), \
        "fixture no longer exercises nonzero weight decay"

    restored = restore_frozen_params(
        decayed, params, tuple(get_frozen_param_names()))
    z = jnp.stack([_codebook()[1], _codebook()[2]])
    updated, ema_new = _jitted_ema(
        restored, ema, z, {'self_idx': jnp.array([1, 2])},
        {'gamma_sparse': 0.99, 'gamma_man': 0.99, 'gamma_bind': 0.99,
         'gamma_self': 0.99, 'lambda_shrink': 1e-4})

    np.testing.assert_array_equal(
        np.asarray(updated['self']['C_self'][0]), core_before)
    np.testing.assert_array_equal(
        np.asarray(updated['self']['bias_self']), bias_before)
    assert not np.array_equal(
        np.asarray(updated['self']['C_self'][1:]), modes_before)
    # Unselected modes keep their vectors; only assigned rows move.
    np.testing.assert_array_equal(
        np.asarray(updated['self']['C_self'][3:]), np.asarray(C[3:]))
    assert float(np.sum(np.asarray(ema_new['self']['N'][1:]))) > 0.0


def test_ema_state_has_a_self_branch():
    cfg = _tiny_cfg()
    params, _ = init_self_params(jax.random.PRNGKey(0), D, M_self=M), None
    state = init_ema_state({'self': params}, cfg)
    assert state['self']['N'].shape == (M,)
    assert state['self']['m'].shape == (M, D)
    assert state['self']['last_used'].shape == (M,)


# ─── Fusion: self_bias * z_self before LayerNorm ────────────────────────────


def test_fusion_adds_per_sample_self_bias_before_layer_norm():
    B = 2
    outs = [jnp.ones((B, D)) * (i + 1) for i in range(6)]
    mask = jnp.ones((B, 6)) / 6
    params = init_fusion_params(jax.random.PRNGKey(0), 6, D)
    self_out = jnp.array([
        [0.5, -0.5, 0.25, -0.25, 0.1, 0.2, -0.1, 0.0],
        [-0.3, 0.4, 0.0, 0.2, -0.2, 0.1, 0.3, -0.4],
    ])
    self_bias = jnp.array([0.04, 0.005])  # per-sample familiarity

    z_q, _, _ = fuse_lattices_with_aux(
        outs + [self_out], mask, params, self_bias_weight=self_bias)

    stacked = jnp.stack(outs, axis=-1)
    weights = mask * params['alpha'][None, :]
    weights = weights / weights.sum(axis=-1, keepdims=True)
    pre = (jnp.einsum('bdn,bn->bd', stacked, weights)
           + self_bias[:, None] * self_out)
    want = layer_norm(pre, params['ln_scale'], params['ln_bias'])
    np.testing.assert_allclose(np.asarray(z_q), np.asarray(want), atol=1e-6)

    # Post-LayerNorm addition is observably different: R4/R15 put the self
    # term inside the normalised sum.
    post = layer_norm(jnp.einsum('bdn,bn->bd', stacked, weights),
                      params['ln_scale'], params['ln_bias'])
    assert not np.allclose(np.asarray(z_q),
                           np.asarray(post + self_bias[:, None] * self_out),
                           atol=1e-6)


# ─── Diagnostics ────────────────────────────────────────────────────────────


def test_state_is_a_thin_assignment_diagnostic():
    state = init_self_state(M, D)
    z = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    _, _, aux = self_lattice_forward(_params(), z, training=False,
                                     return_aux=True)
    new_state = update_self_state(state, aux['idx'], z)
    assert new_state['mode_activation'].shape == (M,)
    assert float(jnp.sum(new_state['mode_activation'])) == pytest.approx(1.0)
    assert int(new_state['step']) == 1
    # Diagnostics do not feed back into the output selector.
    z_self, bias = self_lattice_forward(_params(), z, training=False)
    np.testing.assert_allclose(np.asarray(z_self[0]),
                               np.asarray(_codebook()[0]))
    assert float(bias[0]) > 0.0


def test_reg_loss_is_finite_and_has_no_core_term():
    p = _params()
    assert np.isfinite(float(self_lattice_reg_loss(p)))
    # The core is a frozen anchor, not a mode: shifting it must not change the
    # diversity regulariser.
    p2 = {'C_self': p['C_self'].at[0].set(jnp.full((D,), 9.0)),
          'bias_self': p['bias_self']}
    np.testing.assert_allclose(float(self_lattice_reg_loss(p)),
                               float(self_lattice_reg_loss(p2)), atol=1e-7)


# ─── Checkpoint ─────────────────────────────────────────────────────────────


def test_self_bin_roundtrip(tmp_path):
    from train.checkpoint import _load_self, _save_self

    cfg = _tiny_cfg()
    p = init_self_params(jax.random.PRNGKey(3), D, M_self=M,
                         alpha_self=0.07)
    path = str(tmp_path / "self_lattice.bin")
    flat = _save_self({'self': p}, path)
    loaded = _load_self(path, cfg)

    assert flat.size == D + (M - 1) * D + 2
    np.testing.assert_allclose(np.asarray(loaded['C_self']),
                               np.asarray(p['C_self']), atol=1e-7)
    np.testing.assert_allclose(float(loaded['bias_self']),
                               float(p['bias_self']), atol=1e-7)


def test_self_bin_without_r15_tag_is_rejected(tmp_path):
    """An old core+modes file is not migrated (Ruling 11)."""
    from train.checkpoint import _load_self

    cfg = _tiny_cfg()
    old = np.concatenate([
        np.zeros(D, dtype=np.float32),
        np.zeros((M, D), dtype=np.float32).ravel(),
        np.array([1.0, 0.1], dtype=np.float32),  # tau_self, bias_self
    ])
    path = str(tmp_path / "old_self_lattice.bin")
    old.tofile(path)
    with pytest.raises(ValueError, match="R15|C_self|not migrated"):
        _load_self(path, cfg)


def test_self_params_and_ema_state_roundtrip_in_pickle(tmp_path):
    from train.checkpoint import load_cog_state, save_cog_state

    p = init_self_params(jax.random.PRNGKey(4), D, M_self=M)
    ema = init_ema_state({'self': p}, _tiny_cfg())
    save_cog_state(str(tmp_path), params={'self': p}, step=2,
                   ema_state={'self': ema['self']})
    ckpt = load_cog_state(str(tmp_path))

    np.testing.assert_allclose(np.asarray(ckpt['params']['self']['C_self']),
                               np.asarray(p['C_self']), atol=1e-7)
    np.testing.assert_allclose(
        float(ckpt['params']['self']['bias_self']),
        float(p['bias_self']), atol=1e-7)
    assert ckpt['ema_state']['self']['N'].shape == (M,)
    assert ckpt['ema_state']['self']['m'].shape == (M, D)


def test_checkpoint_without_C_self_is_rejected_or_explicitly_reset(tmp_path):
    from train.checkpoint import save_cog_state
    from train.cog_train import load_stage2_params

    cfg = _tiny_cfg()
    old_params = {
        'self': {
            'core': jnp.zeros(D),
            'modes': jnp.zeros((M, D)),
            'tau_self': jnp.array(1.0),
            'bias_self': jnp.array(0.1),
        }
    }
    save_cog_state(str(tmp_path), params=old_params, step=0)

    with pytest.raises(ValueError, match="C_self|R15|self lattice"):
        load_stage2_params(str(tmp_path), cfg, jax.random.PRNGKey(0))

    params, _, _ = load_stage2_params(
        str(tmp_path), cfg, jax.random.PRNGKey(0), allow_missing_self=True)
    assert params['self']['C_self'].shape == (M, D)
    assert params['self']['bias_self'].shape == ()


def test_checkpoint_without_self_ema_branch_is_rejected_or_reset():
    """The EMA buffer schema is checked separately from the params schema."""
    from train.cog_train import _ensure_r15_self_ema

    cfg = _tiny_cfg()
    p = init_self_params(jax.random.PRNGKey(0), D, M_self=M)
    old_ema = {'sparse': {'N': jnp.zeros(4), 'm': jnp.zeros((4, D)),
                          'last_used': jnp.zeros(4, dtype=jnp.int32)}}
    with pytest.raises(ValueError, match="self branch|R15"):
        _ensure_r15_self_ema(old_ema, {'self': p}, cfg, False)

    reset = _ensure_r15_self_ema(old_ema, {'self': p}, cfg, True)
    assert reset['self']['N'].shape == (M,)
    assert reset['self']['m'].shape == (M, D)
    assert 'sparse' in reset, "existing EMA branches must be preserved"
