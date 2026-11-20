"""EMA state and the three EMA lattice updates (R3).

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_ema_state.py -v
"""
import dataclasses
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from train.cog_loop import cog_loop_scan
from train.config import LCMConfig
from train.ema_state import (
    ema_update_binding, ema_update_manifold, ema_update_sparse, init_ema_state,
)
from train.feature_bank import FeatureBank
from train.hyp import exp_map
from train.lattices import binding_forward


def _cfg(**overrides):
    base = dict(M_sparse=4, M_man=3, M_bind=5, n_bind_layers=2,
                d_model=2, gamma_sparse=0.99, gamma_man=0.99, gamma_bind=0.99)
    base.update(overrides)
    return type('C', (), base)()


def _params():
    d = 2

    def simvq(M, fill):
        return {'A': jnp.full((M, d), fill), 'W': jnp.eye(d)}

    return {
        'sparse': {'C': jnp.ones((4, d))},
        'manifold': {'C': jnp.ones((3, d))},
        'binding': {
            'key_cb': [simvq(5, 0.1), simvq(5, 0.2)],
            'val_cb': [simvq(5, 0.3), simvq(5, 0.4)],
            'bind_cb': [simvq(5, 0.5), simvq(5, 0.6)],
        },
    }


def test_sparse_ema_updates_only_selected_rows():
    cfg = _cfg()
    C = jnp.zeros((4, 2))
    state = {'N': jnp.zeros(4), 'm': jnp.zeros((4, 2)),
             'last_used': jnp.zeros(4, dtype=jnp.int32)}
    z = jnp.array([[1.0, 2.0]])
    idx = jnp.array([1])
    C2, s2 = ema_update_sparse(C, state, z, idx, gamma=0.99, lambda_shrink=1e-4)
    assert float(jnp.linalg.norm(C2[1])) > 0.0
    assert float(jnp.linalg.norm(C2[0])) == 0.0
    assert float(s2['N'][1]) > 0.0


def test_init_ema_state_shapes_and_pytree():
    state = init_ema_state(_params(), _cfg())
    assert float(state['sparse']['N'][0]) == 0.0
    assert state['sparse']['N'].shape == (4,)
    assert state['sparse']['m'].shape == (4, 2)
    assert state['sparse']['last_used'].shape == (4,)
    assert state['manifold']['N'].shape == (3,)
    assert state['manifold']['m'].shape == (3, 2)
    assert len(state['binding']['key']) == 2
    assert len(state['binding']['val']) == 2
    assert len(state['binding']['bind']) == 2
    assert state['binding']['key'][1]['m'].shape == (5, 2)

    # JAX-compatible: every leaf is an array, so the state can cross jit.
    leaves = jax.tree_util.tree_leaves(state)
    assert leaves and all(hasattr(x, 'shape') for x in leaves)
    # tree_map over it must return arrays with the original structure.
    doubled = jax.tree_util.tree_map(lambda x: x * 2, state)
    assert jax.tree_util.tree_structure(doubled) == \
        jax.tree_util.tree_structure(state)


def test_ema_does_not_touch_unselected_rows():
    """An untouched row keeps its existing vector, not zero."""
    C = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    state = {'N': jnp.zeros(4), 'm': jnp.zeros((4, 2)),
             'last_used': jnp.zeros(4, dtype=jnp.int32)}
    z = jnp.array([[0.0, 5.0]])
    C2, _ = ema_update_sparse(C, state, z, jnp.array([2]), gamma=0.99,
                              lambda_shrink=0.0)
    np.testing.assert_allclose(np.asarray(C2[0]), [1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(C2[1]), [2.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(C2[2]), [0.0, 5.0], atol=1e-7)
    np.testing.assert_allclose(np.asarray(C2[3]), [4.0, 0.0], atol=1e-7)


def test_ema_is_a_running_average_of_observations():
    C = jnp.zeros((2, 2))
    state = {'N': jnp.zeros(2), 'm': jnp.zeros((2, 2)),
             'last_used': jnp.zeros(2, dtype=jnp.int32)}
    C1, s1 = ema_update_sparse(C, state, jnp.array([[1.0, 0.0]]),
                               jnp.array([0]), gamma=0.9, lambda_shrink=0.0)
    C2, s2 = ema_update_sparse(C1, s1, jnp.array([[3.0, 0.0]]),
                               jnp.array([0]), gamma=0.9, lambda_shrink=0.0)
    # N = 1 then 1.9; m = 1 then 0.9*1 + 3 → C = 3.9 / 1.9
    expected = (0.9 * 1.0 + 3.0) / (0.9 * 1.0 + 1.0)
    np.testing.assert_allclose(float(C2[0, 0]), expected, rtol=1e-6)
    np.testing.assert_allclose(float(s2['N'][0]), 1.9, rtol=1e-6)


def test_manifold_ema_stores_tangent_coordinates_not_exp_map():
    """R3: params['manifold']['C'] is tangent space; EMA must not exp_map."""
    C = jnp.zeros((3, 2))
    state = {'N': jnp.zeros(3), 'm': jnp.zeros((3, 2)),
             'last_used': jnp.zeros(3, dtype=jnp.int32)}
    z = jnp.array([[0.5, 0.25]])
    C2, _ = ema_update_manifold(C, state, z, jnp.array([1]), gamma=0.99)
    np.testing.assert_allclose(np.asarray(C2[1]), np.asarray(z[0]), atol=1e-7)
    ball = exp_map(z)
    assert not np.allclose(np.asarray(C2[1]), np.asarray(ball[0]), atol=1e-4), \
        "manifold EMA applied exp_map; parameters must stay in tangent space"


def test_sparse_soft_threshold_shrinks_toward_zero():
    C = jnp.zeros((1, 2))
    state = {'N': jnp.zeros(1), 'm': jnp.zeros((1, 2)),
             'last_used': jnp.zeros(1, dtype=jnp.int32)}
    C2, _ = ema_update_sparse(C, state, jnp.array([[0.01, -0.02]]),
                              jnp.array([0]), gamma=0.99,
                              lambda_shrink=0.005)
    np.testing.assert_allclose(np.asarray(C2[0]), [0.005, -0.015], atol=1e-7)


def test_binding_ema_updates_every_layer_and_family():
    d, M, n_layers = 2, 5, 2
    rows = jnp.arange(M, dtype=jnp.float32)
    # Row j of every codebook is [j, 0]; a residual at [sel, 0] selects row sel.
    base = jnp.stack([rows, jnp.zeros_like(rows)], axis=-1)
    params = {
        fam: [{'A': base * (f_i + 1), 'W': jnp.eye(d)}
              for _ in range(n_layers)]
        for f_i, fam in enumerate(('key_cb', 'val_cb', 'bind_cb'))
    }
    state = {fam: [{'N': jnp.zeros(M), 'm': jnp.zeros((M, d)),
                    'last_used': jnp.zeros(M, dtype=jnp.int32)}
                   for _ in range(n_layers)]
             for fam in ('key', 'val', 'bind')}
    residuals = {}
    selected = {}
    indices = {}
    for f_i, fam in enumerate(('key', 'val', 'bind')):
        residuals[fam] = []
        selected[fam] = []
        indices[fam] = []
        for l in range(n_layers):
            sel = (f_i + l) % M
            residuals[fam].append(jnp.array(
                [[float(sel) * (f_i + 1) + 0.5, 0.25]]))
            selected[fam].append(sel)
            indices[fam].append(jnp.array([sel], dtype=jnp.int32))

    new_params, new_state = ema_update_binding(
        params, state, residuals, indices, gamma=0.99, step=17)

    for f_i, fam in enumerate(('key', 'val', 'bind')):
        for l in range(n_layers):
            entry = new_params[f'{fam}_cb'][l]
            # The effective codebook (A @ W) must contain the residual at the
            # selected row, and the factorization must be usable next step.
            C = entry['A'] @ entry['W']
            sel = selected[fam][l]
            np.testing.assert_allclose(
                np.asarray(C[sel]), np.asarray(residuals[fam][l][0]),
                atol=1e-6)
            assert int(new_state[fam][l]['last_used'][sel]) == 17
            assert float(new_state[fam][l]['N'][sel]) > 0.0
            # Untouched rows keep their old value (cold rows are not zeroed).
            other = (sel + 1) % M
            np.testing.assert_allclose(
                np.asarray(C[other]),
                np.array([float(other) * (f_i + 1), 0.0]), atol=1e-6)


def test_binding_ema_uses_value_biased_forward_indices():
    """EMA must write the row the forward selected, not a Euclidean argmin.

    ``binding_forward`` selects with ``value_biased_scores`` when
    ``value_scalars`` is active. Recomputing a plain argmin in the EMA pass
    would update a row the forward never quantised.
    """
    d, r_max = 2, 2
    # Rows [0,0] and [1,0]; the query [0.1,0] is Euclidean-nearest to row 0.
    base = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    params = {
        'A_k': jnp.eye(r_max), 'A_v': jnp.eye(r_max),
        'key_cb': [{'A': base, 'W': jnp.eye(d)}],
        'val_cb': [{'A': base, 'W': jnp.eye(d)}],
        'bind_cb': [{'A': base, 'W': jnp.eye(d)}],
    }
    V = jnp.eye(d, r_max)
    z = jnp.array([[0.1, 0.0]])
    # Row 1 has value 1; alpha=20 makes the value term dominate the distance.
    v = jnp.array([0.0, 1.0])
    _, residuals, indices, _ = binding_forward(
        params, z, V, value_scalars=v, alpha_val=20.0)

    key_residual = residuals['key'][0]
    C_key = params['key_cb'][0]['A'] @ params['key_cb'][0]['W']
    plain_argmin = jnp.argmin(
        jnp.sum((key_residual[:, None, :] - C_key[None, :, :]) ** 2, axis=-1),
        axis=-1)
    assert int(plain_argmin[0]) == 0, "fixture no longer has a disagreement"
    assert int(indices['key'][0][0]) == 1, \
        "value bias did not flip the key selection"

    state = {fam: [{'N': jnp.zeros(2), 'm': jnp.zeros((2, d)),
                    'last_used': jnp.zeros(2, dtype=jnp.int32)}]
             for fam in ('key', 'val', 'bind')}
    new_params, new_state = ema_update_binding(
        params, state, residuals, indices, gamma=0.99)

    updated_key = new_params['key_cb'][0]['A'] @ new_params['key_cb'][0]['W']
    # The value-biased row received the residual; the Euclidean-nearest row
    # was not touched by the EMA.
    np.testing.assert_allclose(np.asarray(updated_key[1]),
                               np.asarray(key_residual[0]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(updated_key[0]), [0.0, 0.0],
                               atol=1e-6)
    assert float(new_state['key'][0]['N'][1]) > 0.0
    assert float(new_state['key'][0]['N'][0]) == 0.0


def test_decayed_sum_form_matches_bmd_scaled_form():
    """Our N/m are b.md's state divided by (1-gamma); the codebook is the same.

    For a warm row (N >= 1, both clips are the identity):
        N_ours = N_ref/(1-gamma),  m_ours = m_ref/(1-gamma)
    The update is linear, so after an assignment the ratio m/N is identical.
    """
    gamma = 0.9
    scale = 1.0 - gamma
    N_ref0 = jnp.array([2.0])
    m_ref0 = jnp.array([[2.0, 4.0]])
    state = {'N': N_ref0 / scale, 'm': m_ref0 / scale,
             'last_used': jnp.zeros(1, dtype=jnp.int32)}
    z = jnp.array([[3.0, 5.0]])
    idx = jnp.array([0])

    C_our, state_our = ema_update_sparse(
        jnp.zeros((1, 2)), state, z, idx, gamma, lambda_shrink=0.0)

    N_ref = gamma * N_ref0 + scale * 1.0
    m_ref = gamma * m_ref0 + scale * z
    C_ref = m_ref / jnp.clip(N_ref, 1.0)[:, None]
    np.testing.assert_allclose(np.asarray(C_our), np.asarray(C_ref),
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(state_our['N']),
                               np.asarray(N_ref / scale), atol=1e-6)

    # Cold row: b.md's scaled state is (1-gamma)*[count, z_sum]; the ratio is
    # the observation, which our N=count>=1 clip makes exact.
    cold = {'N': jnp.zeros(1), 'm': jnp.zeros((1, 2)),
            'last_used': jnp.zeros(1, dtype=jnp.int32)}
    C_cold, _ = ema_update_sparse(jnp.zeros((1, 2)), cold, z, idx, gamma,
                                  lambda_shrink=0.0)
    np.testing.assert_allclose(np.asarray(C_cold), np.asarray(z), atol=1e-7)


def test_last_used_only_moves_for_selected_rows():
    C = jnp.zeros((4, 2))
    state = {'N': jnp.zeros(4), 'm': jnp.zeros((4, 2)),
             'last_used': jnp.array([3, 7, 9, 11], dtype=jnp.int32)}
    _, s2 = ema_update_sparse(C, state, jnp.array([[1.0, 1.0]]),
                              jnp.array([2]), gamma=0.99, lambda_shrink=0.0,
                              step=42)
    assert int(s2['last_used'][2]) == 42
    assert int(s2['last_used'][0]) == 3
    assert int(s2['last_used'][1]) == 7
    assert int(s2['last_used'][3]) == 11


# ─── Wiring: cognitive loop stats, train_step, checkpoint ───────────────────


def _tiny_cfg(**overrides):
    base = dict(
        d_model=8, d_ff=16, n_heads=2, n_encoder_layers=1,
        vocab_size=32, max_seq_len=16,
        M_top=4, M_fine=4, n_hrq_layers=1,
        M_sparse=4, M_lr=4, n_lr_layers=1, r_max=2, ranks=(2,),
        M_man=4, t_dim=2,
        M_bind=4, n_bind_layers=1,
        M_contrast=4, n_contrast_layers=1,
        n_self_codes=4, n_lattices=6,
        max_inference_steps=2, n_orth_samples=4,
    )
    base.update(overrides)
    return dataclasses.replace(LCMConfig(), **base)


def _tiny_params(cfg):
    from train.cog_train import init_cog_params
    params, self_state = init_cog_params(cfg, jax.random.PRNGKey(0))
    return params, self_state


def test_cog_loop_scan_stats_are_opt_in_and_default_return_is_unchanged():
    cfg = _tiny_cfg()
    params, _ = _tiny_params(cfg)
    z = jnp.ones((1, cfg.d_model)) * 0.1
    rng = jax.random.PRNGKey(3)

    default = cog_loop_scan(z, params, cfg, max_steps=2, rng=rng)
    assert len(default) == 3, "default cog_loop_scan return must stay a 3-tuple"
    z_qs, diffs, entropies = default
    assert z_qs.shape == (2, 1, cfg.d_model)
    assert diffs.shape == (2, 1)
    assert entropies.shape == (2, 1)

    with_stats = cog_loop_scan(z, params, cfg, max_steps=2, rng=rng,
                               return_stats=True)
    assert len(with_stats) == 4
    z_qs2, diffs2, entropies2, stats = with_stats
    np.testing.assert_array_equal(np.asarray(z_qs2), np.asarray(z_qs))
    np.testing.assert_array_equal(np.asarray(diffs2), np.asarray(diffs))
    np.testing.assert_array_equal(np.asarray(entropies2), np.asarray(entropies))

    assert stats['z_in'].shape == (2, 1, cfg.d_model)
    assert stats['sparse_idx'].shape == (2, 1)
    assert stats['man_idx'].shape == (2, 1)
    for family in ('key', 'val', 'bind'):
        assert len(stats['binding_residuals'][family]) == cfg.n_bind_layers
        assert stats['binding_residuals'][family][0].shape == (2, 1, cfg.d_model)
        assert len(stats['binding_indices'][family]) == cfg.n_bind_layers
        assert stats['binding_indices'][family][0].shape == (2, 1)


def test_train_step_separates_ema_from_adamw_and_updates_ema():
    """The EMA codebook must equal a pure EMA update of the pre-step C.

    A large weight decay makes the check sensitive: if the EMA codebook were
    still inside the AdamW tree, decoupled decay would move it and the result
    would no longer match the EMA-only formula.
    """
    from train.cog_train import make_train_step
    from train.param_groups import partition_params

    cfg = _tiny_cfg()
    params, self_state = _tiny_params(cfg)
    params['danger'] = {'C': jnp.ones((4, cfg.d_model))}
    danger_before = np.asarray(params['danger']['C'])
    ema0 = init_ema_state(params, cfg)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0, weight_decay=0.5))
    trainable, _, _ = partition_params(params)
    opt_state = optimizer.init(trainable)
    train_step = make_train_step(cfg, optimizer, joint=False)

    tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % cfg.vocab_size
    batch = (tokens, jnp.roll(tokens, -1, axis=1))
    initial_embed = params['encoder']['embed']
    ema = ema0
    for step_i in range(2):
        ema_before = ema
        C_before = params['sparse']['C']
        params, opt_state, ema, loss, aux = train_step(
            params, opt_state, ema, batch, 1e-2,
            jax.random.PRNGKey(10 + step_i), self_state=self_state,
            step=step_i)

        assert np.isfinite(float(loss))
        # Trainable parameters move.
        assert not np.array_equal(np.asarray(params['encoder']['embed']),
                                  np.asarray(initial_embed))
        # Frozen parameters do not.
        np.testing.assert_array_equal(np.asarray(params['danger']['C']),
                                      danger_before)

        stats = aux['loop_stats']
        C_expected, state_expected = ema_update_sparse(
            C_before, ema_before['sparse'], stats['z_in'],
            stats['sparse_idx'], cfg.gamma_sparse, cfg.lambda_sparse,
            step=step_i)
        # The EMA codebook must equal a pure EMA update of the pre-step C.
        # A large weight decay makes this sensitive: if the EMA codebook were
        # still inside the AdamW tree, decoupled decay would move it.
        np.testing.assert_allclose(np.asarray(params['sparse']['C']),
                                   np.asarray(C_expected), atol=1e-6)
        np.testing.assert_allclose(np.asarray(ema['sparse']['N']),
                                   np.asarray(state_expected['N']), atol=1e-6)
        np.testing.assert_allclose(np.asarray(ema['sparse']['m']),
                                   np.asarray(state_expected['m']), atol=1e-6)
        if aux.get('self_state') is not None:
            self_state = aux['self_state']

    # The state clock is absolute: a resumed offset-0 step carries abs_step=5,
    # so `last_used` must be stamped 5, not 0.
    params, opt_state, ema, _, aux = train_step(
        params, opt_state, ema, batch, 1e-2, jax.random.PRNGKey(99),
        self_state=self_state, step=5)
    selected = np.asarray(aux['loop_stats']['sparse_idx'])
    np.testing.assert_array_equal(
        np.asarray(ema['sparse']['last_used'])[selected], 5)

    # EMA statistics keep accumulating across steps (the pass really ran).
    assert float(jnp.sum(ema['sparse']['N'])) > 0.0
    assert not np.array_equal(np.asarray(ema['sparse']['N']),
                              np.asarray(ema0['sparse']['N']))
    # Binding was updated layer by layer and collapsed to an effective
    # codebook (C, I), so the next forward quantises the EMA vector exactly.
    assert float(jnp.sum(ema['binding']['key'][0]['N'])) > 0.0
    np.testing.assert_allclose(
        np.asarray(params['binding']['key_cb'][0]['W']),
        np.eye(cfg.d_model), atol=1e-6)


def test_host_bank_update_replaces_dead_rows_only():
    """The host path zeroes N/m and updates last_used for dead rows only."""
    from train.cog_train import _host_bank_update

    cfg = _tiny_cfg(bank_dead_threshold=2)
    params, _ = _tiny_params(cfg)
    ema = init_ema_state(params, cfg)
    # Row 0 is dead at step 5; rows 1..3 were used recently.
    last = jnp.full((cfg.M_sparse,), 4, dtype=jnp.int32).at[0].set(2)
    ema['sparse']['last_used'] = last
    ema['sparse']['N'] = ema['sparse']['N'].at[0].set(5.0)
    ema['sparse']['m'] = ema['sparse']['m'].at[0].set(jnp.ones(cfg.d_model))
    C_before = jnp.asarray(params['sparse']['C'])

    # Empty bank: the dead row cannot be replenished, so nothing changes.
    empty = FeatureBank(4, cfg.d_model)
    params_noop, ema_noop = _host_bank_update(params, ema, empty, 5, cfg)
    np.testing.assert_array_equal(np.asarray(params_noop['sparse']['C']),
                                  np.asarray(C_before))
    assert float(ema_noop['sparse']['N'][0]) == 5.0
    assert int(ema_noop['sparse']['last_used'][0]) == 2

    bank = FeatureBank(4, cfg.d_model)
    bank.add(jnp.full((1, cfg.d_model), 7.0))
    params_new, ema_new = _host_bank_update(params, ema, bank, 5, cfg)
    np.testing.assert_allclose(np.asarray(params_new['sparse']['C'][0]),
                               np.full(cfg.d_model, 7.0), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(params_new['sparse']['C'][1:]),
                                  np.asarray(C_before[1:]))
    assert float(ema_new['sparse']['N'][0]) == 0.0
    assert float(jnp.linalg.norm(ema_new['sparse']['m'][0])) == 0.0
    assert int(ema_new['sparse']['last_used'][0]) == 5
    assert int(ema_new['sparse']['last_used'][1]) == 4


def test_supervisor_forwards_ema_state_for_five_output_steps(tmp_path):
    """Supervisor keeps its 4-tuple API but carries the EMA state through aux."""
    from train.train_supervisor import Supervisor

    cfg = _tiny_cfg()
    sup = Supervisor(str(tmp_path / "sup"), cfg, enable_auto=True)
    calls = []

    def fake_step(params, opt_state, ema_state, batch, lr, rng,
                  self_state=None, step=None):
        calls.append((ema_state, step, self_state))
        new_ema = jax.tree_util.tree_map(lambda x: x + 1.0, ema_state)
        return params, opt_state, new_ema, jnp.array(0.5), \
            {'self_state': self_state}

    ema = {'sparse': {'N': jnp.zeros(3)}}
    params, opt_state, loss, aux = sup.step(
        fake_step, {'p': jnp.zeros(1)}, {'o': jnp.zeros(1)}, None,
        1e-3, jax.random.PRNGKey(0), step=3, self_state='ss',
        ema_state=ema)

    assert calls[0][1] == 3 and calls[0][2] == 'ss'
    assert float(loss) == 0.5
    np.testing.assert_allclose(np.asarray(aux['ema_state']['sparse']['N']),
                               np.ones(3))
    assert sup.best_ema_state is not None
    np.testing.assert_allclose(
        np.asarray(sup.best_ema_state['sparse']['N']), np.ones(3))


class _FakeSpec:
    def __init__(self, tokenizer_id, path, sha256, model_vocab_size):
        self.tokenizer_id = tokenizer_id
        self.path = path
        self.sha256 = sha256
        self.model_vocab_size = model_vocab_size


def _fake_identity(tmp_path, vocab_size):
    from train.cog_train import RunIdentity
    from train.tokenizer_spec import sha256_file

    tok = tmp_path / "source_tokenizer.json"
    tok.write_text("{}")
    return RunIdentity(
        _FakeSpec("fake", str(tok), sha256_file(str(tok)), vocab_size),
        {"model_vocab_size": vocab_size,
         "token_data_sha256": "aa", "document_spans_sha256": "bb",
         "n_tokens": 1, "n_docs": 1},
    )


def test_cog_checkpoint_roundtrips_ema_and_feature_bank(tmp_path):
    from train.cog_train import (
        RunIdentity, load_cog_buffers, load_stage2_params,
        save_cog_checkpoint,
    )

    cfg = _tiny_cfg()
    params, self_state = _tiny_params(cfg)
    ema = init_ema_state(params, cfg)
    ema['sparse']['N'] = ema['sparse']['N'].at[0].set(2.5)
    ema['sparse']['m'] = ema['sparse']['m'].at[0].set(jnp.ones(cfg.d_model))
    last = jnp.full((cfg.M_sparse,), 4, dtype=jnp.int32).at[0].set(2)
    ema['sparse']['last_used'] = last

    bank = FeatureBank(8, cfg.d_model)
    bank.add(np.arange(cfg.d_model, dtype=np.float32))
    bank.add(np.ones(cfg.d_model, dtype=np.float32) * 3)

    out = str(tmp_path / "ckpt")
    save_cog_checkpoint(
        params, out, 5, run_identity=_fake_identity(tmp_path, cfg.vocab_size),
        train_cfg=cfg, self_state=self_state, ema_state=ema,
        feature_bank=bank)

    with open(f"{out}/cog_params.pkl", "rb") as f:
        ckpt = pickle.load(f)
    assert ckpt.get('ema_state') is not None
    assert ckpt.get('feature_bank') is not None
    assert ckpt.get('buffers_present') is True

    params2, _, saved_step = load_stage2_params(
        out, cfg, jax.random.PRNGKey(0))
    assert saved_step == 5, "load_stage2_params must preserve the time origin"
    ema2, bank2, start_step = load_cog_buffers(out, params2, cfg)
    assert start_step == 5, "load_cog_buffers must preserve the time origin"
    np.testing.assert_allclose(np.asarray(ema2['sparse']['N']),
                               np.asarray(ema['sparse']['N']))
    np.testing.assert_allclose(np.asarray(ema2['sparse']['m']),
                               np.asarray(ema['sparse']['m']))
    np.testing.assert_array_equal(np.asarray(ema2['sparse']['last_used']),
                                  np.asarray(ema['sparse']['last_used']))
    assert isinstance(bank2, FeatureBank)
    assert len(bank2) == 2
    np.testing.assert_allclose(np.asarray(bank2.vectors),
                               np.asarray(bank.vectors))

    # Dead-code detection uses the restored absolute step origin: row 0 was
    # last used at step 2, so it is dead at step 5 with t_dead=2. Restarting the
    # clock at 0 would see no dead rows (the bug this guards against).
    dead = bank2.dead_indices(ema2['sparse']['last_used'], start_step,
                              t_dead=2)
    np.testing.assert_array_equal(dead, [0])
    assert len(bank2.dead_indices(ema2['sparse']['last_used'], 0,
                                  t_dead=2)) == 0

    # A v2-era pickle without the buffers is not a valid cognitive
    # checkpoint; it must be rejected instead of silently starting fresh.
    ckpt.pop('ema_state')
    with open(f"{out}/cog_params.pkl", "wb") as f:
        pickle.dump(ckpt, f)
    with np.testing.assert_raises(ValueError):
        load_cog_buffers(out, params2, cfg)


def test_resume_loop_clock_policy_a_and_guards():
    """Offset drives schedule/log/save/bank; abs_step drives state only."""
    from train.cog_train import _lr_decay_steps, _resume_loop_plan

    plan = _resume_loop_plan(start_step=5, steps=3, save_every=2,
                             log_every=1, bank_check_interval=2)
    assert [p['offset'] for p in plan] == [0, 1, 2]
    assert [p['abs_step'] for p in plan] == [5, 6, 7]
    # The first new step (offset 0) is the resume point: never re-save it.
    assert plan[0]['save'] is False
    assert plan[1]['save'] is False and plan[2]['save'] is True
    assert plan[0]['log'] is False
    assert plan[1]['log'] is True and plan[2]['log'] is True
    assert plan[0]['bank'] is False
    assert plan[1]['bank'] is False and plan[2]['bank'] is True

    # Regression shape: start_step is itself a multiple of every guard. Absolute
    # guards would save `step_000005`/log/check the bank on the first new step.
    risky = _resume_loop_plan(start_step=5, steps=2, save_every=5,
                              log_every=5, bank_check_interval=5)
    assert risky[0]['abs_step'] == 5
    assert not (risky[0]['save'] or risky[0]['log'] or risky[0]['bank'])
    assert risky[1]['abs_step'] == 6

    # Policy A: continue the original cosine curve over the extended horizon.
    # At the first resumed step the LR is above the alpha=0.1 floor.
    init = 1e-3
    assert _lr_decay_steps(5, 3) == 8
    schedule = optax.cosine_decay_schedule(
        init_value=init, decay_steps=_lr_decay_steps(5, 3), alpha=0.1)
    first_lr = float(schedule(plan[0]['abs_step']))
    assert first_lr > 0.1 * init + 1e-12

    # Dead-code detection uses the absolute clock: a row last used at step 2 is
    # dead at the resumed first step 5 with t_dead=2, but not at offset 0.
    bank = FeatureBank(4, 2)
    last = jnp.array([2, 4], dtype=jnp.int32)
    np.testing.assert_array_equal(
        bank.dead_indices(last, plan[0]['abs_step'], t_dead=2), [0])
    assert len(bank.dead_indices(last, plan[0]['offset'], t_dead=2)) == 0


def test_checkpoint_without_buffers_is_not_resumable(tmp_path):
    from train.cog_train import load_cog_buffers, save_cog_checkpoint

    cfg = _tiny_cfg()
    params, self_state = _tiny_params(cfg)
    identity = _fake_identity(tmp_path, cfg.vocab_size)

    # Training/export callers that omit the buffers must opt in explicitly.
    with pytest.raises(ValueError, match="requires ema_state"):
        save_cog_checkpoint(
            params, str(tmp_path / "refuse"), 0, run_identity=identity,
            train_cfg=cfg, self_state=self_state)

    out = str(tmp_path / "export")
    save_cog_checkpoint(
        params, out, 0, run_identity=identity, train_cfg=cfg,
        self_state=self_state, allow_missing_buffers=True)
    with open(f"{out}/cog_params.pkl", "rb") as f:
        ckpt = pickle.load(f)
    assert ckpt['buffers_present'] is False
    with pytest.raises(ValueError, match="not resumable"):
        load_cog_buffers(out, params, cfg)
