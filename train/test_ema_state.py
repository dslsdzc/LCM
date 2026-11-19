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

from train.cog_loop import cog_loop_scan
from train.config import LCMConfig
from train.ema_state import (
    ema_update_binding, ema_update_manifold, ema_update_sparse, init_ema_state,
)
from train.feature_bank import FeatureBank
from train.hyp import exp_map


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
    for f_i, fam in enumerate(('key', 'val', 'bind')):
        residuals[fam] = []
        selected[fam] = []
        for l in range(n_layers):
            sel = (f_i + l) % M
            residuals[fam].append(jnp.array(
                [[float(sel) * (f_i + 1) + 0.5, 0.25]]))
            selected[fam].append(sel)

    new_params, new_state = ema_update_binding(
        params, state, residuals, gamma=0.99, step=17)

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
        assert 'danger' not in opt_state

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
        assert 'sparse' not in opt_state
        if aux.get('self_state') is not None:
            self_state = aux['self_state']

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
    ema['sparse']['last_used'] = ema['sparse']['last_used'].at[0].set(7)

    bank = FeatureBank(8, cfg.d_model)
    bank.add(np.arange(cfg.d_model, dtype=np.float32))
    bank.add(np.ones(cfg.d_model, dtype=np.float32) * 3)

    out = str(tmp_path / "ckpt")
    save_cog_checkpoint(
        params, out, 4, run_identity=_fake_identity(tmp_path, cfg.vocab_size),
        train_cfg=cfg, self_state=self_state, ema_state=ema,
        feature_bank=bank)

    with open(f"{out}/cog_params.pkl", "rb") as f:
        ckpt = pickle.load(f)
    assert ckpt.get('ema_state') is not None
    assert ckpt.get('feature_bank') is not None

    params2, _ = load_stage2_params(out, cfg, jax.random.PRNGKey(0))
    ema2, bank2 = load_cog_buffers(out, params2, cfg)
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

    # A v2-era pickle without the buffers is not a valid cognitive
    # checkpoint; it must be rejected instead of silently starting fresh.
    ckpt.pop('ema_state')
    with open(f"{out}/cog_params.pkl", "wb") as f:
        pickle.dump(ckpt, f)
    with np.testing.assert_raises(ValueError):
        load_cog_buffers(out, params2, cfg)
