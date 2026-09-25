"""Dynamic codebook expansion (continual learning) regression tests.

`expand_for_new_task` is a *transaction*: the parameter tree, the EMA
accumulators, the local value scalars, the utilisation mask, the access
counters and the EWC bookkeeping all change shape together, and the optimizer
state is rebuilt. Getting any one of them wrong does not fail at the expansion
— it fails on the NEXT training step, with the expansion already committed and
the checkpoint about to be written.

So these tests do two things: inspect the post-expansion shapes, and then run a
real training step. A shape table can agree with itself and still be wrong; the
step is what proves it.

Run from repo root:  JAX_PLATFORMS=cpu be/bin/python -m train.test_fixes_expand
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from train.config import LCMConfig
from train.continual import (
    expansion_key_count, pad_to_shape, detect_new_task, _param_groups,
)
import train.train as train_mod
from train.train import create_train_state, train_step, expand_for_new_task


def _small_cfg(**over):
    """Small config so CPU tests compile and run fast.

    `r_max` must equal `max(ranks)`: the binding lattice forms
    `W_k = V @ A_k` with V (d, r_max) from the low-rank basis and A_k
    (cfg.r_max, d), so a config where they disagree only fails once a forward
    pass runs (the default r_max=8 with ranks=(2,4) is one such config).
    """
    ranks = over.pop('ranks', (2, 4))
    base = dict(
        d_model=32, vocab_size=32, max_seq_len=16, d_ff=48, n_heads=4,
        n_encoder_layers=1, M_top=16, M_fine=8, n_hrq_layers=2,
        M_sparse=16, M_lr=16, M_man=16, M_bind=16, M_contrast=16,
        n_bind_layers=2, n_contrast_layers=2, n_self_codes=4,
        ranks=ranks, r_max=max(ranks), max_inference_steps=4,
        n_new_codebook_entries=3,          # small, so the resize is observable
        ewc_fisher_samples=3,              # 200 default is far too slow here
        replay_capacity=64, replay_ratio=0.5, replay_weight=0.1,
        shift_detection_min_samples=2,
        use_bf16=False,
    )
    base.update(over)
    return LCMConfig(**base)


def _batch(cfg, B=8, N=8, seed=0):
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, cfg.vocab_size, (B, N)).astype(np.int32)
    targets = rng.integers(0, cfg.vocab_size, (B, N)).astype(np.int32)
    return jnp.array(inputs), jnp.array(targets)


def _step(state, cfg, batch, seed=0):
    """Run one train_step with the config the test controls."""
    original = train_mod._get_global_cfg
    train_mod._get_global_cfg = lambda: cfg
    try:
        return train_step(state, batch, jax.random.PRNGKey(seed))
    finally:
        train_mod._get_global_cfg = original


def _expand(state, cfg, batch, seed=1):
    original = train_mod._get_global_cfg
    train_mod._get_global_cfg = lambda: cfg
    try:
        return expand_for_new_task(state, cfg, jax.random.PRNGKey(seed),
                                   batch=batch)
    finally:
        train_mod._get_global_cfg = original


def _codebook_rows(params):
    """Every array whose leading axis is a codebook size, as path -> rows."""
    out = {'hrq/top/A': params['hrq']['top']['A'].shape[0],
           'sparse/C': params['sparse']['C'].shape[0],
           'manifold/C': params['manifold']['C'].shape[0],
           'manifold/T': params['manifold']['T'].shape[0]}
    for l, fb in enumerate(params['hrq']['fine']):
        out[f'hrq/fine/{l}/A'] = fb['A'].shape[0]
    for l, u in enumerate(params['lowrank']['U']):
        out[f'lowrank/U/{l}'] = u.shape[0]
    for fam in ('key_cb', 'val_cb', 'bind_cb'):
        for l, cb in enumerate(params['binding'][fam]):
            out[f'binding/{fam}/{l}/A'] = cb['A'].shape[0]
    for fam in ('C_a', 'C_b'):
        for l, cb in enumerate(params['contrast'][fam]):
            out[f'contrast/{fam}/{l}/A'] = cb['A'].shape[0]
    for name, v in params['value_scalars'].items():
        out[f'value_scalars/{name}'] = v.shape[0]
    return out


def test_expansion_key_count_matches_consumption():
    """The RNG split size must track the actual number of expanded tensors.

    A hardcoded split that drifts below the real count indexes past the end of
    the key array, and the failure lands on whichever codebook happens to cross
    the boundary rather than on the miscount. Checked on a non-default config
    so a stale constant cannot pass by coincidence.
    """
    for cfg in (_small_cfg(),
                _small_cfg(n_bind_layers=3, n_contrast_layers=4,
                           n_hrq_layers=3, ranks=(2, 4, 8, 16))):
        want = (1 + cfg.n_hrq_layers + 1 + len(cfg.ranks) + 2
                + 3 * cfg.n_bind_layers + 2 * cfg.n_contrast_layers)
        got = expansion_key_count(cfg)
        assert got == want, f"expansion_key_count {got} != {want} for {cfg}"
        # Exercising the expansion is what actually consumes the keys; it
        # asserts internally that ki == n_keys.
        state = create_train_state(cfg, jax.random.PRNGKey(0))
        params = state['params']
        from train.continual import expand_lattice_codebooks
        before = _codebook_rows(params)
        expand_lattice_codebooks(params, cfg.n_new_codebook_entries,
                                 jax.random.PRNGKey(3), cfg)
        after = _codebook_rows(params)
        for k, n_before in before.items():
            if k.startswith('value_scalars/'):
                continue  # expanded by expand_value_scalars, not by this call
            assert after[k] == n_before + cfg.n_new_codebook_entries, \
                f"{k}: {n_before} -> {after[k]}, expected +{cfg.n_new_codebook_entries}"
        print(f"  [PASS] key count == consumption "
              f"({got} keys, {len(before)} codebook tensors grew)")


def test_expansion_resizes_every_codebook_shaped_structure():
    """params / EMA / value scalars / mask / counters / EWC all grow together."""
    cfg = _small_cfg()
    batch = _batch(cfg)
    state = create_train_state(cfg, jax.random.PRNGKey(0))
    # Populate EMA, access counters, seen mask and the replay buffer.
    for s in range(3):
        state, _ = _step(state, cfg, batch, seed=s)

    n_new = cfg.n_new_codebook_entries
    assert state['continual'].access_counters, \
        "access counters should exist after a real step (test setup broken)"
    assert 'hrq' in state['seen_masks'], "seen mask missing (test setup broken)"

    before_rows = _codebook_rows(state['params'])
    before_ema = {'sparse/N': state['ema_state']['sparse']['N'].shape[0],
                  'sparse/m': state['ema_state']['sparse']['m'].shape[0],
                  'manifold/N': state['ema_state']['manifold']['N'].shape[0]}
    before_mask = state['seen_masks']['hrq'].shape[0]
    before_ctr = {k: v.shape[0] for k, v in
                  state['continual'].access_counters.items()}
    # protected_params / fisher_diag are snapshotted *inside* expansion, so the
    # only meaningful pre-expansion record is the param tree they must mirror.
    assert state['continual'].protected_params == {}, \
        "no EWC snapshot should exist before the first expansion"
    before_prot = {p: a.shape for p, a in _param_groups(state['params'])}
    prefix = {'sparse_C': np.asarray(state['params']['sparse']['C']).copy(),
              'vs': np.asarray(state['params']['value_scalars']['sparse']).copy()}

    state = _expand(state, cfg, batch)

    # 1. every codebook-shaped array grew by exactly n_new
    after_rows = _codebook_rows(state['params'])
    assert set(after_rows) == set(before_rows)
    for k, n_before in before_rows.items():
        assert after_rows[k] == n_before + n_new, \
            f"{k}: {n_before} -> {after_rows[k]}, expected +{n_new}"

    # 2. expansion appends; the trained prefix must be untouched
    assert np.array_equal(np.asarray(state['params']['sparse']['C'])[:len(prefix['sparse_C'])],
                          prefix['sparse_C']), "expansion rewrote existing rows"
    assert np.array_equal(np.asarray(state['params']['value_scalars']['sparse'])[:len(prefix['vs'])],
                          prefix['vs']), "expansion rewrote existing value scalars"

    # 3. EMA accumulators grew with their codebooks
    assert state['ema_state']['sparse']['N'].shape[0] == before_ema['sparse/N'] + n_new
    assert state['ema_state']['sparse']['m'].shape[0] == before_ema['sparse/m'] + n_new
    assert state['ema_state']['manifold']['N'].shape[0] == before_ema['manifold/N'] + n_new
    for fam in ('key', 'val', 'bind'):
        for l, st in enumerate(state['ema_state']['binding'][fam]):
            assert st['N'].shape[0] == state['params']['binding'][
                {'key': 'key_cb', 'val': 'val_cb', 'bind': 'bind_cb'}[fam]][l]['A'].shape[0], \
                f"binding EMA {fam}/{l} out of step with its codebook"

    # 4. utilisation mask and access counters grew
    assert state['seen_masks']['hrq'].shape[0] == before_mask + n_new
    for k, n_before in before_ctr.items():
        assert state['continual'].access_counters[k].shape[0] == n_before + n_new, \
            f"access counter {k} not resized"

    # 5. EWC bookkeeping was padded to the NEW shapes (a stale shape makes the
    #    next compute_ewc_loss broadcast-fail), zeros apppended.
    after_prot = {k: v.shape for k, v in state['continual'].protected_params.items()}
    assert set(after_prot) == set(before_prot), \
        "the EWC snapshot must cover exactly the parameter paths"
    assert set(state['continual'].fisher_diag) == set(before_prot), \
        "fisher_diag paths must mirror protected_params"
    grew = 0
    for k, shp in after_prot.items():
        old = before_prot[k]
        if shp != old:
            assert len(shp) == len(old) and all(
                n >= o for n, o in zip(shp, old)), f"protected {k}: {old} -> {shp}"
            assert shp[0] == old[0] + n_new, f"protected {k}: {old} -> {shp}"
            grew += 1
    assert grew > 0, "protected_params did not grow with the codebooks"
    # Zero-padded region ⇒ Fisher 0 ⇒ no protection for the added rows.
    for k, arr in state['continual'].fisher_diag.items():
        if arr.shape != before_prot[k]:
            old_n = before_prot[k][0]
            assert float(jnp.abs(arr[old_n:]).sum()) == 0.0, \
                f"added Fisher rows for {k} must be zero"

    print(f"  [PASS] expansion resized {len(after_rows)} codebook tensors, EMA, "
          f"mask, counters and {grew} EWC tensors by +{n_new}")


def test_training_step_after_expansion():
    """The step after an expansion must run — this is the actual regression.

    Every resize bug in this file surfaces here rather than at the expansion:
    the codebook/EMA mismatch raises, and a stale protected_params broadcasts
    into a wrong-shaped EWC gradient.
    """
    cfg = _small_cfg()
    batch = _batch(cfg)
    state = create_train_state(cfg, jax.random.PRNGKey(0))
    for s in range(4):
        state, _ = _step(state, cfg, batch, seed=s)

    state = _expand(state, cfg, batch)
    assert state['continual'].task_id == 1, "task_id not advanced by expansion"

    state, comps1 = _step(state, cfg, batch, seed=9)
    for k in ('lm', 'vq', 'ewc', 'replay', 'val', 'orth'):
        assert k in comps1, f"component '{k}' missing after expansion"
        assert np.isfinite(float(comps1[k])), f"component '{k}' is not finite"
    assert float(comps1['lm']) > 0, "LM loss collapsed after expansion"

    # First step after expansion: EWC is anchored at the snapshot taken *during*
    # the expansion, so theta == theta* and the term is exactly zero. That is
    # the correct value, and it is why "ewc reads 0" is not by itself evidence
    # that the term is dead — it only becomes non-zero once the parameters move.
    assert float(comps1['ewc']) == 0.0, \
        f"EWC should be zero on the anchor step, got {float(comps1['ewc'])}"

    # Replay needs task_id > 0 and an old-task buffer, both of which exist now.
    assert float(comps1['replay']) > 0.0, \
        "replay contribution is zero after a task switch"

    state, comps2 = _step(state, cfg, batch, seed=10)
    assert float(comps2['ewc']) > 0.0, \
        "EWC never leaves zero — the protection is not wired into the gradient"

    # The Fisher must be a real estimate, not the constant placeholder
    # (`ones_like(val) * 1e-4`) it used to be filled with.
    fish = [np.asarray(v) for v in state['continual'].fisher_diag.values()]
    assert fish, "fisher_diag is empty after an expansion with a batch"
    assert any(v.std() > 0 for v in fish if v.size > 1), \
        "fisher_diag is constant across entries — looks like a placeholder"
    assert any(v.max() > 0 for v in fish), "fisher_diag is all zero"

    print(f"  [PASS] training steps after expansion: lm={float(comps2['lm']):.3f} "
          f"ewc(anchor)={float(comps1['ewc']):.3g} ewc(step2)={float(comps2['ewc']):.3g} "
          f"replay={float(comps1['replay']):.4f}")


def test_expansion_without_batch_skips_ewc_loudly():
    """No batch → no Fisher → EWC must be declared off, not faked.

    The previous implementation filled fisher_diag with a constant
    `ones_like(val) * 1e-4`, which silently turns EWC into an unweighted L2
    pull toward the previous task.
    """
    cfg = _small_cfg()
    batch = _batch(cfg)
    state = create_train_state(cfg, jax.random.PRNGKey(0))
    state, _ = _step(state, cfg, batch, seed=0)
    state = _expand(state, cfg, batch, )
    # second expansion with no batch
    original = train_mod._get_global_cfg
    train_mod._get_global_cfg = lambda: cfg
    try:
        state = expand_for_new_task(state, cfg, jax.random.PRNGKey(4), batch=None)
    finally:
        train_mod._get_global_cfg = original
    assert state['continual'].fisher_diag == {}, \
        "fisher_diag must stay empty when no batch is available, not be faked"
    state, comps = _step(state, cfg, batch, seed=11)
    assert float(comps['ewc']) == 0.0, \
        "EWC should contribute nothing when the Fisher is unavailable"
    print("  [PASS] expansion without a batch leaves the Fisher empty "
          "(EWC off, declared) instead of faking it")


def test_pad_to_shape_never_truncates():
    """pad_to_shape is the resize used for EWC bookkeeping."""
    a = jnp.ones((2, 3))
    assert pad_to_shape(a, (5, 3)).shape == (5, 3)
    assert float(pad_to_shape(a, (5, 3))[4].sum()) == 0.0, "padding must be zero"
    assert pad_to_shape(a, (2, 3)) is a, "no-op resize should return the input"
    assert pad_to_shape(a, (1, 3)).shape == (2, 3), "must never truncate"
    print("  [PASS] pad_to_shape pads with zeros and never truncates")


def test_shift_detection_needs_latents_not_token_ids():
    """detect_new_task must reject a (B, seq_len) token batch."""
    from train.continual import init_continual_state
    cfg = _small_cfg()
    st = init_continual_state(cfg.d_model)
    # token ids of the wrong width must be refused, not silently adopted
    try:
        detect_new_task(jnp.zeros((4, cfg.max_seq_len)),
                        init_continual_state(cfg.d_model), 2.0)
        raise AssertionError(
            "detect_new_task accepted a token-id batch; it must reject a width "
            "that does not match the retained latent dimension")
    except ValueError:
        pass
    # correct width: warm-up suppresses a task signal until min_samples
    z = jnp.zeros((4, cfg.d_model))
    fired, st = detect_new_task(z, st, 2.0, min_samples=32)
    assert fired is False, "shift fired during warm-up"
    st.z_mean_ema = jnp.zeros(cfg.d_model)
    st.z_cov_ema = jnp.eye(cfg.d_model) * 0.1
    st.n_seen = 1000
    fired, st = detect_new_task(jnp.ones((4, cfg.d_model)) * 5.0, st, 2.0)
    assert fired is True, "a 5-sigma mean shift should be detected"
    print("  [PASS] detect_new_task rejects token ids, warms up, then fires")


if __name__ == '__main__':
    test_expansion_key_count_matches_consumption()
    test_expansion_resizes_every_codebook_shaped_structure()
    test_training_step_after_expansion()
    test_expansion_without_batch_skips_ewc_loudly()
    test_pad_to_shape_never_truncates()
    test_shift_detection_needs_latents_not_token_ids()
    print('All expansion tests passed.')
