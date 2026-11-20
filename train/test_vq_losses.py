"""R18/R19: layered codebook/commitment losses + routing training.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_vq_losses.py -v
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from train.config import LCMConfig
from train.encoder import encoder_forward
from train.hyp import exp_map, log_map, mobius_add, poincare_similarity, safe_unit
from train.lattices import (
    contrast_forward, hrq_forward, init_contrast_params, init_hrq_params,
    init_lowrank_params, init_route_params, init_simvq, lowrank_forward,
    route_commit_loss, routing_gate,
)
from train.losses import (
    codebook_loss, cognitive_vq_losses, commitment_loss, hrq_vq_loss,
)
from train.param_groups import EMA_PATHS


def _leaf_mass(tree):
    return sum(float(jnp.sum(jnp.abs(x)))
               for x in jax.tree_util.tree_leaves(tree))


def test_codebook_loss_grad_only_to_C():
    r = jnp.array([[1.0, 0.0]])
    C = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    idx = jnp.array([1])
    g_r, g_C = jax.grad(lambda r, C: codebook_loss(r, C, idx), argnums=(0, 1))(r, C)
    assert jnp.allclose(g_r, 0.0)
    assert not jnp.allclose(g_C, 0.0)


def test_commitment_loss_grad_only_to_r():
    r = jnp.array([[1.0, 0.0]])
    C = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    idx = jnp.array([1])
    g_r, g_C = jax.grad(lambda r, C: commitment_loss(r, C, idx), argnums=(0, 1))(r, C)
    assert not jnp.allclose(g_r, 0.0)
    assert jnp.allclose(g_C, 0.0)


def test_routing_hard_is_deterministic():
    p = init_route_params(jax.random.PRNGKey(0), 6, 8)
    z = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    m1, _, _ = routing_gate(p, z, tau=0.5, hard=True, rng=jax.random.PRNGKey(2))
    m2, _, _ = routing_gate(p, z, tau=0.5, hard=True, rng=jax.random.PRNGKey(3))
    assert jnp.array_equal(m1, m2)


def test_route_commit_loss_gives_C_route_gradient():
    p = init_route_params(jax.random.PRNGKey(0), 6, 8)
    z = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    idx = jnp.array([0, 1])
    g = jax.grad(lambda C: route_commit_loss(z, C[idx], 0.25))(p['C_route'])
    assert not jnp.allclose(g, 0.0), "C_route must receive gradient"


# ── HRQ loss geometry (Important 1) ─────────────────────────────────────────

def _two_proto_hrq(top_vectors, tau_fallback=0.0):
    d = top_vectors.shape[-1]
    return {
        'top': {'A': top_vectors, 'W': jnp.eye(d)},
        'fine': [init_simvq(jax.random.PRNGKey(7), 1, d)],
    }


def test_hrq_fine_pair_is_log_map_of_composed_state_not_tangent_residual():
    """cont_k must be log_map(c_top ⊕ r), not log_map(r).

    c_top sits far from the origin so the two differ by much more than the
    test tolerance; the fine codebook is tiny, so the comparison isolates the
    composition, not the quantised code.
    """
    C_top = jnp.array([[0.9, 0.0], [0.0, 0.0]])
    params = _two_proto_hrq(C_top)
    # z_P lands near the far prototype (Poincaré similarity is larger=farther,
    # so the nearest is argmin), making c_top far from the origin.
    z = jnp.array([[0.6, 0.0]])

    _, top_idx, _, pairs = hrq_forward(params, z, tau_fallback=0.0)

    C_top_P = exp_map(C_top)
    c_top = C_top_P[int(top_idx[0])]
    # R6: HRQ's query domain is exp_map(safe_unit(z)).
    r0 = mobius_add(exp_map(safe_unit(z)), -c_top)
    want = log_map(mobius_add(c_top, r0))       # composed-output space
    wrong = log_map(r0)                          # tangent residual
    assert jnp.allclose(pairs['cont'][1], want, atol=1e-5)
    assert not jnp.allclose(want, wrong, atol=1e-4), \
        "test setup must separate the two spaces"


def test_hrq_fallback_top_term_targets_the_softmax_blend():
    """On the fallback branch the top target is log_map(c_top_w), not c_top*."""
    C_top = jnp.array([[0.5, 0.0], [-0.5, 0.0]])
    params = _two_proto_hrq(C_top)
    z = jnp.zeros((1, 2))  # exactly between the two prototypes → fallback

    _, _, _, pairs = hrq_forward(params, z, tau_fallback=0.1)

    C_top_P = exp_map(C_top)
    sims = poincare_similarity(
        exp_map(z)[:, None, :], C_top_P[None, :, :]).squeeze(-1)
    weights = jax.nn.softmax(-sims, axis=-1)
    c_top_w = jnp.einsum('bm,md->bd', weights, C_top_P)
    assert jnp.allclose(pairs['quant'][0], log_map(c_top_w), atol=1e-5)


def test_hrq_vq_loss_requires_aux_key():
    params = _two_proto_hrq(jnp.array([[0.9, 0.0], [0.0, 0.0]]))
    z = jnp.array([[0.3, 0.0]])
    cfg = SimpleNamespace(beta_vq=0.25)
    with pytest.raises(KeyError, match="hrq_vq"):
        hrq_vq_loss(params, z, {'wrong_key': None}, cfg)


# ── Per-layer count/gradient coverage (n_layers > 1) ────────────────────────

def test_per_layer_gradients_and_layer_counts_with_multiple_layers():
    d, B = 4, 2
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (B, d)) * 0.1
    beta = SimpleNamespace(beta_vq=0.25)

    # HRQ: top + two fine layers, both terms; every fine codebook gets gradient.
    hp = init_hrq_params(key, d, 4, 3, n_layers=2)
    _, _, _, pairs = hrq_forward(hp, z, tau_fallback=0.0)
    assert len(pairs['cont']) == 3 and len(pairs['quant']) == 3
    cb, cm = hrq_vq_loss(hp, z, pairs, beta)
    assert cb.shape == (3,) and cm.shape == (3,)

    def hrq_codebook(p):
        _, _, _, pr = hrq_forward(p, z, tau_fallback=0.0)
        return jnp.sum(hrq_vq_loss(p, z, pr, beta)[0])

    g_hrq = jax.grad(hrq_codebook)(hp)
    assert _leaf_mass(g_hrq['top']) > 0.0
    for layer in g_hrq['fine']:
        assert _leaf_mass(layer) > 0.0

    # LowRank: two rank layers; U_k, A_V and W_V all receive codebook gradient.
    ranks = (1, 2)
    lp = init_lowrank_params(key, d, 3, ranks)
    _, lr_aux = lowrank_forward(lp, z, ranks, return_aux=True)
    assert len(lr_aux['residuals']) == len(ranks) == 2

    def lowrank_codebook(p):
        _, aux = lowrank_forward(p, z, ranks, return_aux=True)
        V = p['A_V'] @ p['W_V']
        total = jnp.array(0.0)
        for u_k, rank, res, idx in zip(
                p['U'], ranks, aux['residuals'], aux['indices']):
            total = total + codebook_loss(
                res, u_k @ V[:, :rank].T, idx, 0.25)
        return total

    g_lr = jax.grad(lowrank_codebook)(lp)
    assert _leaf_mass(g_lr['A_V']) > 0.0
    assert _leaf_mass(g_lr['W_V']) > 0.0
    for layer in g_lr['U']:
        assert _leaf_mass(layer) > 0.0

    # Contrast: two layers per family; C_b (and C_a) get gradient on every layer.
    cp = init_contrast_params(key, d, 3, n_layers=2)
    _, ct_idx, ct_res = contrast_forward(cp, z)
    assert len(ct_res['a']) == len(ct_res['b']) == 2
    assert len(ct_idx['a']) == len(ct_idx['b']) == 2

    def contrast_codebook(p):
        _, aux_idx, aux_res = contrast_forward(p, z)
        total = jnp.array(0.0)
        for fam in ('a', 'b'):
            for cb, res, idx in zip(
                    p[f'C_{fam}'], aux_res[fam], aux_idx[fam]):
                total = total + codebook_loss(
                    res, cb['A'] @ cb['W'], idx, 0.25)
        return total

    g_ct = jax.grad(contrast_codebook)(cp)
    for fam in ('a', 'b'):
        assert len(g_ct[f'C_{fam}']) == 2
        for layer in g_ct[f'C_{fam}']:
            assert _leaf_mass(layer) > 0.0


# ── Full-step composition ──────────────────────────────────────────────────

CFG = LCMConfig(
    d_model=8, d_ff=12, n_heads=2, d_head=4, vocab_size=16,
    max_seq_len=6, n_encoder_layers=1,
    M_top=6, M_fine=3, n_hrq_layers=1,
    M_sparse=6, M_lr=6, M_man=6, M_bind=6, M_contrast=6,
    n_bind_layers=1, n_contrast_layers=1, n_self_codes=3,
    use_bf16=False, max_inference_steps=2,
)


@pytest.fixture(scope="module")
def built():
    from train.cognitive_step import six_lattice_step
    from train.model import init_all_params

    params, _, _ = init_all_params(CFG, jax.random.PRNGKey(0))
    x = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    z = safe_unit(encoder_forward(params['encoder'], x, CFG.n_heads))
    _, aux = six_lattice_step(
        z, params, CFG, training=True, rng=jax.random.PRNGKey(1))
    return params, z, aux


def test_hrq_vq_loss_has_top_plus_each_fine_layer(built):
    params, z, aux = built
    cb, cm = hrq_vq_loss(params['hrq'], z, aux, CFG)
    n_expected = 1 + CFG.n_hrq_layers
    assert cb.shape == (n_expected,), f"codebook terms: {cb.shape}"
    assert cm.shape == (n_expected,), f"commitment terms: {cm.shape}"
    assert jnp.all(cb >= 0.0)
    assert jnp.all(cm >= 0.0)
    assert jnp.sum(cb) > 0.0
    assert jnp.sum(cm) > 0.0


def test_ema_codebook_loss_skipped_but_commitment_trains_r(built):
    """EMA-owned codebooks get no codebook gradient; commitment still moves r.

    The skip set is iterated from ``EMA_PATHS`` (the parameter-group truth),
    not from a second hardcoded list. The gradient check runs through the real
    step composition.
    """
    from train.cognitive_step import six_lattice_step

    params, z, aux = built
    total, parts = cognitive_vq_losses(params, aux, CFG)
    assert float(parts['sparse_codebook']) == 0.0
    assert float(parts['sparse_commitment']) > 0.0
    assert float(total) > 0.0

    # Reported codebook terms for the EMA-managed lattices must be zero too.
    ema_codebook_parts = {
        ('sparse', 'C'): 'sparse_codebook',
        ('manifold', 'C'): 'manifold_codebook',
        ('binding', 'key_cb'): 'binding_codebook',
        ('binding', 'val_cb'): 'binding_codebook',
        ('binding', 'bind_cb'): 'binding_codebook',
    }
    assert set(ema_codebook_parts) <= set(EMA_PATHS)
    for path, part in ema_codebook_parts.items():
        assert float(parts[part]) == 0.0, f"{path} codebook term must be 0"

    def loss_fn(p, zz):
        _, a = six_lattice_step(
            zz, p, CFG, training=True, rng=jax.random.PRNGKey(1))
        return cognitive_vq_losses(p, a, CFG)[0]

    (g_params, g_z) = jax.grad(loss_fn, argnums=(0, 1))(params, z)
    assert not jnp.allclose(g_z, 0.0), \
        "commitment loss must give the encoder-side input a gradient"

    for path in sorted(EMA_PATHS):
        node = g_params
        present = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                present = False  # e.g. a partial hand-built params tree
                break
            node = node[key]
        if not present:
            continue
        assert _leaf_mass(node) == 0.0, \
            f"EMA codebook {path} must not receive a codebook gradient"

    for path in (('hrq',), ('lowrank', 'U'), ('contrast', 'C_a'),
                 ('contrast', 'C_b'), ('route', 'C_route')):
        node = g_params
        for key in path:
            node = node[key]
        assert _leaf_mass(node) > 0.0, \
            f"pure-gradient codebook {path} must receive gradient"


def test_legacy_compute_vq_loss_route_matches_r18_formula(built):
    """The legacy Stage-1/2 path keeps the R18 routing formula."""
    from train.losses import compute_vq_loss

    params, z, aux = built
    comps = compute_vq_loss(params, aux, z, CFG)
    C_route = params['route']['C_route']
    expected = CFG.beta_vq * jnp.mean(
        (jax.lax.stop_gradient(safe_unit(z))
         - C_route[aux['route_idx']]) ** 2)
    assert jnp.allclose(comps['route'], expected, atol=1e-6)
