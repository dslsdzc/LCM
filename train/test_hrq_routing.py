"""R1: Poincare top-k HRQ routing, value rescoring, relative-gap fallback.

The codebook is crafted, not initialised: random init leaves the top-2
similarities so close together that every sample takes the fallback branch and
the hard-routing assertions pass vacuously.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_hrq_routing.py -v
"""
import jax
import jax.numpy as jnp
import pytest

from train.hyp import exp_map, log_map, mobius_add, poincare_similarity, safe_unit
from train.lattices import _topk_indices, hrq_forward, init_hrq_params, init_simvq


def _crafted_hrq(d=8, M_top=4):
    """Codebook with one well-separated prototype per axis."""
    p = init_hrq_params(jax.random.PRNGKey(0), d, M_top, 4, 1)
    p['top'] = {'A': jnp.eye(M_top, d) * 0.9, 'W': jnp.eye(d)}
    return p


def _sims(params, z):
    """R6 reference: z_P = exp_map(safe_unit(z)), C_P = exp_map(C)."""
    z_P = exp_map(safe_unit(z))
    C_P = exp_map(params['top']['A'] @ params['top']['W'])
    sims = poincare_similarity(z_P[:, None, :], C_P[None, :, :]).squeeze(-1)
    return z_P, C_P, sims


def _relative_gap(sims):
    s = jnp.sort(sims, axis=-1)
    return (s[:, 1] - s[:, 0]) / (s[:, 0] + 1e-8)


def test_v_zero_matches_pure_poincare():
    """With v=0 the value term vanishes, so top-1 is the pure Poincare argmin.

    ``poincare_similarity`` is a distance-monotone quantity (larger = farther),
    so the nearest prototype is the *minimiser*. This must hold for both
    ``value_scalars=None`` and an explicit zero vector — they are the same
    computation, and a branch on ``alpha_val > 0`` must not change the result.
    """
    p = _crafted_hrq()
    z = jax.random.normal(jax.random.PRNGKey(1), (3, 8)) * 0.3
    _, _, sims = _sims(p, z)
    want = sims.argmin(axis=-1)

    for vs in (None, jnp.zeros(4)):
        o, idx, top_sim, _ = hrq_forward(
            p, z, value_scalars=vs, alpha_val=0.1, value_topk=4)
        assert jnp.array_equal(idx, want)
        assert jnp.allclose(
            top_sim,
            jnp.take_along_axis(sims, want[:, None], axis=-1)[:, 0], atol=1e-6)
        assert jnp.all(jnp.isfinite(o))


def test_topk_helper_matches_the_routing_order():
    p = _crafted_hrq()
    z = jax.random.normal(jax.random.PRNGKey(1), (3, 8)) * 0.3
    _, _, sims = _sims(p, z)
    topk = _topk_indices(p, z, k=2)
    assert topk.shape == (3, 2)
    want = jnp.argsort(sims, axis=-1)[:, :2]
    assert jnp.array_equal(topk, want)


def test_value_bias_changes_top1_only_within_topk():
    """v can promote a candidate inside the top-k; it cannot import an outsider."""
    p = _crafted_hrq(M_top=8)
    z = jax.random.normal(jax.random.PRNGKey(2), (1, 8)) * 0.3
    topk = _topk_indices(p, z, k=2)
    assert topk.shape == (1, 2)

    _, idx0, _, _ = hrq_forward(
        p, z, value_topk=2, value_scalars=jnp.zeros(8), alpha_val=10.0)
    assert int(idx0[0]) == int(topk[0, 0]), "v=0 must keep the geometric top-1"

    # Making the winner even more valuable changes nothing.
    _, idx_same, _, _ = hrq_forward(
        p, z, value_topk=2, alpha_val=10.0,
        value_scalars=jnp.zeros(8).at[topk[0, 0]].set(1e6))
    assert int(idx_same[0]) == int(topk[0, 0])

    # The runner-up is inside the top-k, so a large value may promote it.
    _, idx_flip, _, _ = hrq_forward(
        p, z, value_topk=2, alpha_val=10.0,
        value_scalars=jnp.zeros(8).at[topk[0, 1]].set(1e6))
    assert int(idx_flip[0]) == int(topk[0, 1]), \
        "value rescoring did not reorder inside the top-k"

    # A huge value on a prototype outside the top-k must not pull the route out:
    # the first stage is the Poincare top-k, not a global value argmax.
    topk_set = {int(j) for j in topk[0]}
    outside = [j for j in range(8) if j not in topk_set]
    _, idx_out, _, _ = hrq_forward(
        p, z, value_topk=2, alpha_val=10.0,
        value_scalars=jnp.zeros(8).at[outside[0]].set(1e6))
    assert int(idx_out[0]) in topk_set


def test_value_scalars_receive_gradient():
    """v_j must be a live parameter, not a decoration.

    The hard forward value is the routed prototype; the soft top-k mixture is
    the backward proxy. Without that proxy ``d output / d v`` is exactly zero
    (v only changes an argmin) and the scalars never leave their initialisation.
    """
    p = _crafted_hrq(M_top=8)
    z = jax.random.normal(jax.random.PRNGKey(3), (2, 8)) * 0.3

    def loss(v):
        o = hrq_forward(p, z, value_scalars=v, alpha_val=1.0)[0]
        return jnp.sum(o ** 2)

    g = jax.grad(loss)(jnp.zeros(8))
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0), "value_scalars got no gradient from HRQ routing"


def test_top_codebook_receives_gradient():
    """R1.3: the top-k proxy must give ``C_top`` a real gradient, not only v."""
    p = _crafted_hrq(M_top=8)
    z = jax.random.normal(jax.random.PRNGKey(6), (2, 8)) * 0.3

    def loss(A):
        params = {'top': {'A': A, 'W': p['top']['W']}, 'fine': p['fine']}
        o = hrq_forward(params, z, value_scalars=jnp.zeros(8),
                        alpha_val=1.0)[0]
        return jnp.sum(o ** 2)

    g = jax.grad(loss)(p['top']['A'])
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0), "C_top got no gradient through the top-k proxy"


def test_public_entry_guards():
    p = _crafted_hrq(M_top=4)
    z = jax.random.normal(jax.random.PRNGKey(7), (2, 8)) * 0.3
    with pytest.raises(ValueError, match="value_topk"):
        hrq_forward(p, z, value_topk=0)
    with pytest.raises(ValueError, match="tau_proxy"):
        hrq_forward(p, z, tau_proxy=0.0)
    with pytest.raises(ValueError, match="value_scalars"):
        hrq_forward(p, z, value_scalars=jnp.zeros(3), alpha_val=0.1)


def test_hard_forward_is_bit_exact():
    """The STE's forward correction is exactly zero: output == raw hard route.

    ``o_soft + sg(c_top - o_soft)`` is only exact up to one ulp. The
    implemented form ``sg(c_top) + (o_soft - sg(o_soft))`` adds a literal zero
    in the forward pass, so the result must match the hard-route reference
    bit-for-bit while the backward path is the soft mixture.
    """
    params = _two_proto_hrq([0.60, 0.0], [0.20, 0.0])
    z = jnp.array([[0.5, 0.0]])
    o, _, _, _ = hrq_forward(params, z, tau_fallback=0.1)

    z_P, C_P, sims = _sims(params, z)
    c_top = C_P[sims.argmin(axis=-1)]
    r = mobius_add(z_P, -c_top)
    for fb in params['fine']:
        C_fb_P = exp_map(fb['A'] @ fb['W'])
        fi = poincare_similarity(
            r[:, None, :], C_fb_P[None, :, :]).squeeze(-1).argmin(axis=-1)
        r = mobius_add(r, -C_fb_P[fi])
    ref = log_map(mobius_add(c_top, r))
    assert jnp.array_equal(o, ref), (
        "hard forward is not the raw prototype route; max abs diff "
        f"{jnp.max(jnp.abs(o - ref))}")


def _two_proto_hrq(p0, p1, d=2):
    return {
        'top': {'A': jnp.stack([jnp.asarray(p0), jnp.asarray(p1)]),
                'W': jnp.eye(d)},
        'fine': [init_simvq(jax.random.PRNGKey(7), 1, d)],
    }


def _branch_top_term(params, z, tau_fallback):
    """Compare the returned top quantisation target against both branches."""
    z_P, C_P, sims = _sims(params, z)
    weights = jax.nn.softmax(-sims, axis=-1)  # M_top == 2 here, so top-k == all
    c_top_w = jnp.einsum('bm,md->bd', weights, C_P)
    idx = sims.argmin(axis=-1)
    c_top = C_P[idx]
    _, _, _, pairs = hrq_forward(params, z, tau_fallback=tau_fallback)
    return (pairs['quant'][0], log_map(c_top), log_map(c_top_w),
            float(_relative_gap(sims)[0]))


def test_relative_gap_selects_fallback():
    """Near-identical prototypes → fallback; separated ones → hard route.

    The criterion is relative, ``(sim_2 - sim_1) / (sim_1 + eps)``: the old
    absolute gap ``sim_2 - sim_1 < 0.1`` is true almost always at init scale and
    makes the hard route dead code.
    """
    z = jnp.array([[0.5, 0.0]])

    near = _two_proto_hrq([0.60, 0.0], [0.59, 0.0])
    got, hard, soft, gap = _branch_top_term(near, z, tau_fallback=0.1)
    assert gap < 0.1, f"setup must be a relative near-tie, gap={gap}"
    assert jnp.allclose(got, soft, atol=1e-5), \
        "near-tie must take the top-k softmax fallback"
    assert not jnp.allclose(got, hard, atol=1e-4)

    far = _two_proto_hrq([0.60, 0.0], [0.20, 0.0])
    got, hard, soft, gap = _branch_top_term(far, z, tau_fallback=0.1)
    assert gap > 0.1, f"setup must be a relative non-tie, gap={gap}"
    assert jnp.allclose(got, hard, atol=1e-5), \
        "well-separated prototypes must take the hard route"
    assert not jnp.allclose(got, soft, atol=1e-3)

    # A relative threshold scales with the similarity magnitude: the near pair
    # is a tie at tau=0.1 but not at tau=1e-6.
    got, hard, _, _ = _branch_top_term(near, z, tau_fallback=1e-6)
    assert jnp.allclose(got, hard, atol=1e-5)


def test_fallback_blends_topk_only_when_more_prototypes_exist():
    """M_top > value_topk: the fallback must not pull in far prototypes.

    The old fallback softmaxed over every prototype. With M_top=512 and a
    value_topk of 4, that lets far codes contribute weight they never earned
    in the geometric first stage.
    """
    params = {
        'top': {'A': jnp.array([[0.60, 0.0], [0.59, 0.0],
                                [0.00, 0.40], [0.20, 0.30]]),
                'W': jnp.eye(2)},
        'fine': [init_simvq(jax.random.PRNGKey(7), 1, 2)],
    }
    z = jnp.array([[0.5, 0.0]])
    _, idx, _, pairs = hrq_forward(params, z, tau_fallback=0.1, value_topk=2)
    _, C_P, sims = _sims(params, z)

    assert float(_relative_gap(sims)[0]) < 0.1
    topk = jnp.argsort(sims, axis=-1)[:, :2]
    assert {int(j) for j in topk[0]} == {0, 1}
    assert int(idx[0]) in (0, 1)

    topk_sims = jnp.take_along_axis(sims, topk, axis=-1)
    c_top_w = jnp.einsum(
        'bk,bkd->bd', jax.nn.softmax(-topk_sims, axis=-1), C_P[topk])
    assert jnp.allclose(pairs['quant'][0], log_map(c_top_w), atol=1e-5), \
        "fallback did not restrict its softmax to the top-k"

    # A blend over all four prototypes is meaningfully different here, so the
    # equality above is a real constraint rather than a near-tie.
    c_all = jnp.einsum('bm,md->bd', jax.nn.softmax(-sims, axis=-1), C_P)
    assert not jnp.allclose(log_map(c_top_w), log_map(c_all), atol=1e-2)


def test_top_sim_is_per_sample():
    """``top_sim`` stays per-sample (B,); the sparse LFQ threshold is not a batch mean."""
    p = _crafted_hrq()
    z = jnp.stack([jax.random.normal(jax.random.PRNGKey(4), (8,)) * 0.3,
                   jax.random.normal(jax.random.PRNGKey(5), (8,)) * 0.3])
    _, _, top_sim, _ = hrq_forward(p, z)
    assert top_sim.shape == (2,)
    assert not jnp.allclose(top_sim[0], top_sim[1])
