import jax.numpy as jnp
import pytest
import re
from pathlib import Path

from train.config import LCMConfig
from train.hyp import to_poincare, poincare_similarity, poincare_distance
from train.gvalue import make_global_value_vectors, GValueCodebook


def test_python_consistency_threshold_matches_c_header():
    """D-8: the Python default and the compiled C constant must not drift.

    The C v3 loop uses the compile-time ``LCM_CONSISTENCY_THRESHOLD``; it does
    not read ``config.json`` at runtime. This test makes a one-sided change fail
    loudly instead of silently producing different training and inference
    verdicts.
    """
    header = Path(__file__).resolve().parents[1] / "infer" / "lcm.h"
    text = header.read_text()
    match = re.search(
        r"#define\s+LCM_CONSISTENCY_THRESHOLD\s+([0-9.]+)f", text)
    assert match is not None, "LCM_CONSISTENCY_THRESHOLD missing from lcm.h"
    c_value = float(match.group(1))
    assert LCMConfig().consistency_threshold == c_value == 0.4


def test_poincare_similarity_finite_at_degenerate_norms():
    """z = 0, ‖z‖ = 1 and ‖z‖ = 2 must not produce inf/NaN.

    The denominator ``(1-‖u‖²)(1-‖v‖²)`` is 0 at the boundary and negative
    outside the ball; the max-clamp keeps the similarity finite (the value can
    be huge, which is correct for the near-boundary limit).
    """
    for scale in (0.0, 1.0, 2.0):
        z = jnp.zeros((1, 256)).at[0, 0].set(scale)
        anchor = jnp.zeros((1, 256)).at[0, 0].set(0.5)
        sim = poincare_similarity(z, anchor)
        assert bool(jnp.all(jnp.isfinite(sim))), f"scale={scale}: {sim}"
        dist = poincare_distance(z, anchor)
        assert bool(jnp.all(jnp.isfinite(dist))), f"scale={scale}: {dist}"


@pytest.mark.parametrize("d,expected", [(2, 0.125), (4, 0.0625)])
def test_r18_hrq_top_uses_unit_tangent_pointer(d, expected):
    """A fused z with ‖z‖ > 1 must use ``safe_unit(z)``, norm 1.0.

    The forward is ``z_P = exp_map(safe_unit(z))``; its tangent pointer is
    ``log_map(z_P) = safe_unit(z)``. Applying ``log_map`` a second time gives
    norm ≈ 3.804 and pulls the top codebook to the wrong radius. With a zero
    quantised row the returned commitment is ``beta * ‖cont‖² / d``, so it
    independently reveals the continuous-side norm.
    """
    from train.losses import hrq_vq_loss
    from train.hyp import safe_unit

    z = jnp.zeros((1, d)).at[0, 0].set(3.0)   # ‖z‖ = 3, outside the ball
    params = {'top': {'A': jnp.zeros((1, d)), 'W': jnp.eye(d)}, 'fine': []}
    cfg = type('C', (), {'beta_vq': 0.25})()
    aux = {'cont': [z], 'quant': [jnp.zeros((1, d))]}
    codebook, commitment = hrq_vq_loss(params, z, aux, cfg)

    # Hand-computed: safe_unit([3,0,...]) has norm 1, mean square = 1/d.
    assert jnp.allclose(codebook[0], expected, atol=1e-6)
    assert jnp.allclose(commitment[0], expected, atol=1e-6)
    implied_norm = jnp.sqrt(commitment[0] * d / 0.25)
    assert jnp.allclose(implied_norm, 1.0, atol=1e-5)
    assert not jnp.allclose(implied_norm, 3.804, atol=1e-2)


@pytest.mark.parametrize("d,expected", [(2, 0.125), (4, 0.0625)])
def test_r18_manifold_commitment_uses_unit_tangent_pointer(d, expected):
    """The manifold commitment side is ``safe_unit(z)``, not a second log_map."""
    from train.losses import cognitive_vq_losses

    z = jnp.zeros((1, d)).at[0, 0].set(3.0)   # ‖z‖ = 3
    ident = jnp.eye(d)
    M = 1
    params = {
        'hrq': {'top': {'A': jnp.zeros((M, d)), 'W': ident}, 'fine': []},
        'sparse': {'C': jnp.zeros((M, d))},
        'lowrank': {'U': [jnp.zeros((M, 1))], 'A_V': jnp.zeros((d, 1)),
                    'W_V': jnp.eye(1)},
        'manifold': {'C': jnp.zeros((M, d))},
        'binding': {
            fam: [{'A': jnp.zeros((M, d)), 'W': ident}]
            for fam in ('key_cb', 'val_cb', 'bind_cb')},
        'contrast': {
            fam: [{'A': jnp.zeros((M, d)), 'W': ident}]
            for fam in ('C_a', 'C_b')},
        'route': {'C_route': jnp.zeros((6, d))},
    }
    stats = {
        'z_in': z,
        'hrq_vq': {'cont': [z], 'quant': [jnp.zeros((1, d))]},
        'sparse_idx': jnp.array([0]),
        'lowrank_aux': {'residuals': [z], 'indices': [jnp.array([0])]},
        'man_idx': jnp.array([0]),
        'binding_residuals': {f: [z] for f in ('key', 'val', 'bind')},
        'binding_indices': {f: [jnp.array([0])]
                            for f in ('key', 'val', 'bind')},
        'contrast_residuals': {'a': [z], 'b': [z]},
        'contrast_indices': {'a': [jnp.array([0])],
                             'b': [jnp.array([0])]},
        'route_idx': jnp.array([0]),
    }
    cfg = type('C', (), {'beta_vq': 0.25, 'ranks': (1,)})()
    _, parts = cognitive_vq_losses(params, stats, cfg)

    # Hand-computed: C_man = 0, cont = safe_unit(z), beta=0.25, mean = 1/d.
    assert jnp.allclose(parts['manifold_commitment'], expected, atol=1e-6)
    # Old double-log_map value for d=2 is ~1.808; raw z is 2.25.
    assert not jnp.allclose(parts['manifold_commitment'], 1.808, atol=1e-2)
    assert not jnp.allclose(parts['manifold_commitment'], 2.25, atol=1e-3)


def _outside_ball_codebook(d=256, scale=4.0):
    """Anchors deliberately outside the Poincaré ball (norm = scale)."""
    C_pos = jnp.zeros((4, d))
    C_neg = jnp.zeros((4, d))
    for i in range(4):
        C_pos = C_pos.at[i, i].set(scale)
        C_neg = C_neg.at[i, i].set(-scale)
    return C_pos, C_neg


def test_to_poincare_keeps_points_inside_ball():
    for scale in [0.0, 0.5, 1.0, 3.0]:
        z = jnp.ones((4, 256)) * scale
        z_P = to_poincare(z)
        assert jnp.all(jnp.linalg.norm(z_P, axis=-1) < 1.0)
        assert jnp.all(jnp.isfinite(z_P))


def test_strict_three_laws_neutral_is_violation():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    z = jnp.zeros((1, 256))
    is_safe, law = gv.check_safety(z, safety_margin_relative=0.5)
    assert not bool(is_safe)
    assert int(law) >= 0


def test_margin_zero_allows_neutral():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    z = jnp.zeros((1, 256))
    is_safe, _ = gv.check_safety(z, safety_margin_relative=0.0)
    assert bool(is_safe)


def test_batch_strict_neutral_is_violation():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    z = jnp.zeros((2, 256))
    is_safe, margins, law = gv.check_safety_batch(
        z, safety_margin_relative=0.5)
    assert not bool(jnp.any(is_safe))
    assert jnp.allclose(margins, 0.0, atol=1e-6)
    assert bool(jnp.all(law >= 0))


def test_positive_input_is_safe():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    # A point near a positive anchor stays on the same side after exp_map.
    z = jnp.asarray(C_pos[0])[None, :] * 0.5
    is_safe, _ = gv.check_safety(z, safety_margin_relative=0.0)
    assert bool(is_safe)


def test_projection_flips_raw_domain_decision():
    """Only the projected domain makes a near-positive input safe at 0.5."""
    C_pos, C_neg = _outside_ball_codebook()
    gv = GValueCodebook(C_pos, C_neg)
    z = jnp.zeros((1, 256)).at[0, 0].set(0.5)

    is_safe, _ = gv.check_safety(z, safety_margin_relative=0.5)
    assert bool(is_safe)

    # The same input in the raw domain would violate the margin.
    pos_sim = poincare_similarity(z, C_pos).min()
    neg_sim = poincare_similarity(z, C_neg).min()
    raw_balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + 1e-8)
    assert float(raw_balance) > -0.5


def test_safety_margin_loss_uses_boundary_relative_slack():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)

    # The four anchor axes have components (a, b, b, b).  With margin
    # m = (a+b)/(2-a+b), choose b=0.3 and m=0.55 → a≈0.6226.  The remaining
    # norm is spread over the other axes (each < b, so the min stays b).
    m_target = 0.55
    b = 0.3
    a = (2 * m_target + m_target * b - b) / (1 + m_target)
    rest = jnp.sqrt((1 - a ** 2 - 3 * b ** 2) / 252.0)
    z_near = jnp.zeros((1, 256))
    z_near = z_near.at[0, 0].set(a)
    z_near = z_near.at[0, 1:4].set(b)
    z_near = z_near.at[0, 4:].set(rest)
    is_safe, margins, _ = gv.check_safety_batch(
        z_near, safety_margin_relative=0.5)
    assert bool(is_safe[0])
    assert 0.5 < float(margins[0]) < 0.6, f"margins={float(margins[0])}"

    loss = gv.safety_margin_loss(
        z_near, safety_margin_relative=0.5, weight=1.0)
    expected = (0.2 - (float(margins[0]) - 0.5)) ** 2
    assert abs(float(loss) - expected) < 1e-5
    assert float(loss) > 0.0

    # Clearly safe (direction e0 → margin 1.0, slack 0.5 > 0.2): zero loss.
    z_safe = jnp.zeros((1, 256)).at[0, 0].set(1.0)
    assert float(gv.safety_margin_loss(z_safe, safety_margin_relative=0.5)) == 0.0


def test_value_signal_uses_projected_domain():
    C_pos, C_neg = _outside_ball_codebook()
    gv = GValueCodebook(C_pos, C_neg)
    o = jnp.zeros(256).at[0].set(0.5)

    signal = float(gv.compute_value_signal_for_output(o))
    assert signal > 0.99, f"projected-domain signal={signal}"

    # Raw-domain distances are so large that both exponentials underflow to 0.
    d_pos = poincare_distance(o, C_pos).min()
    d_neg = poincare_distance(o, C_neg).min()
    pos_w = jnp.exp(-d_pos / 0.1)
    neg_w = jnp.exp(-d_neg / 0.1)
    raw_signal = float((pos_w - neg_w) / (pos_w + neg_w + 1e-8))
    assert abs(raw_signal) < 1e-6, f"raw-domain signal={raw_signal}"

    signal_batch = gv.compute_value_signal_batch([o[None, :]])
    assert signal_batch.shape == (1, 1)
    assert float(signal_batch[0, 0]) > 0.99
