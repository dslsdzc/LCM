import jax.numpy as jnp
import numpy as np
from train.hyp import to_poincare, poincare_similarity
from train.gvalue import make_global_value_vectors, GValueCodebook

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

def test_positive_input_is_safe():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    # 取一个正锚点附近的点：exp_map 后仍在球的同一侧
    z = jnp.asarray(C_pos[0])[None, :] * 0.5
    is_safe, _ = gv.check_safety(z, safety_margin_relative=0.0)
    assert bool(is_safe)
