"""Feature bank: ring buffer and dead-code replacement (R3/R21).

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_feature_bank.py -v
"""
import numpy as np
import jax.numpy as jnp

from train.feature_bank import FeatureBank


def test_dead_code_replaced_by_random():
    bank = FeatureBank(capacity=8, dim=2)
    for i in range(4):
        bank.add(jnp.array([[float(i), 0.0]]))
    C = jnp.zeros((3, 2))
    last_used = jnp.array([0, 100, 100], dtype=jnp.int32)
    C2, last2 = bank.replace_dead_sparse(C, last_used, step=1001, t_dead=1000)
    assert not jnp.array_equal(C2[0], C[0])
    assert int(last2[0]) == 1001


def test_contrast_uses_farthest_point():
    bank = FeatureBank(capacity=4, dim=2)
    for v in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]:
        bank.add(jnp.array([v]))
    C = jnp.array([[5.0, 0.0]])
    C2, _ = bank.replace_dead_contrast(C, jnp.array([0], dtype=jnp.int32),
                                       step=1001, t_dead=1000)
    # The farthest point is [-1, 0].
    assert float(C2[0, 0]) < 0.0


def test_ring_buffer_wraps_and_keeps_capacity():
    bank = FeatureBank(capacity=3, dim=2)
    assert len(bank) == 0
    for i in range(5):
        bank.add(jnp.array([[float(i), float(i)]]))
    assert len(bank) == 3
    stored = np.asarray(bank.vectors)
    np.testing.assert_allclose(stored, [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # add() accepts a (B, d) batch as well as a (d,) vector.
    bank.add(jnp.array([[9.0, 9.0], [10.0, 10.0]]))
    np.testing.assert_allclose(np.asarray(bank.vectors)[-2:],
                               [[9.0, 9.0], [10.0, 10.0]])


def test_dead_indices_uses_step_minus_last_used():
    bank = FeatureBank(capacity=8, dim=2)
    last_used = jnp.array([0, 100, 950, 1002, 1002], dtype=jnp.int32)
    dead = np.asarray(bank.dead_indices(last_used, step=1001, t_dead=1000))
    np.testing.assert_array_equal(dead, [0])
    dead = np.asarray(bank.dead_indices(last_used, step=2002, t_dead=1000))
    np.testing.assert_array_equal(dead, [0, 1, 2])


def test_sparse_replacement_cold_starts_n_m_and_last_used():
    bank = FeatureBank(capacity=4, dim=2)
    bank.add(jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    C = jnp.zeros((2, 2))
    state = {
        'N': jnp.array([5.0, 7.0]),
        'm': jnp.array([[5.0, 5.0], [7.0, 7.0]]),
        'last_used': jnp.array([0, 500], dtype=jnp.int32),
    }
    C2, s2 = bank.replace_dead_sparse(C, state, step=1001, t_dead=1000)
    # Row 0 was dead: new vector from the bank, statistics cold-started.
    assert not jnp.array_equal(C2[0], C[0])
    assert float(s2['N'][0]) == 0.0
    assert float(jnp.linalg.norm(s2['m'][0])) == 0.0
    assert int(s2['last_used'][0]) == 1001
    # Row 1 was used recently: untouched.
    np.testing.assert_allclose(np.asarray(C2[1]), np.asarray(C[1]))
    assert float(s2['N'][1]) == 7.0
    assert int(s2['last_used'][1]) == 500


def test_contrast_replacement_leaves_live_rows_alone():
    bank = FeatureBank(capacity=4, dim=2)
    for v in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]:
        bank.add(jnp.array([v]))
    C = jnp.array([[5.0, 0.0], [0.25, 0.25]])
    last = jnp.array([0, 1001], dtype=jnp.int32)
    C2, last2 = bank.replace_dead_contrast(C, last, step=1001, t_dead=1000)
    assert float(C2[0, 0]) < 0.0
    np.testing.assert_allclose(np.asarray(C2[1]), [0.25, 0.25])
    assert int(last2[0]) == 1001
    assert int(last2[1]) == 1001


def test_empty_bank_is_a_no_op():
    bank = FeatureBank(capacity=4, dim=2)
    C = jnp.array([[1.0, 2.0]])
    last = jnp.array([0], dtype=jnp.int32)
    C2, last2 = bank.replace_dead_sparse(C, last, step=1001, t_dead=1000)
    np.testing.assert_array_equal(np.asarray(C2), np.asarray(C))
    np.testing.assert_array_equal(np.asarray(last2), np.asarray(last))
