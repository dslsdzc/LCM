"""R18/R19: layered codebook/commitment losses + routing training.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_vq_losses.py -v
"""
import jax
import jax.numpy as jnp
import pytest

from train.config import LCMConfig
from train.encoder import encoder_forward
from train.hyp import safe_unit
from train.lattices import init_route_params, route_commit_loss, routing_gate
from train.losses import (
    codebook_loss, cognitive_vq_losses, commitment_loss, hrq_vq_loss,
)


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


CFG = LCMConfig(
    d_model=16, d_ff=24, n_heads=2, d_head=8, vocab_size=32,
    max_seq_len=8, n_encoder_layers=1,
    M_top=8, M_fine=4, n_hrq_layers=1,
    M_sparse=8, M_lr=8, M_man=8, M_bind=8, M_contrast=8,
    n_bind_layers=1, n_contrast_layers=1, n_self_codes=4,
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
    """Sparse is EMA-owned: no codebook term, but commitment still moves r.

    The gradient check runs through the real step composition, so it also pins
    R18's routing: EMA codebooks get *exactly* zero gradient, pure-gradient
    codebooks get a non-zero one.
    """
    from train.cognitive_step import six_lattice_step

    params, z, aux = built
    total, parts = cognitive_vq_losses(params, aux, CFG)
    assert float(parts['sparse_codebook']) == 0.0
    assert float(parts['sparse_commitment']) > 0.0
    assert float(total) > 0.0

    def loss_fn(p, zz):
        _, a = six_lattice_step(
            zz, p, CFG, training=True, rng=jax.random.PRNGKey(1))
        return cognitive_vq_losses(p, a, CFG)[0]

    (g_params, g_z) = jax.grad(loss_fn, argnums=(0, 1))(params, z)
    assert not jnp.allclose(g_z, 0.0), \
        "commitment loss must give the encoder-side input a gradient"

    def leaf_mass(tree):
        return sum(float(jnp.sum(jnp.abs(x)))
                   for x in jax.tree_util.tree_leaves(tree))

    for path in (('sparse', 'C'), ('manifold', 'C'),
                 ('binding', 'key_cb'), ('binding', 'val_cb'),
                 ('binding', 'bind_cb')):
        node = g_params
        for key in path:
            node = node[key]
        assert leaf_mass(node) == 0.0, \
            f"EMA codebook {path} must not receive a codebook gradient"

    for path in (('hrq',), ('lowrank', 'U'), ('contrast', 'C_a'),
                 ('route', 'C_route')):
        node = g_params
        for key in path:
            node = node[key]
        assert leaf_mass(node) > 0.0, \
            f"pure-gradient codebook {path} must receive gradient"


def test_train_step_joint_false_still_runs_layered_vq_losses():
    """R18: codebook/commitment losses are the main loss, not Stage-3 extras."""
    import optax

    from train.cog_train import make_train_step
    from train.ema_state import init_ema_state
    from train.model import init_all_params
    from train.param_groups import partition_params

    cfg = LCMConfig(
        d_model=16, d_ff=24, n_heads=2, d_head=8, vocab_size=32,
        max_seq_len=8, n_encoder_layers=1,
        M_top=8, M_fine=4, n_hrq_layers=1,
        M_sparse=8, M_lr=8, M_man=8, M_bind=8, M_contrast=8,
        n_bind_layers=1, n_contrast_layers=1, n_self_codes=4,
        use_bf16=False, max_inference_steps=2,
    )
    params, _, _ = init_all_params(cfg, jax.random.PRNGKey(0))
    trainable, _, _ = partition_params(params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0),
    )
    opt_state = optimizer.init(trainable)
    ema_state = init_ema_state(params, cfg)
    step_fn = make_train_step(cfg, optimizer, joint=False)
    x = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    batch = (x, x)
    _, _, _, loss, aux = step_fn(
        params, opt_state, ema_state, batch, 1e-3,
        jax.random.PRNGKey(2), step=0)
    assert jnp.isfinite(loss)
    stage3 = aux['stage3']
    for key in ('vq_codebook', 'vq_commitment', 'vq_route_codebook',
                'vq_route_commitment'):
        assert key in stage3, f"missing {key} from joint=False step"
    assert float(stage3['vq_codebook']) >= 0.0
    assert float(stage3['vq_commitment']) > 0.0
