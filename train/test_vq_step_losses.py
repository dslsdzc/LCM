"""R18: the layered VQ losses are composed into the cognitive training loss.

Split out of ``train/test_vq_losses.py`` so the per-layer/gradient suite stays
well under the focused-test time budget (minor 12 of the Task-4 review).

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_vq_step_losses.py -v
"""
import dataclasses

import jax
import jax.numpy as jnp
import optax
import pytest

from train.config import LCMConfig


CFG = LCMConfig(
    d_model=8, d_ff=12, n_heads=2, d_head=4, vocab_size=16,
    max_seq_len=6, n_encoder_layers=1,
    M_top=6, M_fine=3, n_hrq_layers=1,
    M_sparse=6, M_lr=6, M_man=6, M_bind=6, M_contrast=6,
    n_bind_layers=1, n_contrast_layers=1, n_self_codes=3,
    use_bf16=False, max_inference_steps=2,
)


def test_train_step_joint_false_still_runs_layered_vq_losses():
    """R18: the VQ losses are added to the loss when joint=False.

    Comparing against an otherwise identical step with ``beta_vq=0`` pins the
    composition: deleting ``loss = loss + vq_loss`` makes the two losses equal
    and fails the difference assertion.
    """
    from train.cog_train import make_train_step
    from train.ema_state import init_ema_state
    from train.model import init_all_params
    from train.param_groups import partition_params

    params, _, _ = init_all_params(CFG, jax.random.PRNGKey(0))
    trainable, _, _ = partition_params(params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0),
    )
    opt_state = optimizer.init(trainable)
    ema_state = init_ema_state(params, CFG)
    x = jnp.array([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)
    batch = (x, x)
    rng = jax.random.PRNGKey(2)

    step_fn = make_train_step(CFG, optimizer, joint=False)
    _, _, _, loss, aux = step_fn(
        params, opt_state, ema_state, batch, 1e-3, rng, step=0)
    assert jnp.isfinite(loss)
    stage3 = aux['stage3']
    for key in ('vq_codebook', 'vq_commitment', 'vq_route_codebook',
                'vq_route_commitment'):
        assert key in stage3, f"missing {key} from joint=False step"
    assert float(stage3['vq_codebook']) >= 0.0
    assert float(stage3['vq_commitment']) > 0.0
    vq_total = float(stage3['vq_codebook'] + stage3['vq_commitment'])
    assert vq_total > 0.0

    zero_cfg = dataclasses.replace(CFG, beta_vq=0.0)
    step_zero = make_train_step(zero_cfg, optimizer, joint=False)
    _, _, _, loss_zero, _ = step_zero(
        params, opt_state, ema_state, batch, 1e-3, rng, step=0)
    assert float(loss) == pytest.approx(
        float(loss_zero) + vq_total, rel=1e-5, abs=1e-6)
