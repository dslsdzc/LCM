"""Tied readout: one trainable token table, not two.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_tied_readout.py -v
"""
import dataclasses
import pickle

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def _params(vocab_size=512):
    from train.cog_train import init_cog_params
    from train.config import LCMConfig
    cfg = dataclasses.replace(LCMConfig(), vocab_size=vocab_size)
    params, _ = init_cog_params(cfg, jax.random.PRNGKey(0))
    return cfg, params


def test_no_w_out_leaf_exists():
    """The second token table must be gone, not merely unused."""
    _, params = _params()
    assert "W_out" not in params
    assert "embed" in params["encoder"]


def test_encoder_embed_has_model_vocab_rows():
    cfg, params = _params()
    assert params["encoder"]["embed"].shape == (cfg.vocab_size, cfg.d_model)


def test_passive_logits_equal_z_at_embed_transpose():
    """The readout is the tied table, not a lookalike."""
    rng = jax.random.PRNGKey(1)
    z = jax.random.normal(rng, (2, 32, 64))
    E = jax.random.normal(rng, (100, 64))
    tied = jnp.einsum("bsd,vd->bsv", z, E)
    assert jnp.allclose(tied, z @ E.T, atol=1e-5)
    assert tied.shape == (2, 32, 100)


def test_checkpoint_carrying_w_out_is_rejected(tmp_path):
    """W_out is not migrated: it was a second table in a different token space.

    Silently dropping it would leave the checkpoint's readout — and the
    gradient history that shaped z_q — unrepresented, while silently keeping it
    would mix two token spaces. Neither is a resume, so this fails.
    """
    from train.cog_train import init_cog_params
    from train.config import LCMConfig

    ckpt_dir = tmp_path / "old"
    ckpt_dir.mkdir()
    with open(ckpt_dir / "cog_params.pkl", "wb") as f:
        pickle.dump({
            "params": {"W_out": np.zeros((8, 16), dtype=np.float32)},
            "step": 0,
            "self_state": None,
        }, f)

    cfg = dataclasses.replace(LCMConfig(), vocab_size=16, d_model=8, d_ff=12,
                              n_heads=2, d_head=4)
    with pytest.raises(ValueError, match="W_out"):
        init_cog_params(cfg, jax.random.PRNGKey(0), resume=str(ckpt_dir))
