"""Causal mask tests — masks must block future tokens (issue #1).

The reported bug: `_causal_mask` was inverted — `jnp.tril(fill(-1e9), k=0)`
keeps the fill on the LOWER triangle (past + self) and zeros the upper
triangle (future), so earlier tokens could attend to later tokens. That
made training PPL collapse (~1.x, model copies the future token) while
generation stayed garbage.

Run from repo root:  python -m train.test_causal_mask
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from train.lang_lcm import _causal_mask, _softmax_attention
from train.qwen_lm import qwen_forward


def test_lang_lcm_causal_mask_values():
    """Mask contract: allowed (0) for j <= i, blocked (-1e9) for j > i."""
    N = 4
    m = _causal_mask(N)
    for i in range(N):
        for j in range(N):
            if j <= i:
                assert m[i, j] == 0.0, \
                    f"past/self position j={j} must be visible from i={i}"
            else:
                assert m[i, j] == -1e9, \
                    f"future position j={j} must be blocked from i={i}"


def test_lang_lcm_attention_cannot_peek_future():
    """Attention output at position i must only mix positions <= i."""
    N = 4
    q = jnp.ones((1, 1, N, 8))
    k = jnp.ones((1, 1, N, 8))
    # Values unique per position → output reveals which positions were attended.
    v = jnp.arange(N, dtype=jnp.float32).reshape(1, 1, N, 1).repeat(8, axis=-1)
    mask = _causal_mask(N)[None, None, :, :]
    out = _softmax_attention(q, k, v, mask)
    # Correct mask: position i attends {0..i} uniformly → out[i] = mean(v[0..i]).
    for i in range(N):
        expected = jnp.arange(i + 1, dtype=jnp.float32).mean()
        assert jnp.allclose(out[0, 0, i, 0], expected, atol=1e-4), \
            f"position {i} attends the wrong span"


def _fake_qwen_params():
    d = 896  # Qwen2.5-0.5B d_model (d // 14 heads = 64 head_dim)
    vocab = 64
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 7)
    return {
        'model.embed_tokens.weight': jax.random.normal(keys[0], (vocab, d)) * 0.02,
        'model.norm.weight': jnp.ones(d),
        'model.layers.0.input_layernorm.weight': jnp.ones(d),
        'model.layers.0.post_attention_layernorm.weight': jnp.ones(d),
        'model.layers.0.self_attn.q_proj.weight': jax.random.normal(keys[1], (d, d)) * 0.02,
        'model.layers.0.self_attn.k_proj.weight': jax.random.normal(keys[2], (2 * 64, d)) * 0.02,
        'model.layers.0.self_attn.v_proj.weight': jax.random.normal(keys[3], (2 * 64, d)) * 0.02,
        'model.layers.0.self_attn.o_proj.weight': jax.random.normal(keys[4], (d, d)) * 0.02,
        'model.layers.0.mlp.gate_proj.weight': jax.random.normal(keys[5], (2048, d)) * 0.02,
        'model.layers.0.mlp.up_proj.weight': jax.random.normal(keys[6], (2048, d)) * 0.02,
        'model.layers.0.mlp.down_proj.weight': jax.random.normal(keys[5], (d, 2048)) * 0.02,
    }


def test_qwen_forward_cannot_peek_future():
    """Logits at position i must not change when the token at i+1 changes."""
    params = _fake_qwen_params()
    ids1 = jnp.array([[1, 2, 3, 4]])
    ids2 = jnp.array([[1, 2, 9, 4]])  # only position 2 differs
    logits1 = qwen_forward(params, ids1, n_layers=1)
    logits2 = qwen_forward(params, ids2, n_layers=1)

    # Position 1 must not attend position 2 (future) → identical logits.
    assert jnp.allclose(logits1[:, 1, :], logits2[:, 1, :], atol=1e-5), \
        "position 1 leaks the token at position 2 (future)"

    # Positive control: position 3 does attend position 2 → logits differ.
    assert not jnp.allclose(logits1[:, 3, :], logits2[:, 3, :], atol=1e-5), \
        "positive control failed: position 3 should see position 2"


if __name__ == '__main__':
    test_lang_lcm_causal_mask_values()
    test_lang_lcm_attention_cannot_peek_future()
    test_qwen_forward_cannot_peek_future()
    print('All causal mask tests passed.')
