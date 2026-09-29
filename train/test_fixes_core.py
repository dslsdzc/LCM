"""Core training-loop + Python-C bridge regression tests (B-group fixes).

Covers:
  1. --use-qwen active channel actually trains (a_loss > 0, z_proj gradient
     nonzero, frozen qwen untouched by the optimizer).
  2. Passive channel reads out inputs[:, ctx_len] — the token right after the
     context segment z was computed from. The batch's `targets` feeds the
     active channel's generation segment only (context/generation split).
  4. Supervisor.step no longer raises "multiple values for argument 'lr'"
     and rolls back to the BEST params on spikes.
  7. load_decoder handles a bare d×V decoder.bin (w_proj = identity).
 13. load_codebook_for_c raises ValueError on CRC mismatch.
 14. _check_lcm_dim raises RuntimeError when the C engine's LCM_D ≠ d_model.

Run from repo root:  be/bin/python -m train.test_fixes_core
"""
import functools
import os
import struct
import sys
import tempfile
import zlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from train.config import LCMConfig
from train.cog_train import init_cog_params, make_train_step
from train.train_supervisor import Supervisor
from train.checkpoint import _pack_header, _compute_checksum
import lcm as lcm_mod


def _small_cfg():
    """Small config so CPU smoke tests compile/run fast."""
    return LCMConfig(
        d_model=64, vocab_size=64, max_seq_len=16, d_ff=96,
        n_heads=4, n_encoder_layers=1,
        M_top=16, M_fine=8, n_hrq_layers=1,
        M_sparse=16, M_lr=16, M_man=16, M_bind=16, M_contrast=16,
        n_bind_layers=1, n_contrast_layers=1, n_self_codes=8,
        max_inference_steps=8, ranks=(2, 4),
        # r_max must equal max(ranks): binding_forward projects the shared
        # low-rank base V (last dim max(ranks)) through params['A_k'], shaped
        # (r_max, ...). This config had r_max defaulting to 8 against
        # ranks=(2, 4), which was unreachable while the cognitive loop ran the
        # generic codebook path and became a dot_general shape error the moment
        # it started calling the real binding_forward.
        r_max=4,
        use_qwen=True,
    )


@functools.lru_cache(maxsize=1)
def _shared_train_step():
    """One shared jitted train_step for all train-loop tests (one compile)."""
    cfg = _small_cfg()
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-4, weight_decay=0.0),  # wd=0 → zero grad
    )                                                       # keeps z_proj put
    return cfg, opt, make_train_step(cfg, opt)


def _fake_qwen():
    """4-layer fake Qwen2.5-0.5B (d=896), dims mirror test_causal_mask.

    Carries the q/k/v biases real Qwen2 has. Without them this fixture does not
    describe Qwen2, `qwen_attn` refuses it, and a test that passed anyway would
    be testing a model the checkpoint loader can never produce.
    """
    d = 896
    vocab = 64
    n_kv = 2 * 64
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, 48)
    p = {
        'model.embed_tokens.weight': jax.random.normal(keys[0], (vocab, d)) * 0.02,
        'model.norm.weight': jnp.ones(d),
    }
    ki = 1
    for li in range(4):
        p[f'model.layers.{li}.input_layernorm.weight'] = jnp.ones(d)
        p[f'model.layers.{li}.post_attention_layernorm.weight'] = jnp.ones(d)
        p[f'model.layers.{li}.self_attn.q_proj.weight'] = jax.random.normal(keys[ki], (d, d)) * 0.02; ki += 1
        p[f'model.layers.{li}.self_attn.q_proj.bias'] = jax.random.normal(keys[ki], (d,)) * 0.02; ki += 1
        p[f'model.layers.{li}.self_attn.k_proj.weight'] = jax.random.normal(keys[ki], (n_kv, d)) * 0.02; ki += 1
        p[f'model.layers.{li}.self_attn.k_proj.bias'] = jax.random.normal(keys[ki], (n_kv,)) * 0.02; ki += 1
        p[f'model.layers.{li}.self_attn.v_proj.weight'] = jax.random.normal(keys[ki], (n_kv, d)) * 0.02; ki += 1
        p[f'model.layers.{li}.self_attn.v_proj.bias'] = jax.random.normal(keys[ki], (n_kv,)) * 0.02; ki += 1
        p[f'model.layers.{li}.self_attn.o_proj.weight'] = jax.random.normal(keys[ki], (d, d)) * 0.02; ki += 1
        p[f'model.layers.{li}.mlp.gate_proj.weight'] = jax.random.normal(keys[ki], (2048, d)) * 0.02; ki += 1
        p[f'model.layers.{li}.mlp.up_proj.weight'] = jax.random.normal(keys[ki], (2048, d)) * 0.02; ki += 1
        p[f'model.layers.{li}.mlp.down_proj.weight'] = jax.random.normal(keys[ki], (d, 2048)) * 0.02; ki += 1
    return p


def _run_steps(train_step, cfg, params, self_state, batch, n_steps=3, seed=123):
    """Run n_steps with a fixed rng sequence; return (losses, params, self_state)."""
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-4, weight_decay=0.0))
    opt_state = opt.init({k: v for k, v in params.items() if k != 'qwen'})
    rng = jax.random.PRNGKey(seed)
    losses = []
    for _ in range(n_steps):
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss, aux = train_step(
            params, opt_state, batch, 3e-4, step_rng, self_state=self_state)
        if aux.get('self_state') is not None:
            self_state = aux['self_state']
        losses.append(float(loss))
    return losses, params, self_state


# ── #1: --use-qwen active channel must actually train ─────────────────────────

def test_qwen_active_channel_trains():
    cfg, opt, train_step = _shared_train_step()
    d = cfg.d_model
    batch = (jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]]),
             jnp.array([[2, 3, 4, 5, 6, 7, 8, 9]]))

    # Run A: with frozen fake Qwen + trainable z_proj
    params_q, self_state_q = init_cog_params(
        cfg, jax.random.split(jax.random.PRNGKey(0))[1], qwen_ckpt=None)
    params_q['qwen'] = _fake_qwen()
    params_q['z_proj'] = jax.random.normal(
        jax.random.PRNGKey(7), (896, d)) * (d ** -0.5)
    z_proj_before = params_q['z_proj']
    qwen_before = params_q['qwen']
    losses_q, params_q, _ = _run_steps(train_step, cfg, params_q, self_state_q, batch)

    # Run B: same everything, but active channel disabled (a_loss ≡ 0)
    params_n, self_state_n = init_cog_params(
        cfg, jax.random.split(jax.random.PRNGKey(0))[1], qwen_ckpt=None)
    losses_n, _, _ = _run_steps(train_step, cfg, params_n, self_state_n, batch)

    # a_loss > 0: qwen run must be strictly worse (CE ≈ log(64) ≈ 4.16)
    assert losses_q[0] > losses_n[0] + 1.0, \
        f"active channel contributes nothing (qwen={losses_q[0]:.4f}, none={losses_n[0]:.4f})"
    # all steps finite and sane
    assert all(np.isfinite(l) and l > 0 for l in losses_q), f"losses: {losses_q}"

    # z_proj gradient is nonzero (wd=0 → zero-grad would leave it untouched)
    assert not np.allclose(np.asarray(params_q['z_proj']), np.asarray(z_proj_before)), \
        "z_proj did not move — no gradient flows through the active channel"

    # frozen Qwen is byte-identical after 3 optimizer steps (incl. no decay)
    for k in qwen_before:
        assert np.allclose(np.asarray(params_q['qwen'][k]),
                           np.asarray(qwen_before[k])), \
            f"frozen qwen leaf {k} was modified by the optimizer"
    print(f"  #1 OK: a_loss present (qwen={losses_q[0]:.4f} vs none={losses_n[0]:.4f}), "
          f"z_proj moved, qwen pristine")


# ── #2: passive channel must predict the true next token ─────────────────────

def test_passive_target_is_next_token():
    """Passive channel reads out inputs[:, ctx_len] — the token after the context.

    Under the context/generation split, z comes from inputs[:, :ctx_len] and
    both channels are supervised on what follows. The passive readout target is
    therefore a token *inside* the batch's `inputs`, and the `targets` tensor
    carries the generation-segment supervision consumed by the active channel
    only. (Reading targets[:, -1] instead asked the converged state to jump
    ctx_len tokens ahead; it leaked nothing, but it was no longer the same
    "next step" the active channel is trained on.)
    """
    cfg, opt, train_step = _shared_train_step()
    params, self_state = init_cog_params(
        cfg, jax.random.split(jax.random.PRNGKey(1))[1], qwen_ckpt=None)

    inputs = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    ctx_len = max(1, min(int(inputs.shape[1] * cfg.cog_context_frac),
                         inputs.shape[1] - 1))
    t1 = jnp.array([[2, 3, 4, 5, 6, 7, 8, 9]])
    t2 = jnp.array([[2, 3, 4, 5, 6, 7, 8, 42]])   # ONLY the last target differs
    i2 = inputs.at[0, ctx_len].set(9)             # ONLY the readout token differs

    opt_state = opt.init({k: v for k, v in params.items() if k != 'qwen'})
    # SAME rng for all calls: the self-lattice uses rng (Gumbel-Softmax), so a
    # differing rng would muddy the discriminator.
    rng = jax.random.PRNGKey(5)
    _, _, loss_a, _ = train_step(params, opt_state, (inputs, t1), 3e-4, rng,
                                 self_state=self_state)
    _, _, loss_b, _ = train_step(params, opt_state, (inputs, t2), 3e-4, rng,
                                 self_state=self_state)
    _, _, loss_c, _ = train_step(params, opt_state, (i2, t1), 3e-4, rng,
                                 self_state=self_state)
    # The passive readout target must move the loss...
    assert abs(float(loss_a) - float(loss_c)) > 1e-4, \
        f"loss insensitive to inputs[:, {ctx_len}] (the passive readout target)"
    # ...and with the active channel disabled, `targets` is not read at all.
    assert abs(float(loss_a) - float(loss_b)) < 1e-6, \
        f"targets changed the loss with the active channel disabled " \
        f"({float(loss_a):.6f} vs {float(loss_b):.6f}); only the active " \
        f"channel consumes them"

    # With the active channel present, the generation-segment targets DO matter.
    params_q, self_state_q = init_cog_params(
        cfg, jax.random.split(jax.random.PRNGKey(1))[1], qwen_ckpt=None)
    params_q['qwen'] = _fake_qwen()
    params_q['z_proj'] = jax.random.normal(
        jax.random.PRNGKey(7), (896, cfg.d_model)) * (cfg.d_model ** -0.5)
    opt_state_q = opt.init({k: v for k, v in params_q.items() if k != 'qwen'})
    _, _, q1, _ = train_step(params_q, opt_state_q, (inputs, t1), 3e-4, rng,
                             self_state=self_state_q)
    _, _, q2, _ = train_step(params_q, opt_state_q, (inputs, t2), 3e-4, rng,
                             self_state=self_state_q)
    assert abs(float(q1) - float(q2)) > 1e-4, \
        f"active channel insensitive to the generation-segment targets " \
        f"({float(q1):.6f} vs {float(q2):.6f})"

    # 2-step smoke: finite, non-degenerate loss
    losses, _, _ = _run_steps(train_step, cfg, params, self_state,
                              (inputs, t1), n_steps=2, seed=99)
    assert all(np.isfinite(l) and l > 0.1 for l in losses), f"losses: {losses}"
    print(f"  #2 OK: readout-token sensitivity {float(loss_a):.4f} vs "
          f"{float(loss_c):.4f}; targets inert without active channel "
          f"({float(loss_a):.6f} == {float(loss_b):.6f}); active-channel "
          f"target sensitivity {float(q1):.4f} vs {float(q2):.4f}; "
          f"2-step smoke {[f'{l:.3f}' for l in losses]}")


# ── #4: Supervisor.step signature / rollback-to-best ──────────────────────────

def test_supervisor_step_signature_and_rollback():
    tmp = tempfile.mkdtemp(prefix="lcm_sup_")
    cfg = LCMConfig()
    sup = Supervisor(tmp, cfg, enable_auto=True, patience=2, lr_decay=0.5)
    batch = (jnp.zeros((1, 4), dtype=jnp.int32), jnp.ones((1, 4), dtype=jnp.int32))
    p0 = {'a': jnp.ones(2)}
    o0 = {'s': jnp.zeros(2)}
    calls = []
    losses = iter([1.0, 4.0, 9.0, 27.0])

    def fake_train_fn(params, opt_state, batch, lr, rng, self_state=None):
        calls.append((float(lr), rng, self_state))
        return params, opt_state, jnp.array(next(losses)), {'self_state': None}

    # No TypeError: lr must land in the lr slot, rng in the rng slot
    p, o, loss, aux = sup.step(fake_train_fn, p0, o0, batch, 0.001,
                               jax.random.PRNGKey(0), step=0, self_state='ss')
    assert calls[0][0] == 0.001 and calls[0][2] == 'ss', f"args misrouted: {calls[0]}"
    assert float(loss) == 1.0
    assert sup.best_loss == 1.0

    # Step 2: loss 4.0 > 3×best → spike streak 1
    sup.step(fake_train_fn, p0, o0, batch, 0.001, jax.random.PRNGKey(1), step=1)
    # Step 3: loss 9.0 → streak 2 ≥ patience → rollback to BEST params
    p_rb, o_rb, loss_rb, aux_rb = sup.step(
        fake_train_fn, p0, o0, batch, 0.001, jax.random.PRNGKey(2), step=2)

    # The rollback restores the BEST params. Identity no longer holds: the
    # frozen bridge is deliberately excluded from the snapshot (it is ~2 GB and
    # never changes) and has to be re-attached before the step can run, so the
    # returned dict is a rebuild rather than the snapshot object itself.
    got = {k: v for k, v in p_rb.items() if k != 'qwen'}
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda a, b: jnp.allclose(a, b), got, sup.best_params)), \
        "rollback must return the BEST params, not the degraded current ones"

    # ...and it must actually RETRY at the reduced LR. The old version returned
    # a fabricated 1e10 and never re-ran the step; the retry consumes the next
    # loss from the iterator (27.0) instead.
    assert aux_rb.get('rollback') is True
    assert float(loss_rb) == 27.0, f"rollback must retry the step, got {loss_rb}"
    assert sup.lr_scale == 0.5, f"lr_scale not decayed: {sup.lr_scale}"
    assert calls[-1][0] == 0.0005, \
        f"retry must run at the reduced LR, got {calls[-1][0]}"
    print("  #4 OK: step() forwards lr/rng/self_state; spike rollback restores "
          "best params and retries at the reduced LR")


# ── #7: bare d×V decoder.bin (cog checkpoint) ────────────────────────────────

def test_decoder_bin_bare_wout():
    d, V = 8, 16
    tmp = tempfile.mkdtemp(prefix="lcm_dec_")
    w = np.random.randn(d, V).astype(np.float32)
    w.tofile(os.path.join(tmp, "decoder.bin"))
    dec = lcm_mod.load_decoder(tmp, d, V)
    # Sweep-3: bare W_out now loads as 'cog' — linear readout (no ELU the
    # training loss never saw).
    assert dec['format'] == 'cog', f"format={dec['format']}, expected 'cog'"
    assert dec['w_out'].shape == (d, V)
    assert np.allclose(dec['w_out'], w)
    print("  #7 OK: bare d×V decoder.bin → 'cog' linear readout")


# ── #13: CRC mismatch raises in load_codebook_for_c ──────────────────────────

def test_crc_mismatch_raises():
    tmp = tempfile.mkdtemp(prefix="lcm_crc_")
    M, d = 4, 8
    data = np.random.randn(M * d).astype(np.float32).tobytes()

    # Corrupt file: header CRC = 0, data ≠ empty → mismatch
    bad = os.path.join(tmp, "foo_codebook.bin")
    with open(bad, 'wb') as f:
        f.write(_pack_header(M, d, 1, 1, 1.0))
        f.write(data)
    try:
        lcm_mod.load_codebook_for_c(tmp, "foo_codebook.bin")
        raise AssertionError("expected ValueError on CRC mismatch")
    except ValueError as e:
        assert 'CRC mismatch' in str(e) and 'foo_codebook.bin' in str(e), str(e)

    # Valid file: same layout, correct CRC → loads fine
    good = os.path.join(tmp, "bar_codebook.bin")
    with open(good, 'wb') as f:
        f.write(struct.pack('<iiiifI', M, d, 1, 1, 1.0, _compute_checksum(data)))
        f.write(data)
    arr, M2, d2 = lcm_mod.load_codebook_for_c(tmp, "bar_codebook.bin")
    assert (M2, d2) == (M, d) and arr.shape == (M * d,)
    print("  #13 OK: corrupt CRC raises ValueError; valid file loads")


class _FakeSpec:
    """Duck-typed stand-in for TokenizerSpec — this test only needs the path
    that gets copied into the checkpoint and the extent it claims."""

    def __init__(self, tokenizer_id, path, sha256, model_vocab_size):
        self.tokenizer_id = tokenizer_id
        self.path = path
        self.sha256 = sha256
        self.model_vocab_size = model_vocab_size


def _fake_run_identity(tmp, vocab_size):
    """A RunIdentity whose extent matches the params being exported."""
    from train.cog_train import RunIdentity
    from train.tokenizer_spec import sha256_file

    tok = os.path.join(tmp, "source_tokenizer.json")
    with open(tok, "w") as f:
        f.write("{}")
    return RunIdentity(
        _FakeSpec("fake", tok, sha256_file(tok), vocab_size),
        {"model_vocab_size": vocab_size, "token_data_sha256": "aa",
         "document_spans_sha256": "bb", "n_tokens": 1, "n_docs": 1},
    )


# ── #9/#13: exported codebook .bin files round-trip through lcm.py ────────────

def test_export_roundtrip_through_lcm():
    """save_cog_checkpoint output must parse with 24-byte headers + valid CRC:
    gvalue (M=4), sparse (was written with a shadowed 36-byte writer), danger.
    """
    from train.cog_train import save_cog_checkpoint

    cfg = _small_cfg()
    params, self_state = init_cog_params(
        cfg, jax.random.split(jax.random.PRNGKey(3))[1], qwen_ckpt=None)
    out = tempfile.mkdtemp(prefix="lcm_export_")
    save_cog_checkpoint(params, out, 1, self_state=self_state,
                        run_identity=_fake_run_identity(out, cfg.vocab_size),
                        train_cfg=cfg)

    for fname in ["hrq_codebook.bin", "sparse_codebook.bin",
                  "manifold_codebook.bin", "gvalue_codebook.bin",
                  "danger_codebook.bin"]:
        data, M, d = lcm_mod.load_codebook_for_c(out, fname)  # raises on bad CRC
        assert d == cfg.d_model, f"{fname}: d={d}"

    gv_data, gv_n, _ = lcm_mod.load_codebook_for_c(out, "gvalue_codebook.bin")
    assert gv_n == 4, f"gvalue M parsed as {gv_n} (must be 4)"
    assert len(gv_data) == 2 * gv_n * cfg.d_model

    danger_data, danger_M, _ = lcm_mod.load_codebook_for_c(out, "danger_codebook.bin")
    assert len(danger_data) >= 2 * danger_M * cfg.d_model, \
        "danger file must carry threats + normals halves"
    print("  #9/#13 round-trip OK: all exported .bin files parse, CRC valid, "
          f"gv_n={gv_n}, danger halves=2")


# ── kv layout: incremental encoder must match the full batch encode ───────────

def _random_encoder(d=32, V=64, n_heads=4, n_layers=2, d_ff=48, seed=0):
    """Random numpy encoder params in the layout _encoder_full_with_state needs."""
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(n_layers):
        layers.append({
            'ln1_scale': rng.uniform(0.5, 1.5, d).astype(np.float32),
            'ln1_bias': rng.normal(0, 0.1, d).astype(np.float32),
            'w_q': rng.normal(0, 0.1, (d, d)).astype(np.float32),
            'w_k': rng.normal(0, 0.1, (d, d)).astype(np.float32),
            'w_v': rng.normal(0, 0.1, (d, d)).astype(np.float32),
            'w_o': rng.normal(0, 0.1, (d, d)).astype(np.float32),
            'ln2_scale': rng.uniform(0.5, 1.5, d).astype(np.float32),
            'ln2_bias': rng.normal(0, 0.1, d).astype(np.float32),
            'w_1': rng.normal(0, 0.1, (d, d_ff)).astype(np.float32),
            'w_2': rng.normal(0, 0.1, (d, d_ff)).astype(np.float32),
            'w_3': rng.normal(0, 0.1, (d_ff, d)).astype(np.float32),
        })
    return {
        'embed': rng.normal(0, 0.1, (V, d)).astype(np.float32),
        'rel_bias': np.zeros(2 * 32 - 1, dtype=np.float32),
        'layers': layers,
        'q_pool': rng.normal(0, 0.1, d).astype(np.float32),
        'w_proj': rng.normal(0, 0.1, (d, d)).astype(np.float32),
    }


def _ref_causal_encoder(enc, tokens, n_heads):
    """Reference causal incremental encoder with per-head (H, d_h, d_h) state.

    This is the algorithm spec: the batch encoder (_encoder_full_with_state)
    is BIDIRECTIONAL (every position sees all tokens), while the incremental
    encoder is CAUSAL by design — so it must be checked against this
    per-token reference, not against the batch encoder.
    """
    d = enc['q_pool'].shape[0]
    n_layers = len(enc['layers'])
    d_h = d // n_heads
    state = {
        'embed': enc['embed'],
        'layers': [{'kv': np.zeros((n_heads, d_h, d_h), dtype=np.float32),
                    'k': np.zeros((n_heads, d_h), dtype=np.float32)}
                   for _ in range(n_layers)],
        'pool_kv': np.zeros((d, d), dtype=np.float32),
        'pool_k': np.zeros(d, dtype=np.float32),
        'q_pool': enc['q_pool'],
        'w_proj': enc['w_proj'],
    }
    h = np.zeros(d, dtype=np.float32)
    for t in tokens:
        h = enc['embed'][t]
        for l, layer in enumerate(enc['layers']):
            ls = state['layers'][l]
            h_norm = lcm_mod._layer_norm(h, layer['ln1_scale'], layer['ln1_bias'])
            q = lcm_mod._elu_plus_one(h_norm @ layer['w_q']).reshape(n_heads, d_h)
            k = lcm_mod._elu_plus_one(h_norm @ layer['w_k']).reshape(n_heads, d_h)
            v = (h_norm @ layer['w_v']).reshape(n_heads, d_h)
            ls['kv'] += np.einsum('hd,he->hde', k, v)
            ls['k'] += k
            num = np.einsum('hd,hde->he', q, ls['kv'])
            den = np.einsum('hd,hd->h', q, ls['k'])
            attn_out = (num / (den[:, None] + 1e-6)).reshape(d) @ layer['w_o']
            h = h + attn_out
            h = h + lcm_mod._glu(
                lcm_mod._layer_norm(h, layer['ln2_scale'], layer['ln2_bias']),
                layer['w_1'], layer['w_2'], layer['w_3'])
        k_pool = lcm_mod._elu_plus_one(h)
        state['pool_kv'] += np.outer(k_pool, h)
        state['pool_k'] += k_pool
    q_pool = lcm_mod._elu_plus_one(state['q_pool'][None, :])[0]
    z = (q_pool @ state['pool_kv']) / (q_pool @ state['pool_k'] + 1e-6)
    return z @ state['w_proj']


def test_incremental_encoder_matches_full_encode():
    """Regression for the kv-layout bug: the recurrent state must live in the
    Cython (d, d) block layout — passing (H, d_h, d_h) to the boundscheck-free
    pyx function is undefined behaviour (out-of-bounds reads)."""
    d, n_heads, n_layers = 32, 4, 2
    enc = _random_encoder(d=d, n_heads=n_heads, n_layers=n_layers)
    tokens = np.array([3, 7, 1, 5, 9, 2, 8, 4], dtype=np.int32)

    z_ref = _ref_causal_encoder(enc, tokens, n_heads)

    # Pure-Python fallback path (block layout)
    _, st_py = lcm_mod._encoder_full_with_state(enc, tokens[:1], n_heads)
    assert st_py['layers'][0]['kv'].shape == (d, d), \
        f"state kv must be block layout (d,d), got {st_py['layers'][0]['kv'].shape}"
    for t in tokens[1:]:
        z_py = lcm_mod._encoder_recurrent_step(st_py, int(t), enc['layers'], n_heads)
    assert np.allclose(z_py, z_ref, atol=1e-3, rtol=1e-3), \
        f"pure-python incremental vs reference mismatch: max diff {np.max(np.abs(z_py - z_ref)):.2e}"

    # Cython path (when the .so is importable)
    cy_ok = False
    try:
        import importlib
        importlib.import_module('train._lcm_cy')
        cy_ok = True
    except ImportError:
        pass
    if cy_ok:
        _, st_cy = lcm_mod._encoder_full_with_state(enc, tokens[:1], n_heads)
        for t in tokens[1:]:
            z_cy = lcm_mod.encoder_recurrent_step_cy(enc, st_cy, int(t), n_heads)
        assert np.allclose(z_cy, z_ref, atol=1e-3, rtol=1e-3), \
            f"cython incremental vs reference mismatch: max diff {np.max(np.abs(z_cy - z_ref)):.2e}"
    print(f"  kv-layout OK: cy+py incremental == per-head reference (cy={'yes' if cy_ok else 'no'})")


# ── #14: LCM_D runtime validation ─────────────────────────────────────────────

def test_lcm_dim_mismatch_raises():
    class StubLib:
        def __init__(self, dim):
            self._dim = dim

        def lcm_dim(self):
            return self._dim

    class NoDimLib:
        pass

    try:
        lcm_mod._check_lcm_dim(StubLib(128), 256)
        raise AssertionError("expected RuntimeError on LCM_D mismatch")
    except RuntimeError as e:
        assert 'LCM_D=128' in str(e) and '256' in str(e), str(e)
    lcm_mod._check_lcm_dim(StubLib(256), 256)  # match → no raise
    lcm_mod._check_lcm_dim(NoDimLib(), 256)    # old .so → tolerated
    print("  #14 OK: LCM_D mismatch raises RuntimeError; match/absent tolerated")


if __name__ == '__main__':
    print("[test_fixes_core]")
    test_qwen_active_channel_trains()
    test_passive_target_is_next_token()
    test_supervisor_step_signature_and_rollback()
    test_decoder_bin_bare_wout()
    test_crc_mismatch_raises()
    test_export_roundtrip_through_lcm()
    test_incremental_encoder_matches_full_encode()
    test_lcm_dim_mismatch_raises()
    print("All core fixes tests passed.")
