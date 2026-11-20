"""Checkpoint export: identity, tied readout, and a config that is not invented.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_export.py -v
"""
import dataclasses
import inspect
import json
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")


def test_run_identity_is_required():
    """Omitting it must be a TypeError, not a silent global fallback."""
    from train.cog_train import save_cog_checkpoint
    p = inspect.signature(save_cog_checkpoint).parameters["run_identity"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty


def test_train_cfg_is_required():
    """The real training config must be passed, not reconstructed locally."""
    from train.cog_train import save_cog_checkpoint
    p = inspect.signature(save_cog_checkpoint).parameters["train_cfg"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty


def _small_params(vocab_size=512):
    from train.cog_train import init_cog_params
    from train.config import LCMConfig
    cfg = dataclasses.replace(LCMConfig(), vocab_size=vocab_size)
    params, self_state = init_cog_params(cfg, jax.random.PRNGKey(0))
    return cfg, params, self_state


@pytest.fixture
def exported(tmp_path, tiny_spec):
    from train.conftest import build_tiny_corpus
    from train.cog_train import RunIdentity, save_cog_checkpoint
    meta, _ = build_tiny_corpus(tmp_path, tiny_spec, ["北京是中国的首都" * 4])
    cfg, params, self_state = _small_params(vocab_size=tiny_spec.model_vocab_size)
    out = str(tmp_path / "ckpt")
    save_cog_checkpoint(params, out, 0,
                        run_identity=RunIdentity(tiny_spec, dict(meta)),
                        train_cfg=cfg, self_state=self_state,
                        allow_missing_buffers=True)
    return cfg, params, out, tiny_spec, meta


def test_e_has_model_vocab_rows(exported):
    cfg, params, _, tiny_spec, _ = exported
    E = np.asarray(params["encoder"]["embed"])
    assert E.shape == (tiny_spec.model_vocab_size, cfg.d_model)


def test_decoder_bin_is_the_transposed_tied_table(exported):
    cfg, params, out, tiny_spec, _ = exported
    E = np.asarray(params["encoder"]["embed"])
    decoder = np.fromfile(os.path.join(out, "decoder.bin"), dtype=np.float32)
    assert decoder.size == cfg.d_model * tiny_spec.model_vocab_size
    assert np.allclose(decoder.reshape(cfg.d_model, -1), E.T, atol=1e-6)


def test_exported_config_reports_the_tied_extent(exported):
    cfg, params, out, tiny_spec, _ = exported
    with open(os.path.join(out, "config.json")) as f:
        exported_cfg = json.load(f)
    assert exported_cfg["vocab_size"] == tiny_spec.model_vocab_size
    assert exported_cfg["d_model"] == cfg.d_model
    assert exported_cfg["format_version"] == 2


def test_manifest_covers_exported_tensors(exported):
    """Every exported .bin has a per-tensor canonical hash (R10)."""
    from train.bin_format import (
        FORMAT_VERSION, HEADER_SIZE_V2, canonical_sha256, load_manifest,
    )
    _, _, out, _, _ = exported
    manifest = load_manifest(out)
    assert manifest["format_version"] == FORMAT_VERSION
    tensors = manifest["tensors"]
    for name in ("encoder", "decoder", "hrq_codebook", "sparse_codebook",
                 "lowrank_codebook", "manifold_codebook", "bind_codebook",
                 "contrast_codebook", "gvalue_codebook", "danger_codebook"):
        assert name in tensors, f"manifest is missing {name}"
        entry = tensors[name]
        assert entry["name"] == name
        assert entry["dtype"] == "float32"
        assert len(entry["sha256"]) == 64
        assert entry["length"] == int(np.prod(entry["shape"])) * 4
    assert "params.encoder.embed" in tensors
    # Hash a real headerless payload and a real header-bearing payload.
    enc = np.fromfile(os.path.join(out, "encoder.bin"), dtype=np.float32)
    assert tensors["encoder"]["sha256"] == canonical_sha256({"encoder": enc})
    sparse = np.fromfile(os.path.join(out, "sparse_codebook.bin"),
                         dtype=np.float32, offset=HEADER_SIZE_V2)
    assert tensors["sparse_codebook"]["sha256"] == canonical_sha256(
        {"sparse_codebook": sparse})
    # canonical_offset/length locate each tensor in the canonical stream.
    offset = 0
    for entry in sorted(tensors.values(),
                        key=lambda e: e["canonical_offset"]):
        assert entry["canonical_offset"] == offset
        offset += entry["length"]
    # Ruling 9: exported payloads also map to their real file and byte offset.
    for name, filename in (
            ("encoder", "encoder.bin"),
            ("decoder", "decoder.bin"),
            ("hrq_codebook", "hrq_codebook.bin"),
            ("sparse_codebook", "sparse_codebook.bin"),
            ("lowrank_codebook", "lowrank_codebook.bin"),
            ("manifold_codebook", "manifold_codebook.bin"),
            ("bind_codebook", "bind_codebook.bin"),
            ("contrast_codebook", "contrast_codebook.bin"),
            ("gvalue_codebook", "gvalue_codebook.bin"),
            ("danger_codebook", "danger_codebook.bin"),
            ("route_codebook", "route_codebook.bin"),
            ("fusion", "fusion.bin"),
            ("value_scalars", "value_scalars.bin"),
            ("self_lattice", "self_lattice.bin"),
            ("binding_proj", "binding_proj.bin")):
        assert tensors[name]["file"] == filename
    for name in ("hrq_codebook", "sparse_codebook", "manifold_codebook",
                 "gvalue_codebook", "danger_codebook", "value_scalars"):
        assert tensors[name]["offset"] == HEADER_SIZE_V2
    for name in ("encoder", "decoder", "lowrank_codebook", "bind_codebook",
                 "contrast_codebook", "route_codebook", "fusion",
                 "self_lattice", "binding_proj"):
        assert tensors[name]["offset"] == 0


def test_exported_self_layout_matches_c_loader(exported):
    """R15: C_self rows + tag + bias, and config counts EMA modes only."""
    from train.checkpoint import SELF_LAYOUT_TAG

    cfg, params, out, _, _ = exported
    with open(os.path.join(out, "config.json")) as f:
        exported_cfg = json.load(f)
    n_modes = exported_cfg["n_self_codes"]
    assert n_modes == params["self"]["C_self"].shape[0] - 1

    flat = np.fromfile(os.path.join(out, "self_lattice.bin"),
                       dtype=np.float32)
    assert flat.size == cfg.d_model + n_modes * cfg.d_model + 2
    rows = flat[:flat.size - 2].reshape(-1, cfg.d_model)
    assert rows.shape[0] == 1 + n_modes
    np.testing.assert_allclose(rows, np.asarray(params["self"]["C_self"]),
                               atol=1e-6)
    assert flat[-2] == SELF_LAYOUT_TAG
    np.testing.assert_allclose(float(flat[-1]),
                               float(params["self"]["bias_self"]), atol=1e-7)


def test_exported_config_carries_the_real_training_config(exported):
    """Every exported setting must come from the real config or real tensors.

    n_heads drives attention reshaping at inference. The rest were hardcoded
    literals too, so assert all of them rather than only the known-bad one.
    """
    cfg, _, out, _, _ = exported
    with open(os.path.join(out, "config.json")) as f:
        exported_cfg = json.load(f)
    assert exported_cfg["n_heads"] == cfg.n_heads
    assert exported_cfg["max_seq_len"] == cfg.max_seq_len
    assert exported_cfg["n_encoder_layers"] == cfg.n_encoder_layers
    assert exported_cfg["d_ff"] == cfg.d_ff
    assert exported_cfg["n_lattices"] == cfg.n_lattices
    assert exported_cfg["n_lr_layers"] == cfg.n_lr_layers
    assert exported_cfg["r_max"] == cfg.r_max
    assert exported_cfg["t_dim"] == cfg.t_dim
    assert exported_cfg["n_value_pairs"] == cfg.n_value_pairs
    assert exported_cfg["M_danger"] == cfg.M_danger
    assert exported_cfg["max_inference_steps"] == cfg.max_inference_steps
    assert exported_cfg["convergence_tol"] == cfg.convergence_tol
    assert exported_cfg["entropy_threshold"] == cfg.entropy_threshold


def test_exported_config_M_lr_comes_from_U_rows(tmp_path, tiny_spec):
    """M_lr is U[0].shape[0] (=codebook rows), never d (A_V.shape[0])."""
    from train.conftest import build_tiny_corpus
    from train.cog_train import RunIdentity, init_cog_params, save_cog_checkpoint
    from train.config import LCMConfig

    meta, _ = build_tiny_corpus(tmp_path, tiny_spec, ["北京是中国的首都" * 4])
    cfg = dataclasses.replace(
        LCMConfig(),
        vocab_size=tiny_spec.model_vocab_size,
        d_model=16, d_ff=32, n_heads=4, n_encoder_layers=1,
        max_seq_len=16, max_inference_steps=4,
        M_top=8, M_fine=4, n_hrq_layers=1, M_sparse=8,
        M_lr=4, ranks=(2, 4), r_max=4,
        # Deliberately stale: the tensor has two U layers (ranks), not three.
        n_lr_layers=3,
        M_man=8, M_bind=8, n_bind_layers=1,
        M_contrast=8, n_contrast_layers=1, n_self_codes=4,
    )
    params, self_state = init_cog_params(cfg, jax.random.PRNGKey(0))
    assert params["lowrank"]["U"][0].shape[0] == 4
    assert params["lowrank"]["U"][0].shape[0] != cfg.d_model

    out = str(tmp_path / "ckpt")
    save_cog_checkpoint(
        params, out, 0, run_identity=RunIdentity(tiny_spec, dict(meta)),
        train_cfg=cfg, self_state=self_state, allow_missing_buffers=True)

    with open(os.path.join(out, "config.json")) as f:
        exported_cfg = json.load(f)
    assert exported_cfg["M_lr"] == params["lowrank"]["U"][0].shape[0]
    assert exported_cfg["M_lr"] == 4
    assert exported_cfg["M_lr"] != exported_cfg["d_model"]
    assert exported_cfg["n_lr_layers"] == len(params["lowrank"]["U"]) == 2


def test_checkpoint_carries_the_trained_tokenizer(exported):
    _, _, out, tiny_spec, _ = exported
    from train.tokenizer_spec import sha256_file
    copied = os.path.join(out, "tokenizer.json")
    assert os.path.exists(copied)
    assert sha256_file(copied) == tiny_spec.sha256


def test_checkpoint_carries_run_identity(exported):
    _, _, out, tiny_spec, meta = exported
    with open(os.path.join(out, "run_identity.json")) as f:
        ident = json.load(f)
    assert ident["tokenizer_sha256"] == tiny_spec.sha256
    assert ident["token_data_sha256"] == meta["token_data_sha256"]
    assert ident["document_spans_sha256"] == meta["document_spans_sha256"]
    assert ident["model_vocab_size"] == tiny_spec.model_vocab_size
    assert ident["n_tokens"] == meta["n_tokens"]
    assert ident["n_docs"] == meta["n_docs"]


def test_export_rejects_identity_extent_mismatch(tmp_path, tiny_spec):
    """An inconsistent identity must not produce a checkpoint at all."""
    from train.conftest import build_tiny_corpus
    from train.cog_train import RunIdentity, save_cog_checkpoint
    meta, _ = build_tiny_corpus(tmp_path, tiny_spec, ["北京是中国的首都" * 4])
    lying = dict(meta)
    lying["model_vocab_size"] = tiny_spec.model_vocab_size + 1
    cfg, params, self_state = _small_params(vocab_size=tiny_spec.model_vocab_size)
    with pytest.raises(ValueError, match="tied table"):
        save_cog_checkpoint(params, str(tmp_path / "bad"), 0,
                            run_identity=RunIdentity(tiny_spec, lying),
                            train_cfg=cfg, self_state=self_state)
