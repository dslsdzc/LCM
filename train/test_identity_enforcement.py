"""Fatal identity enforcement: tokenizer, corpus, spans, tensors must agree.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_identity_enforcement.py -v
"""
import dataclasses

import numpy as np
import pytest

from train.conftest import build_tiny_corpus
from train.cog_train import (
    resolve_dataset_identity, verify_qwen_extent, verify_tied_table,
)
from train.dataset_meta import load_dataset_meta, save_dataset_meta
from train.tokenizer_spec import sha256_file


@pytest.fixture
def corpus(tmp_path, registered_tiny):
    """A tiny corpus whose tokenizer_id is resolvable.

    registered_tiny patches TOKENIZER_REGISTRY. Without it, the metadata's
    tokenizer_id="tiny" fails the registry lookup inside
    resolve_dataset_identity before any other check runs.
    """
    meta, paths = build_tiny_corpus(tmp_path, registered_tiny,
                                    ["北京是中国的首都" * 5, "上海很大" * 5])
    return registered_tiny, paths


def _tamper(shape_path, key, value):
    meta = load_dataset_meta(shape_path)
    meta[key] = value
    save_dataset_meta(shape_path, meta)


def _resolve(paths, spec, **kw):
    # tokenizer_path is passed explicitly: tokenizer_path_for("tiny") would
    # resolve under $LCM_DATA_ROOT, where the synthetic tokenizer does not live.
    return resolve_dataset_identity(*paths, tokenizer_path=spec.path, **kw)


def test_clean_corpus_passes(corpus):
    spec, paths = corpus
    got_spec, meta = _resolve(paths, spec)
    assert got_spec.sha256 == spec.sha256
    assert meta["n_tokens"] > 0


@pytest.mark.parametrize("key,value", [
    ("tokenizer_sha256", "deadbeef"),
    ("separator_id", 999999),
    ("token_id_max", 12345),
    ("dtype", "uint32"),
    ("n_tokens", 1),
    ("document_spans_sha256", "deadbeef"),
    ("model_vocab_size", 8),
])
def test_tampered_metadata_raises(corpus, key, value):
    spec, paths = corpus
    _tamper(paths[1], key, value)
    with pytest.raises(ValueError):
        _resolve(paths, spec)


def test_token_data_sha256_is_checked_only_under_full_verify(corpus):
    """Distinct from the other tamper cases: this digest is opt-in."""
    spec, paths = corpus
    _tamper(paths[1], "token_data_sha256", "deadbeef")

    # Default: not checked, so this passes.
    _resolve(paths, spec)

    # full_verify: checked, so this raises.
    with pytest.raises(ValueError, match="token_data_sha256"):
        _resolve(paths, spec, full_verify=True)


def test_full_verify_accepts_a_clean_corpus(corpus):
    spec, paths = corpus
    _resolve(paths, spec, full_verify=True)


def test_wrong_boundary_raises_even_when_partition_is_valid(corpus):
    """Structure alone cannot catch boundaries that are not on separators."""
    spec, (data_path, shape_path, spans_path) = corpus

    spans = np.load(spans_path)
    bad = spans.copy()
    bad[0, 1] = bad[0, 0] + 1          # still a partition, no longer a boundary
    bad[1, 0] = bad[0, 1]
    with open(spans_path, "wb") as f:
        np.save(f, bad)
    # Re-pin the hash so only the semantic check can catch it.
    _tamper(shape_path, "document_spans_sha256", sha256_file(spans_path))

    with pytest.raises(ValueError, match="separator"):
        _resolve((data_path, shape_path, spans_path), spec)


def test_unknown_tokenizer_id_in_metadata_raises(corpus):
    spec, paths = corpus
    _tamper(paths[1], "tokenizer_id", "no-such-tokenizer")
    with pytest.raises(ValueError, match="unknown tokenizer_id"):
        _resolve(paths, spec)


# ── tied table ──────────────────────────────────────────────────────────────

class _Cfg:
    d_model = 256


def test_tied_table_shape_is_asserted():
    meta = {"model_vocab_size": 512}
    good = {"encoder": {"embed": np.zeros((512, 256), dtype=np.float32)}}
    verify_tied_table(good, meta, _Cfg())
    bad = {"encoder": {"embed": np.zeros((300, 256), dtype=np.float32)}}
    with pytest.raises(ValueError, match="tied table"):
        verify_tied_table(bad, meta, _Cfg())


# ── Qwen bridge extent ──────────────────────────────────────────────────────

META = {"model_vocab_size": 512}


def _qwen_stub(embed_rows, lm_rows=None):
    q = {"model.embed_tokens.weight": np.zeros((embed_rows, 8), dtype=np.float32)}
    if lm_rows is not None:
        q["lm_head.weight"] = np.zeros((lm_rows, 8), dtype=np.float32)
    return {"qwen": q}


def test_qwen_extent_skips_when_qwen_params_absent():
    """Passive-only runs do not require a Qwen artifact."""
    verify_qwen_extent({"qwen": None}, META)
    verify_qwen_extent({}, META)


def test_qwen_extent_accepts_matching_embed_and_lm_head():
    verify_qwen_extent(_qwen_stub(512, 512), META)


def test_qwen_extent_accepts_tied_checkpoint_without_lm_head():
    """The lm_head-less case must PASS; it is a legitimate tied checkpoint."""
    verify_qwen_extent(_qwen_stub(512, None), META)


def test_qwen_extent_rejects_wrong_embed_rows():
    with pytest.raises(ValueError, match="embed_tokens"):
        verify_qwen_extent(_qwen_stub(300, 300), META)


def test_qwen_extent_rejects_wrong_lm_head_rows():
    with pytest.raises(ValueError, match="lm_head"):
        verify_qwen_extent(_qwen_stub(512, 300), META)


def test_retired_language_lcm_checkpoint_is_rejected(tmp_path):
    """A .pkl Language-LCM bridge must not silently reactivate legacy code."""
    jax = pytest.importorskip("jax")
    from train.cog_train import init_cog_params
    from train.config import LCMConfig

    old = tmp_path / "language_lcm.pkl"
    old.write_bytes(b"legacy")
    cfg = dataclasses.replace(LCMConfig(), d_model=32, d_ff=48, n_heads=4,
                              d_head=8, vocab_size=64,
                              M_top=16, M_fine=8, M_sparse=16, M_lr=16,
                              M_man=16, M_bind=16, M_contrast=16,
                              n_self_codes=8, use_bf16=False)
    with pytest.raises(ValueError, match="Language LCM.*retired"):
        init_cog_params(cfg, jax.random.PRNGKey(0), qwen_ckpt=str(old))
