"""A resume must continue the dataset the checkpoint was fitted to.

Dataset identity only: optimizer state, RNG and scheduler position are not in
the checkpoint, so this is identity-preserving, not state-exact.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_resume_identity.py -v
"""
import json

import pytest

from train.conftest import build_tiny_corpus
from train.cog_train import RunIdentity, verify_resume_identity

IDENTITY_KEYS = ("tokenizer_sha256", "token_data_sha256",
                 "document_spans_sha256", "model_vocab_size")


def _write_ckpt(tmp_path, meta, name="ckpt"):
    d = tmp_path / name
    d.mkdir()
    with open(d / "run_identity.json", "w") as f:
        json.dump({k: meta[k] for k in IDENTITY_KEYS}, f)
    return str(d)


@pytest.fixture
def identity(tmp_path, registered_tiny):
    meta, _ = build_tiny_corpus(tmp_path, registered_tiny,
                                ["北京是中国的首都" * 5])
    return RunIdentity(registered_tiny, meta)


def test_matching_identity_passes(tmp_path, identity):
    verify_resume_identity(_write_ckpt(tmp_path, identity.dataset_meta),
                           identity)


@pytest.mark.parametrize("key", IDENTITY_KEYS)
def test_each_identity_field_is_compared(tmp_path, identity, key):
    meta = dict(identity.dataset_meta)
    meta[key] = "0" * 8 if isinstance(meta[key], str) else meta[key] + 1
    with pytest.raises(ValueError, match=key):
        verify_resume_identity(_write_ckpt(tmp_path, meta), identity)


def test_legacy_checkpoint_without_identity_is_fatal(tmp_path, identity):
    """Legacy cognitive checkpoints cannot be resumed after unification."""
    d = tmp_path / "legacy"
    d.mkdir()
    with pytest.raises(ValueError, match="no run identity"):
        verify_resume_identity(str(d), identity)


def test_incomplete_identity_file_is_fatal(tmp_path, identity):
    d = tmp_path / "partial"
    d.mkdir()
    with open(d / "run_identity.json", "w") as f:
        json.dump({"tokenizer_sha256": identity.dataset_meta["tokenizer_sha256"]}, f)
    with pytest.raises(ValueError, match="missing identity keys"):
        verify_resume_identity(str(d), identity)


def test_different_corpus_same_tokenizer_is_caught(tmp_path, registered_tiny):
    """The case a shape guard cannot catch.

    Same tokenizer, same model_vocab_size, same dtype; only the text differs.
    So only token_data_sha256 can tell these apart.
    """
    meta_a, _ = build_tiny_corpus(tmp_path, registered_tiny,
                                  ["北京是中国的首都" * 5], prefix="a")
    ckpt = _write_ckpt(tmp_path, meta_a)

    meta_b, _ = build_tiny_corpus(tmp_path, registered_tiny,
                                  ["上海是最大的城市" * 5], prefix="b")
    assert meta_b["model_vocab_size"] == meta_a["model_vocab_size"]
    assert meta_b["tokenizer_sha256"] == meta_a["tokenizer_sha256"]
    assert meta_b["token_data_sha256"] != meta_a["token_data_sha256"]

    with pytest.raises(ValueError, match="token_data_sha256"):
        verify_resume_identity(ckpt, RunIdentity(registered_tiny, meta_b))
