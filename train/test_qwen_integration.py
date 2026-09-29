"""Acceptance test. Requires the real Qwen artifacts (plan Task 1).

Everything else in the suite runs model-free; this file is the single place
that does not, and it skips cleanly when the artifacts are absent.

This drives train_cog itself rather than re-assembling its startup sequence.
A test that hand-rolled identity -> cfg -> init -> verify -> sample -> step
would still pass if train_cog forgot to wire any of it in, which is precisely
the failure this plan exists to prevent.

Run from repo root (CPU is mandatory here: the local GPU is a 4 GB GTX 1650 and
a single 0.5B forward pass exhausts it):
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_qwen_integration.py -v
"""
import glob
import json
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from train.tokenizer_spec import TOKENIZER_REGISTRY, data_root, tokenizer_path_for

_QWEN_TOK = tokenizer_path_for("qwen2.5-0.5b")
_MODEL_DIR = os.path.join(data_root(), "models", "qwen2.5-0.5b")
_CONFIG = os.path.join(_MODEL_DIR, "config.json")
_PARAMS = os.path.join(_MODEL_DIR, "qwen_params.npz")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_QWEN_TOK) and os.path.exists(_CONFIG)
         and os.path.exists(_PARAMS)),
    reason="real Qwen artifacts not present; see plan Task 1")

SEQ_LEN = 32


def _model_vocab_size():
    with open(_CONFIG) as f:
        return int(json.load(f)["vocab_size"])


def test_qwen_tokenizer_facts():
    """The two values the spec could not verify from the dev machine."""
    from train.tokenizer_spec import resolve_tokenizer_spec
    spec = resolve_tokenizer_spec(
        _QWEN_TOK,
        document_separator=TOKENIZER_REGISTRY["qwen2.5-0.5b"]["separator"],
        model_vocab_size=_model_vocab_size(), tokenizer_id="qwen2.5-0.5b")
    assert spec.document_separator_id == 151643
    assert spec.max_token_id < spec.model_vocab_size
    assert spec.dtype == "uint32"


DEFAULT_ARTICLES = [
    "北京是中国的首都，有着悠久的历史和丰富的文化遗产。" * 8,
    "上海是最大的城市之一，位于长江入海口，经济发达。" * 8,
    "广州地处南方，气候温暖湿润，四季常青。" * 8,
]

# Disjoint from DEFAULT_ARTICLES, so the two corpora differ in content while
# sharing a tokenizer, a dtype and a tensor extent.
DIFFERENT_ARTICLES = [
    "这是一份完全不同的数据集，内容与前者毫无重叠。" * 8,
    "第二篇不同的文档，讲述另外的主题。" * 8,
    "第三篇不同的文档，用于验证语料身份。" * 8,
]


def _build_corpus(tmp_path, articles=None):
    """Three documents, each longer than SEQ_LEN + 1.

    `articles` defaults to DEFAULT_ARTICLES; pass DIFFERENT_ARTICLES to get a
    corpus built by the same tokenizer over different text.
    """
    from train.data import build_corpus
    from train.dataset_meta import load_dataset_meta
    from train.tokenizer_spec import resolve_tokenizer_spec

    if articles is None:
        articles = DEFAULT_ARTICLES

    spec = resolve_tokenizer_spec(
        _QWEN_TOK,
        document_separator=TOKENIZER_REGISTRY["qwen2.5-0.5b"]["separator"],
        model_vocab_size=_model_vocab_size(), tokenizer_id="qwen2.5-0.5b")

    text = tmp_path / "c.txt"
    text.write_text("\n".join(articles) + "\n", encoding="utf-8")
    paths = tuple(str(tmp_path / n)
                  for n in ("t.dat", "t_shape.json", "t_docs.npy"))
    build_corpus(str(text), spec, *paths)

    meta = load_dataset_meta(paths[1])
    spans = np.load(paths[2])
    assert ((spans[:, 1] - spans[:, 0]) >= (SEQ_LEN + 1)).all(), \
        "fixture documents are too short for SEQ_LEN; WikiDataIter will refuse"
    return spec, paths, meta


def _checkpoint_dirs(out):
    """Whatever train_cog actually writes.

    Confirm against the save_cog_checkpoint call inside train_cog before
    relying on this: it may write step_NNNNNN subdirectories, or write straight
    into output_dir.
    """
    found = sorted(glob.glob(os.path.join(out, "step_*")))
    if found:
        return found
    if os.path.exists(os.path.join(out, "cog_params.pkl")):
        return [out]
    return []


def test_train_cog_end_to_end_and_resume(tmp_path):
    """The real entry point, not a hand-assembled copy of its startup."""
    from train.cog_train import load_cog_checkpoint, train_cog
    from train.config import LCMConfig

    spec, paths, meta = _build_corpus(tmp_path)
    data, shape, spans = paths
    out = str(tmp_path / "run")

    # If train_cog does not wire in resolve_dataset_identity, this either raises
    # or the assertions below fail. That is the point: a test that reproduced
    # the startup sequence by hand would pass even with the wiring missing.
    train_cog(
        cfg=LCMConfig(),
        output_dir=out,
        steps=2,
        lr=3e-4,
        batch_size=1,
        seq_len=SEQ_LEN,
        log_every=1,
        save_every=1,
        data_path=data,
        shape_path=shape,
        spans_path=spans,
        qwen_ckpt=_PARAMS,
        full_verify=True,
    )

    ckpts = _checkpoint_dirs(out)
    assert ckpts, f"train_cog wrote no checkpoint under {out}"
    ckpt = ckpts[-1]

    for name in ("cog_params.pkl", "tokenizer.json", "run_identity.json"):
        assert os.path.exists(os.path.join(ckpt, name)), \
            f"{name} missing from {ckpt}"

    with open(os.path.join(ckpt, "run_identity.json")) as f:
        ident = json.load(f)
    assert ident["model_vocab_size"] == _model_vocab_size()
    assert ident["tokenizer_sha256"] == spec.sha256
    assert ident["token_data_sha256"] == meta["token_data_sha256"]
    assert ident["document_spans_sha256"] == meta["document_spans_sha256"]
    assert ident["n_docs"] == meta["n_docs"]

    # The tied table carries the Qwen extent, not the tokenizer's, and there is
    # no second token table.
    params, _, _ = load_cog_checkpoint(
        os.path.join(ckpt, "cog_params.pkl"), d_model=LCMConfig().d_model)
    E = np.asarray(params["encoder"]["embed"])
    assert E.shape[0] == _model_vocab_size()
    assert "W_out" not in params

    # Resume through the real entry point: the identity guard must still hold.
    out2 = str(tmp_path / "run2")
    train_cog(
        cfg=LCMConfig(),
        output_dir=out2,
        steps=1,
        lr=3e-4,
        batch_size=1,
        seq_len=SEQ_LEN,
        log_every=1,
        save_every=1,
        data_path=data,
        shape_path=shape,
        spans_path=spans,
        qwen_ckpt=_PARAMS,
        resume=ckpt,
        full_verify=True,
    )
    assert _checkpoint_dirs(out2), "resume wrote no checkpoint"

    # A resume against a DIFFERENT corpus must be fatal even though the
    # tokenizer, dtype and tensor extent are all identical. Only
    # token_data_sha256 can tell these two corpora apart. This fails before
    # any parameter is loaded, so it costs nothing.
    other = tmp_path / "other"
    other.mkdir()
    # DIFFERENT_ARTICLES matters: building the same articles under a different
    # directory would produce an identical token_data_sha256, and the assertion
    # below would fail before the guard was ever reached.
    _, paths_b, meta_b = _build_corpus(other, DIFFERENT_ARTICLES)
    assert meta_b["model_vocab_size"] == meta["model_vocab_size"]
    assert meta_b["tokenizer_sha256"] == meta["tokenizer_sha256"]
    assert meta_b["token_data_sha256"] != meta["token_data_sha256"]

    with pytest.raises(ValueError, match="identity mismatch"):
        train_cog(
            cfg=LCMConfig(),
            output_dir=str(tmp_path / "run3"),
            steps=1,
            lr=3e-4,
            batch_size=1,
            seq_len=SEQ_LEN,
            log_every=1,
            save_every=1,
            data_path=paths_b[0],
            shape_path=paths_b[1],
            spans_path=paths_b[2],
            qwen_ckpt=_PARAMS,
            resume=ckpt,
            full_verify=True,
        )
