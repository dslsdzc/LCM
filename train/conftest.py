"""Shared fixtures for the cognitive-training tests.

Unit tests must not depend on the real Qwen artifacts: a clean CI machine or a
fresh cloud instance has to run them with no download. The single test that does
need them lives in `test_qwen_integration.py` and skips when they are absent.

Two repo conventions this file exists to absorb:

  * `JAX_PLATFORMS` (plural) is set here, at import time, before any test module
    imports jax. The singular `JAX_PLATFORM` is silently ignored by jax 0.10 —
    tests carrying it ran on the GPU by accident. The local GPU is a 4 GB
    GTX 1650, which cannot hold a 0.5B forward pass.
  * `train/` is a package, so pytest puts the repo root on sys.path and these
    helpers import as `from train.conftest import ...`.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

SEPARATOR = "<|endoftext|>"

_WORDS = ["北京", "上海", "首都", "城市", "历史", "文化", "长江", "入海口", "很大"]


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory):
    """A synthetic BPE tokenizer with a known separator token."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    d = tmp_path_factory.mktemp("tiny_tok")
    corpus = d / "corpus.txt"
    corpus.write_text("\n".join(" ".join(_WORDS) for _ in range(200)),
                      encoding="utf-8")

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train([str(corpus)], trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["[UNK]", SEPARATOR],
        show_progress=False,
    ))

    path = d / "tokenizer.json"
    tok.save(str(path))
    return str(path)


@pytest.fixture
def tiny_spec(tiny_tokenizer_dir):
    from train.tokenizer_spec import resolve_tokenizer_spec

    return resolve_tokenizer_spec(
        tiny_tokenizer_dir,
        document_separator=SEPARATOR,
        model_vocab_size=512,     # must exceed the tiny vocab's max id
        tokenizer_id="tiny",
    )


@pytest.fixture
def registered_tiny(monkeypatch, tiny_spec):
    """Make the synthetic tokenizer resolvable by identity.

    resolve_dataset_identity looks the id up in TOKENIZER_REGISTRY, so a corpus
    built with tokenizer_id="tiny" is unresolvable without this. Patching here
    keeps "tiny" out of the production registry while still exercising the
    registry as part of the tokenizer identity contract.
    """
    from train import tokenizer_spec

    monkeypatch.setitem(tokenizer_spec.TOKENIZER_REGISTRY, "tiny",
                        {"separator": SEPARATOR})
    return tiny_spec


def build_tiny_corpus(tmp_path, spec, articles, max_tokens=0, prefix="t"):
    """Build a corpus from `articles`; returns (meta, (data, shape, spans))."""
    from train.data import build_corpus

    text = tmp_path / f"{prefix}_corpus.txt"
    text.write_text("\n".join(articles) + "\n", encoding="utf-8")
    paths = tuple(str(tmp_path / f"{prefix}{s}")
                  for s in (".dat", "_shape.json", "_docs.npy"))
    meta = build_corpus(str(text), spec, *paths, max_tokens=max_tokens)
    return meta, paths
