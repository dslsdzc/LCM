"""Corpus builder: framing, spans, digests, atomicity.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_build_corpus.py -v
"""
import glob
import os

import numpy as np
import pytest
from tokenizers import Tokenizer

from train.conftest import build_tiny_corpus
from train.dataset_meta import load_dataset_meta, token_dtype
from train.tokenizer_spec import sha256_file

ARTICLES = ["北京是中国的首都", "上海是城市", "很大"]


def test_framing_is_exactly_payload_plus_separator(tmp_path, tiny_spec):
    """Proves no BOS and no post-processor additions."""
    meta, _ = build_tiny_corpus(tmp_path, tiny_spec, ARTICLES)
    tok = Tokenizer.from_file(tiny_spec.path)
    expected = sum(len(tok.encode(a, add_special_tokens=False).ids) + 1
                   for a in ARTICLES)
    assert meta["n_tokens"] == expected
    assert meta["n_docs"] == len(ARTICLES)


def test_every_span_ends_on_the_separator(tmp_path, tiny_spec):
    meta, (data_path, _, spans_path) = build_tiny_corpus(tmp_path, tiny_spec,
                                                         ARTICLES)
    tokens = np.memmap(data_path, dtype=token_dtype(meta), mode="r",
                       shape=(meta["n_tokens"],))
    spans = np.load(spans_path)
    assert np.all(tokens[spans[:, 1] - 1] == tiny_spec.document_separator_id)


def test_separator_occurs_exactly_once_per_document(tmp_path, tiny_spec):
    meta, (data_path, _, _) = build_tiny_corpus(tmp_path, tiny_spec, ARTICLES)
    tokens = np.memmap(data_path, dtype=token_dtype(meta), mode="r",
                       shape=(meta["n_tokens"],))
    n = int(np.count_nonzero(tokens == tiny_spec.document_separator_id))
    assert n == meta["n_docs"]


def test_metadata_pins_all_three_digests(tmp_path, tiny_spec):
    meta, (data_path, _, spans_path) = build_tiny_corpus(tmp_path, tiny_spec,
                                                         ARTICLES)
    assert meta["token_data_sha256"] == sha256_file(data_path)
    assert meta["document_spans_sha256"] == sha256_file(spans_path)
    assert meta["tokenizer_sha256"] == tiny_spec.sha256
    assert meta["dtype"] == tiny_spec.dtype
    assert meta["model_vocab_size"] == tiny_spec.model_vocab_size
    assert meta["separator_id"] == tiny_spec.document_separator_id
    assert meta["token_id_max"] == tiny_spec.max_token_id


def test_max_tokens_drops_the_partial_trailing_article(tmp_path, tiny_spec):
    """Truncation must never leave a fragment at the end."""
    tok = Tokenizer.from_file(tiny_spec.path)
    first = len(tok.encode(ARTICLES[0], add_special_tokens=False).ids) + 1
    meta, (_, _, spans_path) = build_tiny_corpus(tmp_path, tiny_spec, ARTICLES,
                                                 max_tokens=first + 1)
    spans = np.load(spans_path)
    assert meta["n_docs"] == 1
    assert spans[-1, 1] == meta["n_tokens"]
    assert spans[0, 0] == 0


def test_metadata_is_loadable_and_matches(tmp_path, tiny_spec):
    meta, (data_path, shape_path, spans_path) = build_tiny_corpus(
        tmp_path, tiny_spec, ARTICLES)
    assert os.path.exists(data_path) and os.path.exists(spans_path)
    assert load_dataset_meta(shape_path)["n_tokens"] == meta["n_tokens"]


def test_no_tmp_files_survive(tmp_path, tiny_spec):
    build_tiny_corpus(tmp_path, tiny_spec, ARTICLES)
    assert glob.glob(str(tmp_path / "*.tmp*")) == []


def test_separator_inside_payload_is_rejected(tmp_path, tiny_spec):
    """The separator must stay a unique boundary marker."""
    tok = Tokenizer.from_file(tiny_spec.path)
    probed = tok.encode("x <|endoftext|> y", add_special_tokens=False).ids
    if tiny_spec.document_separator_id not in probed:
        pytest.skip("this tokenizer does not surface the special in plain text")
    with pytest.raises(ValueError, match="inside a document payload"):
        build_tiny_corpus(tmp_path, tiny_spec, ["x <|endoftext|> y"])


def test_empty_input_is_rejected(tmp_path, tiny_spec):
    with pytest.raises(ValueError, match="no articles"):
        build_tiny_corpus(tmp_path, tiny_spec, [""])
