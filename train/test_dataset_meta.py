"""Dataset identity: metadata round-trip, dtype ownership, spans structure.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_dataset_meta.py -v
"""
import numpy as np
import pytest

from train.dataset_meta import (
    META_KEYS, check_data_size, dataset_paths, load_dataset_meta,
    save_dataset_meta, token_dtype, validate_spans,
)
from train.tokenizer_spec import sha256_file


def _meta(n_tokens=10, n_docs=2, dtype="uint32"):
    return {
        "n_tokens": n_tokens, "dtype": dtype,
        "tokenizer_id": "qwen2.5-0.5b", "tokenizer_sha256": "aa",
        "separator_id": 151643, "token_id_max": 151664,
        "model_vocab_size": 151936, "n_docs": n_docs,
        "document_spans_sha256": "bb", "token_data_sha256": "cc",
    }


def _write_spans(path, spans):
    with open(path, "wb") as f:
        np.save(f, np.asarray(spans, dtype=np.int64))


def test_token_dtype_does_not_re_derive():
    """Works with only `dtype` present, so it cannot consult vocab_size."""
    assert token_dtype({"dtype": "uint16"}) == np.dtype(np.uint16)
    assert token_dtype({"dtype": "uint32"}) == np.dtype(np.uint32)


def test_dataset_paths_order_is_data_shape_spans():
    d, s, p = dataset_paths("/tmp/x")
    assert d == "/tmp/x/zhwiki_qwen.dat"
    assert s == "/tmp/x/zhwiki_qwen_shape.json"
    assert p == "/tmp/x/zhwiki_qwen_docs.npy"


def test_meta_round_trip(tmp_path):
    path = str(tmp_path / "m.json")
    save_dataset_meta(path, _meta())
    assert load_dataset_meta(path) == _meta()


def test_meta_requires_every_key(tmp_path):
    partial = _meta()
    del partial["token_data_sha256"]
    with pytest.raises(ValueError, match="token_data_sha256"):
        save_dataset_meta(str(tmp_path / "m.json"), partial)


def test_meta_with_extra_keys_is_normalised(tmp_path):
    path = str(tmp_path / "m.json")
    save_dataset_meta(path, {**_meta(), "stray": 1})
    assert set(load_dataset_meta(path)) == set(META_KEYS)


def test_check_data_size_catches_truncation(tmp_path):
    p = tmp_path / "d.dat"
    p.write_bytes(b"\x00" * (10 * 4))
    check_data_size(_meta(n_tokens=10, dtype="uint32"), str(p))
    with pytest.raises(ValueError, match="bytes"):
        check_data_size(_meta(n_tokens=11, dtype="uint32"), str(p))


def _valid_spans():
    return [[0, 4], [4, 7], [7, 10]]


def _pinned(tmp_path, spans, n_tokens=10, n_docs=3):
    p = str(tmp_path / "s.npy")
    _write_spans(p, spans)
    meta = _meta(n_tokens=n_tokens, n_docs=n_docs)
    meta["document_spans_sha256"] = sha256_file(p)
    return p, meta


def test_validate_spans_accepts_a_partition(tmp_path):
    p, meta = _pinned(tmp_path, _valid_spans())
    assert validate_spans(p, meta).shape == (3, 2)


def test_validate_spans_rejects_hash_mismatch(tmp_path):
    p = str(tmp_path / "s.npy")
    _write_spans(p, _valid_spans())
    meta = _meta(n_tokens=10, n_docs=3)
    meta["document_spans_sha256"] = "wrong"
    with pytest.raises(ValueError, match="document_spans_sha256"):
        validate_spans(p, meta)


def test_validate_spans_rejects_wrong_shape(tmp_path):
    p, meta = _pinned(tmp_path, _valid_spans(), n_docs=99)
    with pytest.raises(ValueError, match="shape"):
        validate_spans(p, meta)


def test_validate_spans_rejects_non_zero_start(tmp_path):
    p, meta = _pinned(tmp_path, [[1, 4], [4, 10]], n_docs=2)
    with pytest.raises(ValueError, match="starts at"):
        validate_spans(p, meta)


def test_validate_spans_rejects_short_tail(tmp_path):
    p, meta = _pinned(tmp_path, [[0, 4], [4, 9]], n_docs=2)
    with pytest.raises(ValueError, match="ends at"):
        validate_spans(p, meta)


def test_validate_spans_rejects_gap(tmp_path):
    """Contiguous-looking but with a gap: not a strict partition."""
    p, meta = _pinned(tmp_path, [[0, 4], [5, 10]], n_docs=2)
    with pytest.raises(ValueError, match="partition"):
        validate_spans(p, meta)


def test_validate_spans_rejects_empty_span(tmp_path):
    p, meta = _pinned(tmp_path, [[0, 4], [4, 4], [4, 10]], n_docs=3)
    with pytest.raises(ValueError, match="empty"):
        validate_spans(p, meta)


def test_validate_spans_rejects_wrong_dtype(tmp_path):
    p = str(tmp_path / "s.npy")
    with open(p, "wb") as f:
        np.save(f, np.asarray([[0, 4], [4, 10]], dtype=np.int32))
    meta = _meta(n_tokens=10, n_docs=2)
    meta["document_spans_sha256"] = sha256_file(p)
    with pytest.raises(ValueError, match="int64"):
        validate_spans(p, meta)
