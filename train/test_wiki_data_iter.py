"""Document-bounded sampling: no window may cross a document boundary.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_wiki_data_iter.py -v
"""
import numpy as np
import pytest

from train.conftest import build_tiny_corpus
from train.data import WikiDataIter
from train.dataset_meta import load_dataset_meta, token_dtype

LONG_A = "北京是中国的首都有着悠久的历史和丰富的文化遗产" * 3
LONG_B = "上海是最大的城市之一位于长江入海口" * 3
SHORT = "很大"


@pytest.fixture
def corpus(tmp_path, tiny_spec):
    meta, paths = build_tiny_corpus(tmp_path, tiny_spec,
                                    [LONG_A, LONG_B, SHORT])
    return tiny_spec, paths, meta


def test_iter_reads_dtype_from_metadata(corpus):
    spec, (data, shape, spans), _ = corpus
    meta = load_dataset_meta(shape)
    it = WikiDataIter(data, shape, spans, B=4, N=8)
    assert it.tokens.dtype == token_dtype(meta)


def test_no_separator_ever_appears_in_inputs(corpus):
    """A separator in inputs means the window crossed a document boundary."""
    spec, (data, shape, spans), _ = corpus
    it = WikiDataIter(data, shape, spans, B=32, N=8)
    for _ in range(100):
        inputs, _ = next(it)
        assert not (inputs == spec.document_separator_id).any()


def test_every_window_lies_inside_one_span(corpus):
    spec, (data, shape, spans), _ = corpus
    it = WikiDataIter(data, shape, spans, B=64, N=8)
    view = np.load(spans)
    for _ in range(50):
        for s in it._sample_starts():
            contained = (view[:, 0] <= s) & (s + it.N + 1 <= view[:, 1])
            assert contained.any(), f"start {s} is not inside any span"


def test_short_documents_are_never_sampled(corpus):
    spec, (data, shape, spans), _ = corpus
    it = WikiDataIter(data, shape, spans, B=64, N=8)
    view = np.load(spans)
    short = view[(view[:, 1] - view[:, 0]) < (it.N + 1)]
    assert len(short) >= 1, "fixture must contain a too-short document"
    starts = np.concatenate([it._sample_starts() for _ in range(200)])
    for lo, hi in short:
        assert not ((starts >= lo) & (starts < hi)).any()


def test_targets_are_inputs_shifted_by_one(corpus):
    """Deterministic: compares within one batch, no resampling."""
    spec, (data, shape, spans), _ = corpus
    it = WikiDataIter(data, shape, spans, B=8, N=8)
    inputs, targets = next(it)
    assert inputs.shape == (8, 8) and targets.shape == (8, 8)
    assert inputs.dtype == np.int32 and targets.dtype == np.int32
    assert (targets[:, :-1] == inputs[:, 1:]).all()


def test_sampling_is_window_weighted(tmp_path, tiny_spec):
    """Documents are weighted by usable windows, not uniformly."""
    _, paths = build_tiny_corpus(tmp_path, tiny_spec, [LONG_A * 4, LONG_B])
    it = WikiDataIter(paths[0], paths[1], paths[2], B=64, N=8)
    starts = np.concatenate([it._sample_starts() for _ in range(300)])
    counts = np.array([int(((starts >= lo) & (starts < hi)).sum())
                       for lo, hi in it.spans])
    expected = it.weights / it.weights.sum()
    observed = counts / counts.sum()
    assert np.allclose(observed, expected, atol=0.05), (observed, expected)


def test_all_documents_too_short_raises(tmp_path, tiny_spec):
    _, paths = build_tiny_corpus(tmp_path, tiny_spec, [SHORT, SHORT])
    with pytest.raises(ValueError, match="long enough"):
        WikiDataIter(paths[0], paths[1], paths[2], B=4, N=512)
