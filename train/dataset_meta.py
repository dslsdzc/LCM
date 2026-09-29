"""Dataset identity: what a tokenized corpus is, and whether it is intact.

Metadata is the authority for how the `.dat` is read. Three separate digests pin
three separate things, because none implies the others:

    tokenizer_sha256         the token ID space
    token_data_sha256        the token payload
    document_spans_sha256    the document boundaries

A rebuilt corpus can reproduce the same document length distribution while
containing entirely different tokens, so spans identity does not imply data
identity. And `max(token_id) < vocab_size` catches nothing at all: a 30k-token
corpus is full of ids below 151936 and looks entirely legal against a Qwen
vocabulary.
"""
import json
import os

import numpy as np

from train.tokenizer_spec import sha256_file

META_KEYS = (
    "n_tokens", "dtype",
    "tokenizer_id", "tokenizer_sha256",
    "separator_id", "token_id_max", "model_vocab_size",
    "n_docs", "document_spans_sha256", "token_data_sha256",
)


def dataset_paths(mmap_dir, name="zhwiki_qwen"):
    """The three files that together constitute a corpus.

    Returns (data_path, shape_path, spans_path). This order is the repo-wide
    convention: build_corpus, WikiDataIter and resolve_dataset_identity all use
    it, and mixing it up opens a .dat as JSON.
    """
    return (
        os.path.join(mmap_dir, f"{name}.dat"),
        os.path.join(mmap_dir, f"{name}_shape.json"),
        os.path.join(mmap_dir, f"{name}_docs.npy"),
    )


def save_dataset_meta(path, meta):
    """Write metadata atomically. Only META_KEYS are persisted."""
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"dataset metadata missing keys: {missing}")
    payload = {k: meta[k] for k in META_KEYS}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_dataset_meta(path):
    with open(path) as f:
        meta = json.load(f)
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ValueError(f"dataset metadata at {path} missing keys: {missing}")
    return meta


def token_dtype(meta):
    """Reading width, taken from metadata and never re-derived.

    Re-deriving it from a vocabulary size would read an old `.dat` with the
    wrong width the moment a tokenizer changes, silently misaligning every
    token.
    """
    return np.dtype(meta["dtype"])


def check_data_size(meta, data_path):
    """Cheap always-on invariant: exact expected byte count."""
    dtype = token_dtype(meta)
    expected = int(meta["n_tokens"]) * dtype.itemsize
    actual = os.path.getsize(data_path)
    if actual != expected:
        raise ValueError(
            f"{data_path} is {actual} bytes but metadata implies {expected} "
            f"({meta['n_tokens']} tokens x {dtype.itemsize} bytes)")


def validate_spans(spans_path, meta):
    """Load and structurally validate document spans.

    Returns an (n_docs, 2) int64 array. Raises on any violation.
    """
    if sha256_file(spans_path) != meta["document_spans_sha256"]:
        raise ValueError(
            f"{spans_path} does not match document_spans_sha256; it is stale or "
            f"belongs to a different corpus")

    spans = np.load(spans_path)
    if spans.dtype != np.int64:
        raise ValueError(f"spans dtype must be int64, got {spans.dtype}")
    if spans.shape != (int(meta["n_docs"]), 2):
        raise ValueError(f"spans shape {spans.shape} != ({meta['n_docs']}, 2)")
    if len(spans) == 0:
        raise ValueError("spans is empty")
    if spans[0, 0] != 0:
        raise ValueError(f"first span starts at {spans[0, 0]}, expected 0")
    if spans[-1, 1] != int(meta["n_tokens"]):
        raise ValueError(
            f"last span ends at {spans[-1, 1]}, expected {meta['n_tokens']}")
    if np.any(spans[:, 0] >= spans[:, 1]):
        raise ValueError("some span is empty or inverted")
    # Documents are concatenated contiguously, so spans must form a strict
    # partition: [0,e0), [e0,e1), ..., [e_{n-1}, n_tokens).
    if np.any(spans[:-1, 1] != spans[1:, 0]):
        bad = np.flatnonzero(spans[:-1, 1] != spans[1:, 0])[:5].tolist()
        raise ValueError(f"spans are not a strict partition; offenders at {bad}")
    return spans
