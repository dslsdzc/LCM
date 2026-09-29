"""Tokenizer identity — the token ID space a dataset was built in.

Three quantities are deliberately kept distinct:

    token_count       entries in the tokenizer vocabulary
    max_token_id      largest id the tokenizer can emit
    model_vocab_size  rows of the embedding / logit matrices

`token_count` and `max_token_id` come from the tokenizer file, and are not
derivable from one another because ids may have gaps. `model_vocab_size` comes
from the model artifact and is deliberately larger: Qwen pads its vocabulary for
tensor-parallel efficiency, so the top rows of `embed_tokens` are never produced
by the tokenizer. Storage is sized by `max_token_id`; tensors by
`model_vocab_size`.

Measured for Qwen2.5-0.5B on 2026-09-29: token_count 151665, max_token_id
151664, `<|endoftext|>` 151643, model_vocab_size 151936.
"""
import dataclasses
import hashlib
import os

# Only what cannot be derived from any file. No paths (this tree moves to a
# cloud instance), no vocab sizes (those come from the model artifact).
TOKENIZER_REGISTRY = {
    "qwen2.5-0.5b": {"separator": "<|endoftext|>"},
    "bpe30k":       {"separator": "[EOS]"},
}

DEFAULT_DATA_ROOT = "/home/DslsDZC/data/lcm"


def data_root():
    """Root of the out-of-repo data tree."""
    return os.environ.get("LCM_DATA_ROOT", DEFAULT_DATA_ROOT)


def tokenizer_path_for(tokenizer_id):
    """$LCM_DATA_ROOT/tokenizers/<tokenizer_id>/tokenizer.json"""
    if tokenizer_id not in TOKENIZER_REGISTRY:
        raise KeyError(
            f"unknown tokenizer_id {tokenizer_id!r}; "
            f"known: {sorted(TOKENIZER_REGISTRY)}")
    return os.path.join(data_root(), "tokenizers", tokenizer_id, "tokenizer.json")


def sha256_file(path, chunk=1 << 20):
    """Streaming SHA-256, so hashing a multi-GB file stays bounded in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def storage_dtype(max_token_id):
    """Storage width for a token id space. Sized by max_token_id only."""
    return "uint16" if max_token_id <= 65535 else "uint32"


@dataclasses.dataclass(frozen=True)
class TokenizerSpec:
    tokenizer_id: str
    path: str

    # what the tokenizer actually defines (read from the file)
    token_count: int
    max_token_id: int

    # storage must hold max_token_id, not the padded tensor extent
    dtype: str

    document_separator_id: int
    sha256: str

    # tensor extent of the token-id space this spec maps onto. Deliberately not
    # Optional: every path that reaches training has a model config, and a Spec
    # carrying None would be an illegal state inside the training object. A
    # tokenizer-only use case, if one ever appears, gets its own type.
    model_vocab_size: int


def resolve_tokenizer_spec(path, document_separator, model_vocab_size,
                           tokenizer_id=None):
    """Read a tokenizer file and pin its identity.

    `document_separator` and `model_vocab_size` are explicit arguments: a bare
    path cannot know which token name marks document boundaries, and the tensor
    extent comes from the model artifact, not from the tokenizer.
    """
    from tokenizers import Tokenizer

    if model_vocab_size is None:
        raise ValueError(
            "model_vocab_size is required and must not be None; read it from "
            "the model artifact's config.json")
    if model_vocab_size <= 0:
        raise ValueError(f"model_vocab_size must be positive, got {model_vocab_size}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer not found: {path}")

    tok = Tokenizer.from_file(path)
    # with_added_tokens=True is passed explicitly so this invariant does not
    # depend on the library default; the special tokens must be counted.
    vocab = tok.get_vocab(with_added_tokens=True)
    if not vocab:
        raise ValueError(f"tokenizer at {path} has an empty vocabulary")

    sep_id = vocab.get(document_separator)
    if sep_id is None:
        raise ValueError(
            f"separator token {document_separator!r} not present in {path}; "
            f"cannot derive document boundaries")

    max_token_id = int(max(vocab.values()))
    if max_token_id >= model_vocab_size:
        raise ValueError(
            f"tokenizer can emit id {max_token_id} but the model matrices have "
            f"only {model_vocab_size} rows; the embedding gather at "
            f"train/qwen_lm.py:176 would be out of bounds")

    return TokenizerSpec(
        tokenizer_id=tokenizer_id or os.path.basename(os.path.dirname(path)),
        path=os.path.abspath(path),
        token_count=len(vocab),
        max_token_id=max_token_id,
        dtype=storage_dtype(max_token_id),
        document_separator_id=int(sep_id),
        sha256=sha256_file(path),
        model_vocab_size=int(model_vocab_size),
    )
