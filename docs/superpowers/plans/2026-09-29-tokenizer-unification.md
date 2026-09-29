# Tokenizer Unification + Tied Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cognitive training use one token ID space end to end — Qwen's — so the active channel's supervision stops being noise, and make every way that space can silently drift a fatal startup error.

**Architecture:** A `TokenizerSpec` pins tokenizer identity (tokenizer hash, separator, storage dtype) independently of the temporary language bridge. Dataset metadata pins corpus identity (token data hash, spans hash, tensor extent). The **current cognitive active bridge is Qwen2.5-0.5B only**; Language LCM is retired and must not remain as a fallback path. `train_cog()` refuses to start unless the tokenizer file, the corpus, the spans, and the model tensors all agree, and that identity follows every checkpoint written. The passive readout becomes weight-tied to the encoder embedding, removing `W_out` as a second token table. `WikiDataIter` samples only within document boundaries, because `z_q = f(context)` cannot supervise an unrelated article. The token-space contract remains bridge-independent so a future student backend can adopt it explicitly without reviving retired Language LCM code.

**Tech Stack:** Python 3, JAX, NumPy, HuggingFace `tokenizers`, pytest. The C engine under `infer/` is untouched.

**Spec:** `docs/superpowers/specs/2026-09-29-tokenizer-unification-design.md`

## Global Constraints

- **Path argument order is `(data_path, shape_path, spans_path)` everywhere.** `build_corpus`, `WikiDataIter`, and `resolve_dataset_identity` all use it. Inconsistent ordering silently opens a `.dat` as JSON.
- Data root is `$LCM_DATA_ROOT`, default `/home/DslsDZC/data/lcm`. **No absolute paths in `TOKENIZER_REGISTRY` and no paths at all in dataset metadata.** This tree is destined for a cloud instance.
- Identity mismatches **raise `ValueError`**. There is no warning path anywhere in this plan.
- `dtype` is derived at *build* time, recorded in metadata, and read back from metadata. Never re-derived at read time.
- `TOKENIZER_REGISTRY` entries contain exactly one key, `separator`. Nothing derivable from a file goes in it.
- **Unit tests must not require the real Qwen artifacts.** A clean CI machine or a fresh cloud instance runs `pytest` with no download. Synthetic tokenizer via `tests/conftest.py`; the one real-Qwen test lives in `tests/test_qwen_integration.py` and skips when the artifacts are absent.
- **Language LCM is retired.** It is not a current active backend, not a fallback, and must not be preserved in `train_cog()` as an interchangeable alternative to Qwen. Historical Language-LCM files may remain outside this plan, but they do not define current LCM architecture.
- **Test paths and commands in this plan are written against a `tests/` directory that does not exist.** Read every path and command below through this table:

  | this plan writes | this repo actually uses |
  |---|---|
  | `tests/conftest.py` | `train/conftest.py` |
  | `tests/test_foo.py` | `train/test_foo.py` |
  | `from conftest import X` | `from train.conftest import X` |
  | `python -m pytest tests/test_foo.py -v` | `JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_foo.py -v` |
  | `python -m pytest -q` | `JAX_PLATFORMS=cpu be/bin/python -m pytest train/ -q` |

  Existing tests live in `train/test_*.py` and are conventionally run as modules
  (`be/bin/python -m train.test_fixes_core`); that keeps working. pytest 9.1.1 is
  installed, so the pytest-style tests this plan specifies are fine once
  relocated.

  Three non-obvious requirements:

  - **`be/bin/python` is the repo environment** (3.14.7, jax 0.10.0, numpy 2.5.3,
    pytest 9.1.1). `.venv/` also has jax but is unusable here: with no CUDA
    device it aborts with `Unable to initialize backend 'cuda'`.
  - **`JAX_PLATFORMS=cpu` is mandatory — and note the plural.** The singular
    `JAX_PLATFORM` is silently ignored by jax 0.10: tests carrying it ran on the
    GPU without anyone choosing that. What that costs is not theoretical. The
    local GPU is a GTX 1650 with 4096 MiB, and a single 0.5B forward pass dies
    on it with `RESOURCE_EXHAUSTED ... Failed to profile configs: Out of memory
    while trying to allocate 535.31MiB` on the `[4,151936]` logits matmul. Any
    local verification of the Qwen path must run on CPU.

    Set it in `train/conftest.py` at import time with
    `os.environ.setdefault("JAX_PLATFORMS", "cpu")`. pytest imports conftest
    before the test modules, so it lands before jax is first imported.
  - **`train/` is a package** (it has `__init__.py`), so pytest puts the repo root
    on `sys.path`, not `train/`. That is why the import is
    `from train.conftest import ...` rather than `from conftest import ...`.
- **An enforcement function that is never called is not enforcement.** Task 7 wires identity checks into `train_cog()` before any parameter is allocated.
- **A task may only use what earlier tasks have produced.** Each commit must be runnable on its own. `RunIdentity` is constructed in Task 7 but is only a local variable there; the checkpoint lifecycle picks it up in Task 9.
- No emoji in code, comments, docstrings, commit messages, or docs.
- Phase 1 keeps `decoder.bin = E.T` on disk. Do not simplify the inference loader.
- The legacy 30k corpus is not rebuilt; the `bpe30k` registry entry stays.
- Every task ends with the test suite green and its own commit.

## Prerequisite: the shell is currently unusable

The root filesystem (`/dev/sdc4`, btrfs — `/` and `/tmp` are the same device) is full. Every shell command fails with `ENOSPC ... mkdir '/tmp/claude-1000/...'`. **Task 1 unblocks everything.** Tasks 2-11 can be written before Task 1 completes but cannot be run, tested, or committed.

## Known export-config drift, fixed in Task 9

`save_cog_checkpoint` hardcodes `'n_heads': 4` in the exported `config.json` (`train/cog_train.py:792`) while `LCMConfig.n_heads` is 8. This is **not** vestigial: Python inference reads `n_heads` back out of the exported config and reshapes attention with it (`Q.reshape(N, n_heads, d_h)`), so an exported checkpoint currently describes `d_head = 64` attention for a model trained at `d_head = 32`. Task 9 makes the export take the real `train_cfg` and ends the drift. Do not treat an `n_heads=4` inference artifact as valid before that lands.

---

### Task 1: Prerequisites — shell, data tree, Qwen artifacts

**Files:**
- Create (outside the repo): `/home/DslsDZC/data/lcm/{raw,tokenizers,mmap,models,checkpoints}`

**Interfaces:**
- Consumes: nothing
- Produces: a working shell, and `/home/DslsDZC/data/lcm/tokenizers/qwen2.5-0.5b/tokenizer.json` for the integration test in Task 11

This task also settles the two facts the spec could not verify from this machine: the tokenizer's real `max_token_id`, and that `<|endoftext|>` resolves.

- [ ] **Step 1: Free the root filesystem**

`/home` and `/home/DslsDZC/data` are separate devices, so space there does not help `/`.

```bash
df -h / /tmp /home /home/DslsDZC/data
du -xh --max-depth=1 / 2>/dev/null | sort -rh | head -20
```

Expected: after cleanup, `df -h /` reports free space and `mkdir -p /tmp/claude-1000/-home-DslsDZC-LCM` succeeds.

- [ ] **Step 2: Create the data tree**

```bash
mkdir -p /home/DslsDZC/data/lcm/{raw,tokenizers/qwen2.5-0.5b,mmap,models/qwen2.5-0.5b,checkpoints}
```

- [ ] **Step 3: Set the environment (fish)**

```fish
set -x LCM_DATA_ROOT            /home/DslsDZC/data/lcm
set -x HF_HOME                  /home/DslsDZC/data/hf
set -x HF_HUB_CACHE             /home/DslsDZC/data/hf/hub
set -x JAX_COMPILATION_CACHE_DIR /home/DslsDZC/data/jax-cache
set -x TMPDIR                   /home/DslsDZC/data/tmp
```

Add these to `~/.config/fish/config.fish` so they survive. Note `TMPDIR` may not move the harness's own temp dir, which appears to be a fixed `/tmp` path.

- [ ] **Step 4: Fetch the Qwen artifacts, including the weights**

`checkpoints/qwen_model/config.json` is currently an HTTP redirect body rather than JSON, so treat that whole directory as suspect. Download everything fresh:

```bash
hf download Qwen/Qwen2.5-0.5B \
  config.json tokenizer.json tokenizer_config.json model.safetensors \
  --local-dir /home/DslsDZC/data/lcm/models/qwen2.5-0.5b

cp /home/DslsDZC/data/lcm/models/qwen2.5-0.5b/tokenizer.json \
   /home/DslsDZC/data/lcm/tokenizers/qwen2.5-0.5b/tokenizer.json
```

Use `hf`, not `huggingface-cli`. On this machine `huggingface-cli` is retired
and **fails silently**: it prints a deprecation notice and help text, downloads
nothing, and still exits 0. A script that checks only the exit code will believe
it succeeded.

- [ ] **Step 5: Convert weights to the format the loader expects**

`train/qwen_lm.py:228` loads `qwen_params.npz` (`load_qwen_params`). Find the existing converter and run it:

```bash
ls train/ | grep -i qwen
```

Expected: an existing conversion script (the repo has one for this). Run it to produce `/home/DslsDZC/data/lcm/models/qwen2.5-0.5b/qwen_params.npz`. If none exists, write a one-off that loads `model.safetensors` and saves every tensor as an `.npz` keyed by its original name — `load_qwen_params` reads keys verbatim and derives layer structure from the `model.layers.N.` prefix, so do not rename anything.

- [ ] **Step 6: Verify the artifacts are real files, not redirect bodies**

```bash
python -c "import json;print(json.load(open('/home/DslsDZC/data/lcm/models/qwen2.5-0.5b/config.json'))['vocab_size'])"
ls -la /home/DslsDZC/data/lcm/models/qwen2.5-0.5b/
```

Expected: `151936`, and a non-trivial `qwen_params.npz` (roughly 2 GB).

- [ ] **Step 7: Verify the tokenizer facts the spec flagged as unverified**

```bash
python - <<'PY'
from tokenizers import Tokenizer
t = Tokenizer.from_file("/home/DslsDZC/data/lcm/tokenizers/qwen2.5-0.5b/tokenizer.json")
v = t.get_vocab(with_added_tokens=True)
print("token_count   =", len(v))
print("max_token_id  =", max(v.values()))
print("endoftext     =", v.get("<|endoftext|>"))
print("default get_vocab len =", len(t.get_vocab()))
PY
```

Expected: `endoftext` prints a non-`None` int (151643), and `max_token_id` is comfortably below 151936. Record both numbers. If `endoftext` is `None`, stop: the registry separator name is wrong for this file.

- [ ] **Step 8: Confirm no repo files changed**

```bash
git status --short
```

Expected: nothing under the repo changed. No commit for this task.

---

### Task 2: Model-free test fixtures

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SEPARATOR` constant; pytest fixtures `tiny_tokenizer_dir` (session-scoped path to a synthetic `tokenizer.json`) and `tiny_spec` (a `TokenizerSpec` over it, `model_vocab_size=512`)

Every later task's unit tests use these. Without them, ordinary unit tests would require a downloaded 2 GB model.

- [ ] **Step 1: Confirm the test layout**

```bash
ls tests/ | head -20
```

Match the naming convention you find. Create `tests/conftest.py` if it does not already exist; if it does, append rather than overwrite.

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared fixtures.

Unit tests must not depend on the real Qwen artifacts: a clean CI machine or a
fresh cloud instance has to run `pytest` with no download. The single test that
does need them lives in tests/test_qwen_integration.py and skips when absent.
"""
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

    resolve_dataset_identity looks the id up in TOKENIZER_REGISTRY, so a
    corpus built with tokenizer_id="tiny" is unresolvable without this. Patching
    here keeps "tiny" out of the production registry while still exercising the
    registry as part of the tokenizer identity contract.
    """
    from train import tokenizer_spec

    monkeypatch.setitem(tokenizer_spec.TOKENIZER_REGISTRY, "tiny",
                        {"separator": SEPARATOR})
    return tiny_spec


def build_tiny_corpus(tmp_path, spec, articles, max_tokens=0, prefix="t"):
    """Build a corpus from `articles`; returns (meta, (data, shape, spans))."""
    from train.data import build_corpus

    text = tmp_path / "corpus.txt"
    text.write_text("\n".join(articles) + "\n", encoding="utf-8")
    paths = tuple(str(tmp_path / f"{prefix}{s}")
                  for s in (".dat", "_shape.json", "_docs.npy"))
    meta = build_corpus(str(text), spec, *paths, max_tokens=max_tokens)
    return meta, paths
```

Note the return order: `paths` is `(data, shape, spans)`, matching the global constraint.

- [ ] **Step 3: Verify the fixture builds**

```bash
python -m pytest tests/conftest.py --collect-only -q
python - <<'PY'
import sys; sys.path.insert(0, "tests")
PY
```

Expected: collection succeeds with no errors. (The fixtures are exercised by Task 3 onward.)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add model-free synthetic tokenizer fixtures

Keeps unit tests runnable on a clean machine with no Qwen download."
```

---

### Task 3: `train/tokenizer_spec.py` — tokenizer identity

**Files:**
- Create: `train/tokenizer_spec.py`
- Test: `tests/test_tokenizer_spec.py`

**Interfaces:**
- Consumes: `tiny_spec` (Task 2)
- Produces:
  - `TOKENIZER_REGISTRY: dict[str, dict]` — each value has exactly `{"separator": str}`
  - `data_root() -> str`
  - `tokenizer_path_for(tokenizer_id: str) -> str`
  - `sha256_file(path: str) -> str`
  - `storage_dtype(max_token_id: int) -> str`
  - `TokenizerSpec` frozen dataclass: `tokenizer_id, path, token_count, max_token_id, dtype, document_separator_id, sha256, model_vocab_size`
  - `resolve_tokenizer_spec(path, document_separator, model_vocab_size, tokenizer_id=None) -> TokenizerSpec`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tokenizer_spec.py`:

```python
import os

import pytest

from conftest import SEPARATOR
from train.tokenizer_spec import (
    TOKENIZER_REGISTRY, TokenizerSpec, resolve_tokenizer_spec, sha256_file,
    storage_dtype, tokenizer_path_for,
)


def test_storage_dtype_boundary():
    """Sized by max_token_id, at the exact uint16 edge."""
    assert storage_dtype(0) == "uint16"
    assert storage_dtype(65535) == "uint16"
    assert storage_dtype(65536) == "uint32"
    assert storage_dtype(151665) == "uint32"


def test_registry_entries_carry_only_the_separator():
    assert TOKENIZER_REGISTRY, "registry is empty"
    for tid, entry in TOKENIZER_REGISTRY.items():
        assert set(entry) == {"separator"}, (tid, sorted(entry))


def test_registry_has_no_absolute_paths():
    for tid, entry in TOKENIZER_REGISTRY.items():
        assert not any(str(v).startswith("/") for v in entry.values()), tid


def test_path_for_unknown_tokenizer_raises():
    with pytest.raises(KeyError):
        tokenizer_path_for("does-not-exist")


def test_model_vocab_size_must_not_be_none(tiny_tokenizer_dir):
    """The Spec deliberately carries no illegal state."""
    with pytest.raises(ValueError, match="model_vocab_size"):
        resolve_tokenizer_spec(tiny_tokenizer_dir, document_separator=SEPARATOR,
                               model_vocab_size=None)


def test_unknown_separator_is_rejected(tiny_tokenizer_dir):
    with pytest.raises(ValueError, match="not present"):
        resolve_tokenizer_spec(tiny_tokenizer_dir,
                               document_separator="<|definitely-not-a-token|>",
                               model_vocab_size=512)


def test_tokenizer_wider_than_model_is_rejected(tiny_tokenizer_dir):
    """The out-of-bounds gather guard, at spec resolution."""
    with pytest.raises(ValueError, match="out of bounds"):
        resolve_tokenizer_spec(tiny_tokenizer_dir, document_separator=SEPARATOR,
                               model_vocab_size=8)


def test_resolve_tiny_spec(tiny_spec):
    assert isinstance(tiny_spec, TokenizerSpec)
    assert tiny_spec.dtype == "uint16"          # tiny vocab fits in uint16
    assert tiny_spec.max_token_id < tiny_spec.model_vocab_size
    assert tiny_spec.document_separator_id >= 0
    assert tiny_spec.sha256 == sha256_file(tiny_spec.path)
    assert os.path.isabs(tiny_spec.path)
    assert tiny_spec.model_vocab_size == 512
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tokenizer_spec.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'train.tokenizer_spec'`

- [ ] **Step 3: Implement `train/tokenizer_spec.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tokenizer_spec.py -v`
Expected: all 8 PASS

- [ ] **Step 5: Commit**

```bash
git add train/tokenizer_spec.py tests/test_tokenizer_spec.py
git commit -m "feat(train): add TokenizerSpec pinning tokenizer identity

Keeps token_count, max_token_id and model_vocab_size distinct, derives
storage dtype from max_token_id, and rejects a tokenizer that can emit an
id outside the model matrices."
```

---

### Task 4: `train/dataset_meta.py` — dataset identity

**Files:**
- Create: `train/dataset_meta.py`
- Test: `tests/test_dataset_meta.py`

**Interfaces:**
- Consumes: `sha256_file` (Task 3)
- Produces:
  - `META_KEYS: tuple[str, ...]`
  - `dataset_paths(mmap_dir, name="zhwiki_qwen") -> tuple[str, str, str]` returning `(data_path, shape_path, spans_path)`
  - `save_dataset_meta(path, meta)` / `load_dataset_meta(path) -> dict`
  - `token_dtype(meta) -> np.dtype`
  - `check_data_size(meta, data_path)`
  - `validate_spans(spans_path, meta) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset_meta.py`:

```python
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
        "separator_id": 151643, "token_id_max": 151665,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dataset_meta.py -v`
Expected: `ModuleNotFoundError: No module named 'train.dataset_meta'`

- [ ] **Step 3: Implement `train/dataset_meta.py`**

```python
"""Dataset identity: what a tokenized corpus is, and whether it is intact.

Metadata is the authority for how the `.dat` is read. Three separate digests pin
three separate things, because none implies the others:

    tokenizer_sha256         the token ID space
    token_data_sha256        the token payload
    document_spans_sha256    the document boundaries

A rebuilt corpus can reproduce the same document length distribution while
containing entirely different tokens, so spans identity does not imply data
identity.
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dataset_meta.py -v`
Expected: all 14 PASS

- [ ] **Step 5: Commit**

```bash
git add train/dataset_meta.py tests/test_dataset_meta.py
git commit -m "feat(train): add dataset identity metadata and spans validation

Pins token data, spans and tokenizer separately; reads dtype from metadata
rather than re-deriving it; validates spans as a strict partition."
```

---

### Task 5: `train/data.py` — one-pass corpus builder

**Files:**
- Modify: `train/data.py` (add `build_corpus`; leave `train_tokenizer` alone)
- Test: `tests/test_build_corpus.py`

**Interfaces:**
- Consumes: `TokenizerSpec` (Task 3), `save_dataset_meta` / `sha256_file` (Tasks 3-4), `build_tiny_corpus` (Task 2)
- Produces: `build_corpus(text_path, spec, data_path, shape_path, spans_path, max_tokens=0) -> dict` returning the metadata dict

This replaces the two-pass `tokenize_and_mmap` for the Qwen path. One pass halves the build time on a 540M-token corpus, and it makes "never emit a partial trailing article" structural rather than an extra guard. `tokenize_and_mmap` is left in place until the legacy path is retired.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_corpus.py`:

```python
import glob
import os

import numpy as np
import pytest
from tokenizers import Tokenizer

from conftest import build_tiny_corpus
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_build_corpus.py -v`
Expected: `ImportError: cannot import name 'build_corpus' from 'train.data'`

- [ ] **Step 3: Implement `build_corpus` in `train/data.py`**

Add to the top of `train/data.py`:

```python
import hashlib

from train.dataset_meta import save_dataset_meta
from train.tokenizer_spec import sha256_file
```

Add the function:

```python
def build_corpus(text_path, spec, data_path, shape_path, spans_path,
                 max_tokens=0):
    """One-pass corpus build: tokenize, write, record spans, hash as we go.

    Framing is owned entirely by this pipeline: each article is encoded with
    add_special_tokens=False and exactly one document_separator_id is appended.
    Nothing else is inserted, so the separator count equals the document count.

    Writes to temporary paths and renames only when complete; metadata is
    written last so an interrupted build cannot leave new data with old
    metadata.
    """
    from tokenizers import Tokenizer
    from tqdm import tqdm

    tok = Tokenizer.from_file(spec.path)
    sep = spec.document_separator_id
    dtype = np.dtype(spec.dtype)

    spans = np.empty((1 << 16, 2), dtype=np.int64)
    n_docs = 0
    pos = 0
    digest = hashlib.sha256()

    data_tmp = data_path + ".tmp"
    with open(data_tmp, "wb") as fout, open(text_path, encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Tokenizing"):
            text = line.strip()
            if not text:
                continue

            ids = tok.encode(text, add_special_tokens=False).ids
            if sep in ids:
                raise ValueError(
                    f"separator {sep} occurred inside a document payload; the "
                    f"separator must be a unique boundary marker")

            ids.append(sep)
            seg = np.asarray(ids, dtype=dtype)
            if max_tokens > 0 and pos + len(seg) > max_tokens:
                # Never emit a partial trailing article: that would break both
                # the strict-partition invariant and "every span ends on the
                # separator".
                break

            raw = seg.tobytes()
            fout.write(raw)
            digest.update(raw)

            if n_docs == len(spans):
                spans = np.vstack([spans, np.empty_like(spans)])
            spans[n_docs] = (pos, pos + len(seg))
            n_docs += 1
            pos += len(seg)

        fout.flush()
        os.fsync(fout.fileno())

    if n_docs == 0:
        raise ValueError("no articles were tokenized")

    spans = np.ascontiguousarray(spans[:n_docs])

    # np.save appends .npy, so the temp name must already end in .npy.
    spans_tmp = spans_path[: -len(".npy")] + ".tmp.npy"
    with open(spans_tmp, "wb") as f:
        np.save(f, spans)
        f.flush()
        os.fsync(f.fileno())

    os.replace(data_tmp, data_path)
    os.replace(spans_tmp, spans_path)

    meta = {
        "n_tokens": int(pos),
        "dtype": spec.dtype,
        "tokenizer_id": spec.tokenizer_id,
        "tokenizer_sha256": spec.sha256,
        "separator_id": int(sep),
        "token_id_max": int(spec.max_token_id),
        "model_vocab_size": int(spec.model_vocab_size),
        "n_docs": int(n_docs),
        "document_spans_sha256": sha256_file(spans_path),
        "token_data_sha256": digest.hexdigest(),
    }
    save_dataset_meta(shape_path, meta)   # metadata LAST
    return meta
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_build_corpus.py -v`
Expected: 8 PASS, 1 possible SKIP (`test_separator_inside_payload_is_rejected`)

- [ ] **Step 5: Commit**

```bash
git add train/data.py tests/test_build_corpus.py
git commit -m "feat(train): one-pass corpus builder with document spans

Framing is payload + one separator, no synthetic BOS. Emits spans, pins
token-data and spans digests, writes metadata last, and drops a partial
trailing article rather than leaving a fragment."
```

---

### Task 6: `WikiDataIter` — document-bounded sampling

**Files:**
- Modify: `train/data.py` (`WikiDataIter`, lines ~232-275)
- Test: `tests/test_wiki_data_iter.py`

**Interfaces:**
- Consumes: `load_dataset_meta`, `validate_spans`, `check_data_size`, `token_dtype` (Task 4); `build_tiny_corpus` (Task 2)
- Produces: `WikiDataIter(data_path, shape_path, spans_path, B=16, N=512)` yielding `(inputs, targets)`, both `(B, N)` int32; plus `_sample_starts() -> np.ndarray` and `get_num_batches(tokens_per_step=None) -> int`

Why this matters: `z_q = f(context)` is a global summary and `active_z_margin` (`train/cog_train.py:459-461`) *forces* the generation segment to depend on it. A window straddling a document boundary makes that requirement unsatisfiable and injects noise straight into the cognitive state.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wiki_data_iter.py`:

```python
import numpy as np
import pytest

from conftest import build_tiny_corpus
from train.data import WikiDataIter
from train.dataset_meta import load_dataset_meta, token_dtype

LONG_A = "北京是中国的首都有着悠久的历史和丰富的文化遗产" * 3
LONG_B = "上海是最大的城市之一位于长江入海口" * 3
SHORT = "很大"


@pytest.fixture
def corpus(tmp_path, tiny_spec):
    meta, paths = build_tiny_corpus(tmp_path, tiny_spec, [LONG_A, LONG_B, SHORT])
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_wiki_data_iter.py -v`
Expected: FAIL — `TypeError: WikiDataIter.__init__() takes ...` (old signature has no `data_path`/`spans_path`)

- [ ] **Step 3: Replace `WikiDataIter` in `train/data.py`**

```python
class WikiDataIter:
    """Random-access iterator over memory-mapped tokens, bounded to documents.

    Yields (inputs, targets) of shape (B, N); targets are inputs shifted by one.
    Windows never cross a document boundary.

    This is not a style preference. z_q = f(context) is a global summary and
    active_z_margin forces the generation segment to depend on it, so pairing
    article A's context with article B's generation is unsatisfiable and pushes
    noise into the cognitive state. Ordinary causal-LM packing tolerates
    straddling windows because each token's loss depends only on its own prefix;
    this architecture does not.
    """

    def __init__(self, data_path=MMAP_PATH, shape_path=MMAP_SHAPE_PATH,
                 spans_path=None, B=16, N=512):
        meta = load_dataset_meta(shape_path)
        check_data_size(meta, data_path)

        self.meta = meta
        self.n_tokens = int(meta["n_tokens"])
        self.B = B
        self.N = N

        self.tokens = np.memmap(data_path, dtype=token_dtype(meta), mode="r",
                                shape=(self.n_tokens,))

        spans = validate_spans(spans_path, meta)
        lengths = (spans[:, 1] - spans[:, 0]).astype(np.int64)
        keep = lengths >= (N + 1)          # N inputs + 1 shifted target
        if not keep.any():
            raise ValueError(
                f"no document is long enough for N={N}; the longest is "
                f"{int(lengths.max())} tokens and N+1={N + 1} is required")
        self.spans = np.ascontiguousarray(spans[keep])

        # Weight by usable windows, not by document count, so every valid
        # window is equally likely.
        self.weights = lengths[keep] - N
        probs = self.weights.astype(np.float64)
        self.cdf = np.cumsum(probs / probs.sum())
        self.cdf[-1] = 1.0                # guard against fp drift

    def __iter__(self):
        return self

    def _sample_starts(self):
        """One start offset per sample, each inside a single document."""
        docs = np.searchsorted(self.cdf, np.random.rand(self.B))
        starts = np.empty(self.B, dtype=np.int64)
        for i, d in enumerate(docs):
            lo = int(self.spans[d, 0])
            hi = int(self.spans[d, 1])
            # valid starts are [lo, hi-N-1]; randint's upper bound is exclusive
            starts[i] = np.random.randint(lo, hi - self.N)
        return starts

    def __next__(self):
        starts = self._sample_starts()
        inputs = np.stack([self.tokens[s:s + self.N] for s in starts])
        targets = np.stack([self.tokens[s + 1:s + self.N + 1] for s in starts])
        return inputs.astype(np.int32), targets.astype(np.int32)

    def get_num_batches(self, tokens_per_step=None):
        """Approximate batches per epoch over usable windows.

        weights.sum() counts usable WINDOWS, so it must be scaled by N before
        being divided by a per-step TOKEN budget. Dividing windows by tokens
        directly undercounts by roughly a factor of N.
        """
        if tokens_per_step is None:
            tokens_per_step = self.B * self.N
        usable_tokens = int(self.weights.sum()) * self.N
        return usable_tokens // tokens_per_step
```

Add to the imports at the top of `train/data.py`:

```python
from train.dataset_meta import (
    check_data_size, load_dataset_meta, token_dtype, validate_spans,
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_wiki_data_iter.py -v`
Expected: all 7 PASS

- [ ] **Step 5: Commit**

```bash
git add train/data.py tests/test_wiki_data_iter.py
git commit -m "fix(train): bound WikiDataIter sampling to document spans

Prevents pairing one article's context with another's generation, which
active_z_margin turns into noise in the cognitive state."
```

---

### Task 7: Identity enforcement, `RunIdentity`, wired into `train_cog()`

**Files:**
- Modify: `train/cog_train.py` (add `RunIdentity`, `resolve_dataset_identity`, `verify_tied_table`, `verify_qwen_extent`; remove the retired Language-LCM active path; wire into `train_cog`)
- Modify: `train/config.py` (document the `vocab_size` contract)
- Modify: `lcm.py` and `train/cli_extras.py` (existing `train_cog` call sites — enumerate them in Step 7)
- Test: `tests/test_identity_enforcement.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 5
- Produces:
  - `RunIdentity` frozen dataclass with fields `tokenizer_spec`, `dataset_meta` — defined **here**, because this task is the first to construct one. Task 9 only consumes it.
  - `resolve_dataset_identity(data_path, shape_path, spans_path, full_verify=False, tokenizer_path=None) -> (TokenizerSpec, dict)`
  - `verify_tied_table(params, meta, cfg) -> None`
  - `verify_qwen_extent(params, meta) -> None`

`tokenizer_path` exists so tests can point at a synthetic tokenizer, and so a run can override the registry-resolved path. Without it, `resolve_dataset_identity` is untestable without the real model.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_identity_enforcement.py`:

```python
import numpy as np
import pytest

from conftest import build_tiny_corpus
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
        _resolve(paths=(data_path, shape_path, spans_path), spec=spec)


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
    """Passive-only / future-student runs do not require a Qwen artifact."""
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
    import dataclasses
    import jax
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_identity_enforcement.py -v`
Expected: `ImportError: cannot import name 'resolve_dataset_identity'`

- [ ] **Step 3: Add `RunIdentity` and the three functions to `train/cog_train.py`**

Add to the imports. `dataclasses` is required here, not in Task 9: this task
already calls `dataclasses.replace` in the wiring step:

```python
import dataclasses

from train.dataset_meta import (
    check_data_size, load_dataset_meta, token_dtype, validate_spans,
)
from train.tokenizer_spec import (
    TOKENIZER_REGISTRY, resolve_tokenizer_spec, sha256_file, tokenizer_path_for,
)
```

Add the identity object and the functions:

```python
@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """What a training run was: its tokenizer and its corpus.

    Constructed once at startup and handed to every checkpoint save, so no save
    path can guess a tokenizer from a global default.
    """
    tokenizer_spec: object      # TokenizerSpec
    dataset_meta: dict


# ─── Dataset / model identity enforcement ───────────────────────────────────
#
# Every check below raises. There is no warning path: no active-channel metric
# from a run whose token space cannot be verified is interpretable.

def resolve_dataset_identity(data_path, shape_path, spans_path,
                             full_verify=False, tokenizer_path=None):
    """Verify the tokenizer, corpus, spans and metadata all agree.

    Returns (TokenizerSpec, metadata). Deliberately does not require a Qwen
    artifact: metadata is the authority for the tensor extent. Qwen is the
    current training bridge, but tokenizer/corpus identity is not owned by the
    bridge implementation.
    """
    meta = load_dataset_meta(shape_path)

    tokenizer_id = meta["tokenizer_id"]
    if tokenizer_id not in TOKENIZER_REGISTRY:
        raise ValueError(f"unknown tokenizer_id {tokenizer_id!r} in metadata")
    entry = TOKENIZER_REGISTRY[tokenizer_id]

    spec = resolve_tokenizer_spec(
        path=tokenizer_path or tokenizer_path_for(tokenizer_id),
        document_separator=entry["separator"],
        model_vocab_size=meta["model_vocab_size"],
        tokenizer_id=tokenizer_id,
    )

    if spec.sha256 != meta["tokenizer_sha256"]:
        raise ValueError("tokenizer file changed since the corpus was built")
    if spec.document_separator_id != meta["separator_id"]:
        raise ValueError(
            f"separator is {spec.document_separator_id} but metadata says "
            f"{meta['separator_id']}")
    if spec.max_token_id != meta["token_id_max"]:
        raise ValueError(
            f"tokenizer max id is {spec.max_token_id} but metadata says "
            f"{meta['token_id_max']}")
    if meta["dtype"] not in ("uint16", "uint32"):
        raise ValueError(f"unsupported token dtype {meta['dtype']!r}")
    if np.dtype(meta["dtype"]) != np.dtype(spec.dtype):
        raise ValueError(
            f"dataset dtype is {meta['dtype']} but tokenizer identity requires "
            f"{spec.dtype}")

    # Always on, and cheap: catches a truncated or replaced .dat. This runs
    # after dtype identity so a width drift is reported as identity drift rather
    # than merely as a byte-size mismatch.
    check_data_size(meta, data_path)

    if full_verify and sha256_file(data_path) != meta["token_data_sha256"]:
        raise ValueError(
            f"{data_path} does not match token_data_sha256; the corpus content "
            f"changed since this metadata was written")

    spans = validate_spans(spans_path, meta)

    tokens = np.memmap(data_path, dtype=token_dtype(meta), mode="r",
                       shape=(int(meta["n_tokens"]),))

    # Structure does not imply semantics: a builder that wrote wrong-but-
    # contiguous offsets passes every partition check above.
    if np.any(tokens[spans[:, 1] - 1] != meta["separator_id"]):
        raise ValueError(
            "some document span does not end on the separator; the boundary "
            "set is not what the metadata claims")

    if full_verify:
        n_sep = int(np.count_nonzero(tokens == meta["separator_id"]))
        if n_sep != int(meta["n_docs"]):
            raise ValueError(
                f"found {n_sep} separators but metadata claims "
                f"{meta['n_docs']} documents; the separator is not a unique "
                f"boundary marker")

    return spec, meta


def verify_tied_table(params, meta, cfg):
    """Assert the tied table was built at the metadata's extent.

    An assertion, not a derivation: the resume path can carry a checkpoint
    built under a different config.
    """
    E = params["encoder"]["embed"]
    want = (int(meta["model_vocab_size"]), cfg.d_model)
    if tuple(E.shape) != want:
        raise ValueError(
            f"tied table E has shape {tuple(E.shape)}, expected {want}")


def verify_qwen_extent(params, meta):
    """If the Qwen bridge is loaded, its tensor vocabulary must match metadata.

    Qwen is the only current cognitive active bridge. If it is absent, the run
    is passive-only (or a future student path) and there is no Qwen extent to
    verify. Retired Language LCM is intentionally not considered here.
    """
    qwen = params.get("qwen")
    if qwen is None:
        return

    embed_vocab = int(qwen["model.embed_tokens.weight"].shape[0])
    if embed_vocab != int(meta["model_vocab_size"]):
        raise ValueError(
            f"Qwen embed_tokens has {embed_vocab} rows but the corpus was built "
            f"for model_vocab_size={meta['model_vocab_size']}")

    # Mirror train/qwen_lm.py:220: the head is weight-tied to the embedding
    # when lm_head.weight is absent. An lm_head-less checkpoint is valid.
    lm_weight = qwen.get("lm_head.weight", qwen["model.embed_tokens.weight"])
    logit_vocab = int(lm_weight.shape[0])
    if logit_vocab != int(meta["model_vocab_size"]):
        raise ValueError(
            f"Qwen lm_head has {logit_vocab} rows but the corpus was built for "
            f"model_vocab_size={meta['model_vocab_size']}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_identity_enforcement.py -v`
Expected: all 19 PASS (the metadata parametrization expands to seven)

- [ ] **Step 5: Wire the checks into `train_cog()`**

Find the training entry point:

```bash
grep -n "^def train_cog\|^def cog_train\|def train_cog" train/cog_train.py lcm.py
```

Add `data_path`, `shape_path`, `spans_path` and `full_verify` to its signature.

Then insert this block **before any parameter is allocated**. Concretely:
`train_cog` currently creates its RNG at `train/cog_train.py:556`
(`rng, init_rng = jax.random.split(rng)`) and calls `init_cog_params` on the
next line. **Keep line 556 where it is**, place steps 1-3 immediately after it,
and let step 4 replace the existing `init_cog_params` call. The block uses
`init_rng`, which line 556 defines — putting the block above the split, "at the
top of the function" as one might read it, raises `NameError`.

```python
    # 1. Verify corpus/tokenizer BEFORE allocating model parameters.
    spec, meta = resolve_dataset_identity(
        data_path, shape_path, spans_path, full_verify=full_verify,
    )

    # 2. The dataset owns the tensor vocabulary extent. Note this does not
    #    require a Qwen artifact: metadata is authoritative.
    cfg = dataclasses.replace(
        cfg, vocab_size=int(meta["model_vocab_size"]),
    )

    # 3. Identity for this run, constructed here and NOT later: Task 10 inserts
    #    the resume check right after this line and before allocation, so
    #    run_identity must already exist. It stays a LOCAL VARIABLE for now —
    #    Task 9 teaches the checkpoint lifecycle to consume it. Do not pass it
    #    to save_cog_checkpoint yet; its signature does not accept it until
    #    then, and this commit has to run.
    run_identity = RunIdentity(spec, meta)

    # 4. Only now is model allocation allowed.
    params, self_state = init_cog_params(
        cfg, init_rng, qwen_ckpt=qwen_ckpt, resume=resume,
    )

    # 5. Assert the actual tensors, do not assume init did the right thing.
    verify_tied_table(params, meta, cfg)
    verify_qwen_extent(params, meta)

    # 6. Sampling must consume the same dataset identity.
    data_iter = WikiDataIter(data_path, shape_path, spans_path,
                             B=batch_size, N=seq_len)
```

Use the names above consistently. This task intentionally renames the cognitive bridge argument from `lang_ckpt` to `qwen_ckpt`; do not preserve the obsolete name as an alias, because it falsely implies Language LCM is still supported.

- [ ] **Step 5b: Remove the retired Language-LCM active path**

Language LCM is historical code, not a current cognitive backend. Remove it from
`train/cog_train.py` rather than carrying it forward as a fallback discovered by
an agent reading old branches.

First enumerate every legacy reference. This grep must be **repo-wide**, not
scoped to `train/cog_train.py`: the rename and the removal reach callers
(`lcm.py`, `train/cli_extras.py`) and any test still passing `lang_ckpt=`. A
single-file grep leaves those references alive behind a backend that no longer
exists.

```bash
grep -rn "Language LCM\|lang_lcm\|_load_lang_lm_checkpoint\|lang_ckpt" \
  --include=*.py . | grep -v "^./docs/"
```

Hits inside `train/lang_lcm.py` and `train/train_lang_lcm.py` are expected and
stay: those files keep their own internals and are not modernized here. Remove
only references reachable from the *cognitive* path.

Then make the cognitive path Qwen-only:

- delete `_load_lang_lm_checkpoint`;
- remove `params["lang_lcm"]` and every `elif p.get("lang_lcm") ...` active-channel branch;
- remove `has_lang` / `Language LCM (frozen)` logging and stale Stage-1 wording;
- rename the cognitive bridge parameter `lang_ckpt` to `qwen_ckpt` in
  `init_cog_params`, `train_cog`, and their callers;
- if `qwen_ckpt` is supplied and is not an `.npz`, raise a fatal error instead
  of trying to interpret it as a Language-LCM pickle:

```python
    if qwen_ckpt:
        if not qwen_ckpt.endswith(".npz"):
            raise ValueError(
                "Language LCM is retired; cognitive training accepts only the "
                "Qwen .npz bridge (or no bridge for passive-only runs)")
        _load_qwen_checkpoint(qwen_ckpt, params, d)
    else:
        params["qwen"] = None
        if "z_proj" not in params:
            params["z_proj"] = None
```

Historical files such as `train/lang_lcm.py` may remain untouched in this plan
if they are retained for archaeology or old experiments. They are not a current
LCM backend and must not be referenced by `train_cog()`.

- [ ] **Step 6: Update every `train_cog` call site for the new arguments**

```bash
grep -rn "train_cog(" --include=*.py .
```

`train_cog` currently takes `data_path` and `shape_path` only (`train/cog_train.py:534-537`). Adding `spans_path` and `full_verify`, and renaming the bridge argument to `qwen_ckpt`, affects every caller. Expect hits in `lcm.py` and `train/cli_extras.py` — enumerate them yourself rather than trusting that list. Replace the obsolete cognitive handoff option `--from-lang-ckpt` with `--qwen-ckpt`; do not keep a compatibility alias that implies the retired backend is valid.

Derive the spans path from the data path unless overridden, and add a `--spans` CLI option:

```python
def _spans_path_for(data_path, override=None):
    """<name>.dat -> <name>_docs.npy, the convention build_corpus writes."""
    if override:
        return override
    base, _ = os.path.splitext(data_path)
    return base + "_docs.npy"
```

```python
train_cog(
    ...,
    data_path=data,
    shape_path=shape,
    spans_path=_spans_path_for(data, getattr(args, "spans", None)),
    full_verify=True,
)
```

`full_verify=True` is the default for **training**: production runs must verify the corpus content hash. `resolve_dataset_identity`'s own default stays `False`, because it is also a cheap utility for tests and quick local starts. If training never passes `True`, "production runs verify the corpus" is design text rather than behaviour.

Then update `CLAUDE.md`, which currently documents the flag and the backend this
task retires:

- the cog-training example invokes
  `--from-lang-ckpt checkpoints/lang_lm/lang_final.pkl`. Rewrite it with
  `--qwen-ckpt` and the `.npz` bridge path;
- the module table lists `train/lang_lcm.py` and `train/train_lang_lcm.py` as
  Transitional. Note there that the cognitive path no longer references them;
- the `LangLCM` section should say the same thing.

Renaming a flag without updating the documentation that advertises it ships a
known-wrong command. This task is where the rename happens, so the doc update
belongs here.

- [ ] **Step 7: Note the config contract**

In `train/config.py`, extend the `vocab_size` comment (line 10):

```python
    vocab_size: int = 30000     # Tensor extent (rows of E). NOT authoritative
                                # under cognitive training: it is replaced from
                                # dataset metadata at startup. Deliberately not
                                # derived from use_qwen.
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(train): fatal dataset identity enforcement wired into train_cog

Adds RunIdentity, verifies tokenizer, token data, spans and the tied table
before allocating parameters, and updates the CLI call sites for spans and
full_verify. The checkpoint lifecycle adopts the identity in a later commit."
```

---

### Task 8: Tied readout — remove `W_out`

**Files:**
- Modify: `train/cog_train.py` (`init_cog_params` lines 168-169 and 187; passive readout at line 399)
- Modify: `train/export_cog_ckpt.py` (if present — verify in Step 6)
- Test: `tests/test_tied_readout.py`

**Interfaces:**
- Consumes: Task 7's `verify_tied_table`
- Produces: parameters with no `W_out` key; passive logits computed as `z_q @ E.T`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tied_readout.py`:

```python
import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def _params(vocab_size=512):
    from train.cog_train import init_cog_params
    from train.config import LCMConfig
    cfg = dataclasses.replace(LCMConfig(), vocab_size=vocab_size)
    params, _ = init_cog_params(cfg, jax.random.PRNGKey(0))
    return cfg, params


def test_no_w_out_leaf_exists():
    """The second token table must be gone, not merely unused."""
    _, params = _params()
    assert "W_out" not in params
    assert "embed" in params["encoder"]


def test_encoder_embed_has_model_vocab_rows():
    cfg, params = _params()
    assert params["encoder"]["embed"].shape == (cfg.vocab_size, cfg.d_model)


def test_passive_logits_equal_z_at_embed_transpose():
    """The readout is the tied table, not a lookalike."""
    rng = jax.random.PRNGKey(1)
    z = jax.random.normal(rng, (2, 32, 64))
    E = jax.random.normal(rng, (100, 64))
    tied = jnp.einsum("bsd,vd->bsv", z, E)
    assert jnp.allclose(tied, z @ E.T, atol=1e-5)
    assert tied.shape == (2, 32, 100)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tied_readout.py -v`
Expected: `test_no_w_out_leaf_exists` FAILS — `assert 'W_out' not in params`

- [ ] **Step 3: Remove `W_out` from `init_cog_params`**

Delete the resume fallback at lines 168-169:

```python
        if 'W_out' not in params:
            params['W_out'] = jax.random.normal(keys[7], (d, cfg.vocab_size)) * (d ** -0.5)
```

Delete the fresh-init line at 187:

```python
        params['W_out'] = jax.random.normal(keys[7], (d, cfg.vocab_size)) * (d ** -0.5)
```

Update the `init_cog_params` docstring: replace `encoder + codebooks + W_out` with `encoder (whose embed is the tied readout) + codebooks`.

- [ ] **Step 4: Rewrite the passive readout**

In `make_train_step`'s `loss_fn`, replace line 399:

```python
            # ── Passive channel: z_q @ E.T (tied to the encoder embedding) ──
            # One trainable token table: E is both the encoder's input
            # embedding and the readout matrix. This is what puts the passive
            # channel in the same token space as the active channel.
            p_logits = jnp.einsum('bsd,vd->bsv', z_qs, p['encoder']['embed'])
```

Leave `p_target` and `p_loss` unchanged; `p_logits.shape[-1]` is now `model_vocab_size`.

- [ ] **Step 5: Keep the export working, in this same commit**

Removing `W_out` breaks `save_cog_checkpoint` immediately: `train/cog_train.py:786-787` derive `d` and `V` from `params['W_out']`, and `:843` writes it as the `decoder.bin` fallback. Deferring this to Task 9 would leave this commit's checkpoint export broken.

Make only the minimal shape fix here. The signature, the tokenizer copy, and everything identity-related stay untouched until Task 9.

```python
    # No W_out: the readout is the encoder embedding, tied.
    E = _to_np(params['encoder']['embed'])
    d = E.shape[1]
    V = E.shape[0]
```

```python
    else:
        # Phase 1: write E.T so the existing inference loader keeps working.
        # This duplicates bytes already in encoder.bin; it is not a second
        # training parameter and not a second Adam state. Dropping it requires
        # changing the loader, which is deliberately out of scope.
        dec = E.T.copy()
```

The local `cfg = {...}` dict at `:791` still reads `d` and `V`, so it needs no change yet.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tied_readout.py -v`
Expected: 3 PASS

- [ ] **Step 7: Classify every other `W_out` hit — do not fix them all**

```bash
grep -rn "W_out" --include=*.py . | grep -v "^./docs/"
ls train/ | grep -i export
```

**Two unrelated things are named `W_out` in this repo.** Only one is being removed:

| Owner | Verdict |
|---|---|
| Cognitive params — `params['W_out']`, the readout being deleted | remove or retarget |
| Retired Language-LCM files (`train/lang_lcm.py`, `train/train_lang_lcm.py`) | **out of scope; do not treat them as a current backend** |

Language LCM has already been retired from the current architecture. Its old
files may still contain an unrelated `W_out`, but those hits are historical and
must not influence the cognitive tying change. Do not delete or modernize those
files merely because this grep finds them; Task 7 has already removed their path
from `train_cog()`.

Expected cognitive hits: `train/cog_train.py:843` (the export fallback, handled in Task 9), plus `train/export_cog_ckpt.py` if that file exists — it reads `params["W_out"]` shapes directly, so retarget it to `E.shape` / `E.T` in the same commit, or delete it with a note in the commit message. Never leave it broken.

Any hit you cannot classify confidently: leave it and report it rather than guessing.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(train): tie the passive readout to the encoder embedding

Removes W_out as a second trainable token table. Passive logits are now
z_q @ E.T, and the export derives d/V from E and writes E.T, keeping the
checkpoint export working in the same commit."
```

---

### Task 9: Checkpoint identity — export, supervisor, every save path

**Files:**
- Modify: `train/cog_train.py` (`save_cog_checkpoint` at line 764; the tokenizer copy at 895-898; the export config block at 791-811)
- Modify: `train/train_supervisor.py` (`save_best` — verify in Step 1)
- Modify: `train/export_cog_ckpt.py` if it exists (Step 8)
- Modify: every other call site
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: Tasks 7-8. `RunIdentity` already exists from Task 7 — consume it, do not redefine it.
- Produces: `save_cog_checkpoint(params, output_dir, step, *, run_identity, train_cfg, self_state=None)`; a `run_identity.json` beside every checkpoint

Two deliberate decisions:

- **`train_cfg` is required, and the internal dict is renamed.** A parameter named `cfg` would be shadowed by the local `cfg = {...}` at line 791, so the local becomes `export_cfg` and the real training config is passed in. This is not cosmetic. That block hardcodes `n_heads: 4` while `LCMConfig.n_heads` is 8, and Python inference reads `n_heads` back out of the exported config and reshapes attention with it (`Q.reshape(N, n_heads, d_h)`). An exported checkpoint therefore describes `d_head = 64` attention for a model trained at `d_head = 32` — a wrong inference artifact, not a metadata quirk.
- **`run_identity` is required.** An omittable argument will be omitted, and the export will fall back to probing a global `data/tokenizer.json` — the exact bug class this plan exists to kill. A missing argument must surface as a `TypeError`.

- [ ] **Step 1: Find every call site**

```bash
grep -rn "save_cog_checkpoint\|save_best" --include=*.py .
```

Known: the periodic/final/SIGINT saves in `train/cog_train.py`, and `train/train_supervisor.py::save_best`. Record every hit; each needs `run_identity`, which means `save_best` needs it added to its own signature and passed down.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_export.py`:

```python
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
    from conftest import build_tiny_corpus
    from train.cog_train import RunIdentity, save_cog_checkpoint
    meta, _ = build_tiny_corpus(tmp_path, tiny_spec, ["北京是中国的首都" * 4])
    cfg, params, self_state = _small_params(vocab_size=tiny_spec.model_vocab_size)
    out = str(tmp_path / "ckpt")
    save_cog_checkpoint(params, out, 0,
                        run_identity=RunIdentity(tiny_spec, dict(meta)),
                        train_cfg=cfg, self_state=self_state)
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
    from conftest import build_tiny_corpus
    from train.cog_train import RunIdentity, save_cog_checkpoint
    meta, _ = build_tiny_corpus(tmp_path, tiny_spec, ["北京是中国的首都" * 4])
    lying = dict(meta)
    lying["model_vocab_size"] = tiny_spec.model_vocab_size + 1
    cfg, params, self_state = _small_params(vocab_size=tiny_spec.model_vocab_size)
    with pytest.raises(ValueError, match="tied table"):
        save_cog_checkpoint(params, str(tmp_path / "bad"), 0,
                            run_identity=RunIdentity(tiny_spec, lying),
                            train_cfg=cfg, self_state=self_state)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL on the signature and `TypeError` assertions — `run_identity` and
`train_cfg` are not yet parameters. `RunIdentity` itself imports fine; it was
defined in Task 7.

- [ ] **Step 4: Implement the export changes**

`RunIdentity` is already defined and imported from Task 7; do not redefine it
here. Task 8 already made `d`, `V` and `decoder.bin` derive from `E`, so this
step is only about identity and config plumbing.

Change the signature at line 764:

```python
def save_cog_checkpoint(params, output_dir, step, *, run_identity, train_cfg,
                        self_state=None):
    """Save full checkpoint + export codebooks + tied readout for C engine.

    run_identity is required: the checkpoint must record the tokenizer and
    corpus it was actually trained with, never a probed global default.
    train_cfg is required because the exported config feeds Python inference,
    which reshapes attention by n_heads.
    """
```

At the top of the body, assert consistency before writing anything:

```python
    meta = run_identity.dataset_meta
    spec = run_identity.tokenizer_spec
    E = _to_np(params['encoder']['embed'])
    if int(meta["model_vocab_size"]) != E.shape[0]:
        raise ValueError(
            f"tied table E has {E.shape[0]} rows but run_identity says "
            f"model_vocab_size={meta['model_vocab_size']}")
```

Rename the local `cfg = {...}` dict at line 791 to `export_cfg`, and take the
values that describe the model from `train_cfg` rather than hardcoding them:

```python
    export_cfg = {
        # Shape facts come from the tensors actually being saved.
        'd_model': d,
        'vocab_size': V,
        'n_encoder_layers': len(enc.get('layers', [])) if enc else train_cfg.n_encoder_layers,
        'M_top': _to_np(params['hrq']['top']['A']).shape[0],
        'M_fine': _to_np(params['hrq']['fine'][0]['A']).shape[0],
        'n_hrq_layers': len(params['hrq']['fine']),
        'M_sparse': _to_np(params['sparse']['C']).shape[0],
        'M_lr': _to_np(params['lowrank']['A_V']).shape[0],
        'M_man': _to_np(params['manifold']['C']).shape[0],
        'M_bind': _to_np(params['binding']['key_cb'][0]['A']).shape[0],
        'M_contrast': _to_np(params['contrast']['C_a'][0]['A']).shape[0],
        'n_bind_layers': len(params['binding']['key_cb']),
        'n_contrast_layers': len(params['contrast']['C_a']),
        'n_self_codes': _to_np(params['self']['modes']).shape[0],

        # Training choices come from train_cfg.
        'max_seq_len': train_cfg.max_seq_len,
        'n_heads': train_cfg.n_heads,
        'd_ff': train_cfg.d_ff,
        'n_lattices': train_cfg.n_lattices,
        'n_lr_layers': train_cfg.n_lr_layers,
        'r_max': train_cfg.r_max,
        't_dim': train_cfg.t_dim,
        'n_value_pairs': train_cfg.n_value_pairs,
        'M_danger': train_cfg.M_danger,
        'max_inference_steps': train_cfg.max_inference_steps,
        'convergence_tol': train_cfg.convergence_tol,
        'entropy_threshold': train_cfg.entropy_threshold,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(export_cfg, f)
```

`n_heads` is the load-bearing one — it was hardcoded to 4 against a training
default of 8, and inference reshapes attention with it. But do not stop there:
with `train_cfg` in hand, no entry should remain a literal. Each is either a
tensor shape (derive it) or a training choice (take it from `train_cfg`).
Change the `json.dump(cfg, f)` below to `export_cfg` and confirm nothing else in
the body still reads the old name.

Replace the tokenizer copy at 895-898:

```python
    # The tokenizer this run actually trained with, not a probed default.
    import shutil
    shutil.copy2(spec.path, os.path.join(output_dir, "tokenizer.json"))

    with open(os.path.join(output_dir, "run_identity.json"), "w") as f:
        json.dump({
            "tokenizer_id": spec.tokenizer_id,
            "tokenizer_sha256": spec.sha256,
            "token_data_sha256": meta["token_data_sha256"],
            "document_spans_sha256": meta["document_spans_sha256"],
            "model_vocab_size": int(meta["model_vocab_size"]),
            "n_tokens": int(meta["n_tokens"]),
            "n_docs": int(meta["n_docs"]),
            "step": int(step),
        }, f, indent=2)
```

- [ ] **Step 5: Update the supervisor**

Add `run_identity` to `train/train_supervisor.py::save_best()`'s signature and pass **both** new arguments through:

```python
    save_cog_checkpoint(
        ...,
        run_identity=run_identity,
        train_cfg=self.cfg,
    )
```

Confirm the attribute the supervisor actually stores its config under; `self.cfg` is the likely name. The supervisor constructs no identity of its own — it must receive the run's, so a checkpoint auto-saved by the supervisor is indistinguishable from one saved by the training loop.

- [ ] **Step 6: Update the remaining call sites**

**Both** new arguments are required, so every call site from Step 1 passes both. Threading only `run_identity` leaves at least one path raising `TypeError: missing 1 required keyword-only argument: 'train_cfg'`:

```python
    save_cog_checkpoint(
        ...,
        run_identity=run_identity,
        train_cfg=cfg,
    )
```

A save path that cannot obtain either should raise, not default.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_export.py -v`
Expected: all 9 PASS

- [ ] **Step 8: Bring the standalone exporter under the identity contract**

```bash
ls train/ | grep -i export
test -f train/export_cog_ckpt.py && grep -n "data-dir\|data_dir\|tokenizer" train/export_cog_ckpt.py
```

If that file exists, it is a second export implementation that can bypass
everything above. It takes an arbitrary `--data-dir` and copies
`<data_dir>/tokenizer.json`, and it also owns a second copy of export-config
construction. Keeping both implementations guarantees future identity/config
drift.

Use one policy only:

```bash
grep -R "export_cog_ckpt" -n --exclude=export_cog_ckpt.py .
```

If there is **no in-repo caller**, delete `train/export_cog_ckpt.py`. The main
`save_cog_checkpoint` path already emits the inference artifacts and is the
single supported exporter after this task. Mention the deletion in the commit.

If an in-repo caller exists, stop this task and report the caller rather than
silently maintaining a second exporter. Do not retarget or duplicate the config
logic inside this plan.

- [ ] **Step 9: Run the whole suite**

```bash
python -m pytest -q
```

Expected: green.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(train): export writes E.T, requires RunIdentity, emits run_identity.json

decoder.bin is the tied table transposed (phase 1 disk duplication only).
Every checkpoint now records the tokenizer, token data and spans it was
trained with, and the export refuses an inconsistent identity."
```

---

### Task 10: Resume identity validation

**Files:**
- Modify: `train/cog_train.py` (add `verify_resume_identity`; call it in `train_cog` before `init_cog_params`)
- Test: `tests/test_resume_identity.py`

**Interfaces:**
- Consumes: Task 9's `run_identity.json`
- Produces: `RESUME_IDENTITY_KEYS: tuple[str, ...]`; `verify_resume_identity(resume_dir, run_identity) -> None`

Task 9 writes the identity; nothing reads it yet. So this remains possible:

```
checkpoint A / tokenizer A / corpus A
        |
        |  resume with tokenizer B / corpus B, same model_vocab_size
        v
verify_tied_table passes      <- E.shape is (151936, 256) either way
```

`load_stage2_params` only reads `cog_params.pkl` and never looks at
`run_identity.json`. That is exactly the drift this plan exists to make fatal.

Define the two operations explicitly:

- **identity-preserving resume** — continue from a checkpoint of the *same*
  dataset. Tokenizer, corpus content, spans and tensor extent must all match.
- **warm start** — start new-data training from old parameters. The dataset may
  change. Explicitly out of scope for this plan.

This is deliberately **not** called an "exact continuation". The checkpoint
lifecycle does not yet save optimizer state, RNG, or scheduler/global-step
state, so a resumed run does not continue the interrupted trajectory
bit-for-bit. Making resume *state*-exact is a separate problem from making the
token space verifiable, and belongs in its own plan.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resume_identity.py`:

```python
import json

import pytest

from conftest import build_tiny_corpus
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_resume_identity.py -v`
Expected: `ImportError: cannot import name 'verify_resume_identity'`

- [ ] **Step 3: Implement `verify_resume_identity`**

```python
# The fields that make a resume identity-preserving. The rest of the metadata
# (n_tokens, n_docs) is implied by these.
RESUME_IDENTITY_KEYS = (
    "tokenizer_sha256",
    "token_data_sha256",
    "document_spans_sha256",
    "model_vocab_size",
)


def verify_resume_identity(resume_dir, run_identity):
    """A resume must continue the dataset the checkpoint was fitted to.

    Dataset identity only. Optimizer state, RNG and scheduler position are not
    saved in the checkpoint, so this makes a resume identity-preserving, not
    state-exact.

    Resuming onto a different corpus is not a warm start with extra steps: the
    parameters were never fitted to that token space. Warm-starting from
    mismatched data is a separate, explicit operation and is not supported here.
    """
    path = os.path.join(resume_dir, "run_identity.json")
    if not os.path.exists(path):
        raise ValueError(
            f"{resume_dir} has no run identity; legacy cognitive checkpoints "
            f"cannot be resumed after tokenizer unification")

    with open(path) as f:
        stored = json.load(f)
    current = run_identity.dataset_meta

    missing = [k for k in RESUME_IDENTITY_KEYS if k not in stored]
    if missing:
        raise ValueError(f"{path} is missing identity keys: {missing}")

    for key in RESUME_IDENTITY_KEYS:
        if stored[key] != current[key]:
            raise ValueError(
                f"resume identity mismatch on {key}: checkpoint has "
                f"{stored[key]!r}, this run has {current[key]!r}; a resume must "
                f"continue the same dataset")
```

Add a module-level `import json` to `train/cog_train.py` — it is currently
imported inside functions only.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_resume_identity.py -v`
Expected: all 7 PASS (the parametrized case expands to four)

- [ ] **Step 5: Call it from `train_cog` before allocation**

In Task 7's startup block, insert immediately after step 3
(`run_identity = RunIdentity(spec, meta)`) and before step 4
(`init_cog_params`):

```python
    # 3b. A resume must continue the same dataset. Checked before allocation so
    #     a mismatched resume fails before anything is loaded.
    if resume:
        verify_resume_identity(resume, run_identity)
```

Step 3 of that block already builds `run_identity` at this point; if it were
still constructed at the end of the block, as an earlier draft had it, this
line would reference an undefined name.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(train): resume must match the checkpoint's run identity

Closes the gap where a checkpoint could be resumed against a different
corpus or tokenizer with the same tensor extent. Legacy checkpoints
without a run identity are rejected."
```

---

### Task 11: End-to-end smoke test

**Files:**
- Test: `tests/test_qwen_integration.py`

**Interfaces:**
- Consumes: every prior task; the real Qwen artifacts from Task 1
- Produces: nothing; this is the acceptance test

Unit tests prove each piece. This proves the chain is connected: `train_cog` startup → identity → `cfg` replacement → `E` → Qwen extent → a real update step → a checkpoint carrying its identity → a resume that still validates.

- [ ] **Step 1: Write the integration test**

Create `tests/test_qwen_integration.py`:

```python
"""Acceptance test. Requires the real Qwen artifacts (plan Task 1).

Everything else in the suite runs model-free; this file is the single place
that does not, and it skips cleanly when the artifacts are absent.

This drives train_cog itself rather than re-assembling its startup sequence.
A test that hand-rolled identity -> cfg -> init -> verify -> sample -> step
would still pass if train_cog forgot to wire any of it in, which is precisely
the failure this plan exists to prevent.
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


def test_qwen_tokenizer_facts():
    """The two values the spec could not verify from the dev machine."""
    from train.tokenizer_spec import resolve_tokenizer_spec
    with open(_CONFIG) as f:
        model_vocab_size = int(json.load(f)["vocab_size"])
    spec = resolve_tokenizer_spec(
        _QWEN_TOK, document_separator=TOKENIZER_REGISTRY["qwen2.5-0.5b"]["separator"],
        model_vocab_size=model_vocab_size, tokenizer_id="qwen2.5-0.5b")
    assert spec.document_separator_id == 151643
    assert spec.max_token_id < spec.model_vocab_size
    assert spec.dtype == "uint32"


SEQ_LEN = 32


def _model_vocab_size():
    with open(_CONFIG) as f:
        return int(json.load(f)["vocab_size"])


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
    # token_data_sha256 can tell these two corpora apart.
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
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_qwen_integration.py -v`
Expected: 2 PASS if the Task 1 artifacts are present, 2 SKIP otherwise

- [ ] **Step 3: Run the whole suite**

```bash
python -m pytest -q
```

Expected: green, with the integration file skipped only if the artifacts are genuinely absent.

- [ ] **Step 4: Commit**

```bash
git add tests/test_qwen_integration.py
git commit -m "test: end-to-end smoke covering identity through one update

Exercises train_cog's startup chain, document-bounded sampling, a real
Qwen forward/backward, and a checkpoint carrying its run identity."
```

---

### Task 12: Build the production corpus

**Files:**
- Create: `/home/DslsDZC/data/lcm/raw/zhwiki_cleaner.txt` (moved from the repo root)
- Create: `/home/DslsDZC/data/lcm/mmap/zhwiki_qwen.{dat,json}` and `zhwiki_qwen_docs.npy`

**Interfaces:**
- Consumes: Tasks 3-7
- Produces: the corpus every later cognitive-training run reads

- [ ] **Step 1: Move the source text off the repo**

```bash
mv /home/DslsDZC/LCM/zhwiki_cleaner.txt /home/DslsDZC/data/lcm/raw/zhwiki_cleaner.txt
df -h /home/DslsDZC/data
```

Expected: room for `2 x` the current corpus size, since uint32 doubles bytes and Qwen's vocabulary may change the token count.

- [ ] **Step 2: Build**

```bash
python - <<'PY'
import json
from train.data import build_corpus
from train.tokenizer_spec import resolve_tokenizer_spec, tokenizer_path_for

ROOT = "/home/DslsDZC/data/lcm"
with open(f"{ROOT}/models/qwen2.5-0.5b/config.json") as f:
    model_vocab_size = int(json.load(f)["vocab_size"])

spec = resolve_tokenizer_spec(
    tokenizer_path_for("qwen2.5-0.5b"),
    document_separator="<|endoftext|>",
    model_vocab_size=model_vocab_size,
    tokenizer_id="qwen2.5-0.5b",
)
meta = build_corpus(
    f"{ROOT}/raw/zhwiki_cleaner.txt", spec,
    f"{ROOT}/mmap/zhwiki_qwen.dat",
    f"{ROOT}/mmap/zhwiki_qwen_shape.json",
    f"{ROOT}/mmap/zhwiki_qwen_docs.npy",
)
print(json.dumps(meta, indent=2))
PY
```

Note `model_vocab_size` is read from `config.json`, never hardcoded. This is a long job; it tokenizes the corpus once.

- [ ] **Step 3: Verify the build**

```bash
python - <<'PY'
from train.cog_train import resolve_dataset_identity
ROOT = "/home/DslsDZC/data/lcm/mmap"
spec, meta = resolve_dataset_identity(
    f"{ROOT}/zhwiki_qwen.dat",
    f"{ROOT}/zhwiki_qwen_shape.json",
    f"{ROOT}/zhwiki_qwen_docs.npy",
    full_verify=True,
)
print("OK", meta["n_tokens"], "tokens,", meta["n_docs"], "docs")
print("dtype", meta["dtype"], "max_token_id", meta["token_id_max"])
PY
```

Expected: `OK ... tokens, ... docs`, no exception.

- [ ] **Step 4: Sanity-check sampling**

```bash
python - <<'PY'
from train.data import WikiDataIter
ROOT = "/home/DslsDZC/data/lcm/mmap"
it = WikiDataIter(f"{ROOT}/zhwiki_qwen.dat", f"{ROOT}/zhwiki_qwen_shape.json",
                  f"{ROOT}/zhwiki_qwen_docs.npy", B=8, N=64)
sep = it.meta["separator_id"]
for _ in range(200):
    inputs, _ = next(it)
    assert not (inputs == sep).any(), "a window crossed a document boundary"
print("sampling OK; usable windows:", int(it.weights.sum()))
PY
```

Expected: `sampling OK`, no assertion error.

- [ ] **Step 5: Record the outcome**

No repo files change, so no commit. Record `n_tokens` and `n_docs` for the run notes.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §3 `TokenizerSpec`, registry, resolution | 3 |
| §3.1 document separator, no BOS | 5 |
| §4 metadata schema, three digests | 4 |
| §4.1 startup enforcement, size invariant, separator check | 7 |
| §4.2 tied table + Qwen bridge extent | 7 |
| §5.1 framing, payload guard, spans, atomic build | 4, 5 |
| §5.1.2 document-bounded sampling | 6 |
| §5.2 config contract | 7 |
| §5.3 tied readout | 8 |
| §5.4 export, `RunIdentity` | 9 |
| §6 storage layout, environment | 1, 12 |
| §8 rollout, phase 1 only | 9 |
| §8 "old checkpoints are void" (extended to resume) | 10 |
| acceptance, beyond the spec's §7 | 11 |

Spec §7's ten verification items map to: (1) Task 3 `test_storage_dtype_boundary` and `test_resolve_tiny_spec`; (2) Task 5 `test_metadata_pins_all_three_digests` + Task 6 `test_iter_reads_dtype_from_metadata`; (3) Task 7 `test_tampered_metadata_raises` + `test_token_data_sha256_is_checked_only_under_full_verify`; (4) Task 5 `test_separator_occurs_exactly_once_per_document` + `test_separator_inside_payload_is_rejected`; (5) Task 8 `test_passive_logits_equal_z_at_embed_transpose`; (6) Task 9 `test_decoder_bin_is_the_transposed_tied_table`; (7) Task 6 document-bounded tests; (8) Task 7 `test_qwen_extent_*`; (9) Task 4 spans tests + Task 7 `test_wrong_boundary_raises_even_when_partition_is_valid`; (10) Task 7 `test_qwen_extent_skips_when_qwen_params_absent` + `test_tied_table_shape_is_asserted`.

**Placeholder scan:** no TBD/TODO. Every code step contains complete code. Task 7 Step 5 and Task 9 Step 6 name the exact `grep` command and the exact block to insert rather than saying "thread it through".

**Type consistency:** `(data_path, shape_path, spans_path)` is used in `build_corpus`, `WikiDataIter`, `resolve_dataset_identity`, and `build_tiny_corpus`'s return value. `TokenizerSpec` field names are identical across Tasks 3, 5, 7, 9. `RunIdentity` is defined in Task 7 and only consumed in Task 9 — no forward reference across commits. `save_cog_checkpoint`'s step protocol (`new_params, new_opt, loss, aux_out`, `train/cog_train.py:527`) is used correctly in Task 11, and the optimizer tree excludes `qwen` on both the init and the update side (`:521-526`). `tiny_spec.model_vocab_size` (512) matches the `vocab_size` used in Tasks 8 and 9.

**Known deviations from the spec:**

1. §5.1 implies keeping the two-pass count-then-write builder. Task 5 uses one pass: it halves build time on a 540M-token corpus and makes "never emit a partial trailing article" structural rather than an extra guard. Observable metadata is identical.
2. §5.4 does not mention a training-config parameter. Task 9 adds a required `train_cfg` and renames the function's local `cfg` dict to `export_cfg`, because the export config feeds Python inference and currently hardcodes `n_heads: 4` against a training default of 8. An earlier draft resolved this by dropping the parameter; that was wrong, because `n_heads` changes real computation at inference.
3. §4.2 describes a conditional active-backend extent check. The current cognitive bridge is Qwen only; Language LCM is retired and Task 7 removes that legacy fallback. `verify_qwen_extent` therefore validates Qwen whenever Qwen params are present and returns when they are absent (passive-only / future-student path). This keeps tokenizer ownership separate from the temporary bridge without pretending retired Language LCM is still interchangeable.
4. The spec does not cover resume. §8 says old checkpoints are void and there is no migration, but nothing stopped a *new* checkpoint from being resumed against a different corpus with a matching tensor extent — the identity was written and never read. Task 10 adds `verify_resume_identity` and defines resume as identity-preserving rather than state-exact, with warm start explicitly out of scope.
