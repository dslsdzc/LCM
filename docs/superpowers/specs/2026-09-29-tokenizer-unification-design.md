# Tokenizer Unification + Tied Readout — Design Spec

Date: 2026-09-29
Status: approved for implementation planning
Scope: cognitive training data path, vocabulary ownership, passive-channel readout

## 1. Problem

Cognitive training with the Qwen bridge currently tokenizes the corpus with a
30k BPE trained on zhwiki (`data/tokenizer.json`), while the active channel is
Qwen2.5-0.5B with a 151936-token vocabulary. The two token spaces are never
reconciled, and nothing fails loudly.

Concretely, in one training step (`train/cog_train.py:339-475`):

| Site | Line | Uses |
|---|---|---|
| `p_target = inputs[:, ctx_len]` | `:406` | 30k-BPE id |
| `p_logits = einsum('bsd,dv->bsv', z_qs, p['W_out'])` | `:399` | LCM 30k vocab |
| `gen_in = inputs[:, ctx_len-1:]` | `:358` | 30k-BPE ids |
| `qwen_forward(qwen_params, gen_in, ...)` | `:433` | gathers Qwen `embed_tokens` (151936, 896) |
| `a_targets = targets[:, ctx_len-1:...]` | `:429` | 30k-BPE ids |
| `active_loss(a_logits, a_targets)` | `:444` | Qwen 151936-wide logits vs 30k labels |

`train/qwen_lm.py:176` indexes Qwen's `(151936, 896)` embedding table with
30k-range ids. There is no shape error and no assertion. The gather returns the
wrong rows, and `a_loss` compares Qwen's logits against labels whose integer
value denotes a *different* token in Qwen's ordering — so the active-channel
loss is noise.

Because `z_q` is trained by both channels jointly, the gradient intended to pull
the cognitive state toward "useful for the frozen language model" was noise.
This is the mechanism behind the observed "active channel looks like it is
learning, but means nothing" behaviour.

Consequence, accepted: every existing cog checkpoint is void. No migration.

## 2. Goals / non-goals

Goals:

- One token ID space across the whole training chain: encoder input, passive
  target, active channel input, active target.
- Vocabulary ownership decoupled from the active-channel backend.
- Dataset identity recorded and enforced; mismatch is fatal, never a warning.
- One trainable token table, not two.

Non-goals:

- No chat formatter. Chat data is a separate future path.
- No changes to the C inference engine.
- No corpus rebuild for the legacy 30k path.
- No inference-loader simplification (deferred, see §8).

## 3. TokenizerSpec

Three quantities are kept distinct, because collapsing them into one
`vocab_size` is how the *next* silent mismatch gets built:

| Quantity | Meaning | Qwen2.5-0.5B |
|---|---|---|
| `token_count` | entries in the tokenizer vocabulary | read from file |
| `max_token_id` | largest id the tokenizer can emit | read from file |
| `model_vocab_size` | rows of the embedding / logit matrices | from model config, verified against weights (§4.2) |

The first two come from the tokenizer file. The third comes from the *model*
configuration, and is already confirmed in-repo: `train/qwen_lm.py:250` declares
`vocab_size: 151936`, and `embed_tokens` (`:262`) and `lm_head` (`:264`) are both
`(151936, 896)`. The tokenizer's own extent is **smaller** than the model's
matrix — Qwen pads its vocabulary for tensor-parallel efficiency, so the top
rows of `embed_tokens` are never produced by the tokenizer.

`vocab_size` must **not** be derived from `use_qwen`. The two are independent
axes:

```
use_qwen         = active channel backend
tokenizer_spec   = token ID space
model_vocab_size = tensor vocabulary extent
```

Valid future combinations this decoupling must permit: Qwen tokenizer with a
non-Qwen active channel; Qwen tokenizer with a student-only run; another
tokenizer with another renderer.

```python
@dataclasses.dataclass(frozen=True)
class TokenizerSpec:
    tokenizer_id: str
    path: str

    # what the tokenizer actually defines (read from the file)
    token_count: int
    max_token_id: int

    # storage must hold max_token_id, not the padded tensor extent
    dtype: str                   # "uint16" | "uint32"

    document_separator_id: int
    sha256: str

    # tensor extent of the token-id space this spec maps onto.
    # Deliberately not Optional: every path that reaches training has a model
    # config, and a Spec carrying None would be an illegal state inside the
    # training object. A tokenizer-only use case, if one ever appears, gets its
    # own type rather than widening this one.
    model_vocab_size: int
```

Resolution takes the separator and the tensor extent **explicitly**. Neither is
guessed from the file — a bare path cannot know which token name is the
separator:

```python
resolve_tokenizer_spec(
    path=qwen_tokenizer_path,
    document_separator="<|endoftext|>",
    model_vocab_size=qwen_model_vocab_size,   # from the model artifact, not a literal
)
```

The registry holds only what genuinely cannot be derived:

```python
TOKENIZER_REGISTRY = {
    "qwen2.5-0.5b": {"separator": "<|endoftext|>"},
    "bpe30k":       {"separator": "[EOS]"},
}
```

No paths, and no `model_vocab_size`. The file is located as
`$LCM_DATA_ROOT/tokenizers/<tokenizer_id>/tokenizer.json`, overridable with
`--tokenizer`. An absolute local path in the registry would resolve to nothing
the moment the tree is uploaded to the 5090 instance, which is exactly where this
data is going.

`model_vocab_size` has one source of truth: the model artifact.

- At **corpus build** time, read it from the downloaded `config.json`
  (`model_config["vocab_size"]`), so the ~2 GB npz need not be loaded.
- At **training start**, verify it against the actual weight tensors (§4.2).

Writing 151936 into the registry would create a second truth source that can
drift from the checkpoint actually being trained. The number itself is not
wrong — `train/qwen_lm.py:250` confirms it — but it belongs in one place.

Inside `resolve_tokenizer_spec`:

```python
vocab = tokenizer.get_vocab(with_added_tokens=True)   # includes <|endoftext|>
max_token_id = max(vocab.values())
token_count  = len(vocab)
dtype = "uint16" if max_token_id <= 65535 else "uint32"
sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
```

`dtype` is sized by `max_token_id`, not `model_vocab_size`: storage only has to
hold what the tokenizer emits, while the tensor extent is deliberately larger.

`token_count` is the number of tokenizer *entries*, not the size of a contiguous
id range — ids may have gaps, so `token_count != max_token_id + 1` in general.
It is informational. `max_token_id` is the safety-relevant quantity: it is what
storage must hold and what §4.2's invariant bounds.

`with_added_tokens=True` is passed explicitly so this invariant does not depend
on the library default.

`sha256` and the separator *id* are never hardcoded, so a swapped tokenizer file
cannot produce a stale spec.

Consequence to expect during training: rows of `E` above `max_token_id` are never
a target. They still sit in the passive softmax denominator, so they receive a
downward gradient and settle at low logit values, but they never receive a
positive signal. This mirrors how Qwen's own padded rows behave and needs no
special handling.

### 3.1 Document separator

`bos_id` / `eos_id` is the wrong abstraction and is removed. The single concept
is `document_separator_id`.

For pretraining-style Wikipedia text, Qwen's path uses `<|endoftext|>` (151643):

```
article A
<|endoftext|>
article B
<|endoftext|>
```

**No synthetic per-article BOS.** Qwen2.5 ships `add_bos_token=false`, so a
fabricated leading BOS has no basis in the tokenizer's own contract.

Qwen2.5-0.5B **base** is self-consistent with this choice: its configuration
sets `bos_token_id` and `eos_token_id` both to 151643, defines 151643 as
`<|endoftext|>`, and ships `add_bos_token=false`. Using `<|endoftext|>` as the
wiki document separator follows the tokenizer's own contract rather than working
around it.

Note that base and Instruct differ in configured EOS/PAD: the Instruct variant
points EOS at `<|im_end|>` and PAD at `<|endoftext|>`. This spec targets the
**base** tokenizer. The separator id is never hardcoded — it is resolved by name
at spec-resolution time, so a wrong or renamed token surfaces as a resolution
error rather than a silently wrong id.

Chat data, when it arrives, goes through a separate chat formatter that uses
`<|im_start|>` / `<|im_end|>`. It does not reuse this path.

## 4. Dataset metadata

`zhwiki_qwen_shape.json` becomes the authority for how the `.dat` is read:

```json
{
  "n_tokens": 540000000,
  "dtype": "uint32",
  "tokenizer_id": "qwen2.5-0.5b",
  "tokenizer_sha256": "...",
  "separator_id": 151643,
  "token_id_max": 151665,
  "model_vocab_size": 151936,
  "n_docs": 1234567,
  "document_spans_sha256": "...",
  "token_data_sha256": "..."
}
```

No machine-bound paths are recorded. `tokenizer_id` is the logical identity;
`tokenizer_sha256` is what actually pins it.

Rules:

- `document_spans_sha256` and `n_docs` pin the spans file as part of dataset
  identity. The sampler depends on spans entirely (§5.1.2), so a `docs.npy` left
  over from a previous build would silently restore cross-document sampling
  without necessarily going out of bounds — reintroducing the exact bug §5.1.2
  exists to prevent. The tokenizer hash does not cover this.
- `token_data_sha256` pins the `.dat` content itself, which the spans hash does
  **not** imply. If a rebuilt corpus happens to produce the same document length
  distribution, its spans are byte-identical while the token data is entirely
  different, and stale metadata would pass every other check. Pinning spans is
  not pinning content. The digest is accumulated incrementally during the build
  (§5.1.3) rather than by re-reading the finished file.

- `dtype` is derived **at build time** and recorded. Readers use
  `np.dtype(meta["dtype"])`. It is never re-derived at read time — if a tokenizer
  changes later, a re-derivation would read an old `.dat` with the wrong width
  and silently misalign every token.
- Metadata, not `LCMConfig`, is the authority for `model_vocab_size`, which is
  what sizes `E`. `token_id_max` sizes storage; the two are different numbers and
  must stay different fields.
- `tokenizer_sha256` is the only check that catches the current class of bug.
  `max(token_id) < model_vocab_size` catches nothing: all 30k ids are below
  151936 and look entirely legal.

### 4.1 Enforcement (fatal, not a warning)

At cognitive-training startup, before any parameter is initialized:

```python
meta = load_dataset_meta(shape_path)
entry = TOKENIZER_REGISTRY[meta["tokenizer_id"]]

# Cheap size invariant, always on: catches a truncated or replaced .dat
# without reading it.
itemsize = np.dtype(meta["dtype"]).itemsize
if os.path.getsize(data_path) != meta["n_tokens"] * itemsize:
    raise ValueError(...)

# Full content verification re-reads the whole corpus, so it is opt-in:
# always for production runs, skippable for quick local starts.
if full_verify:
    if sha256_file(data_path) != meta["token_data_sha256"]:
        raise ValueError(...)

# Metadata is the authority for the tensor extent. This step deliberately does
# NOT require a Qwen artifact — vocabulary ownership is decoupled from the
# active backend (§2), so a student-only run must be able to start.
cfg = dataclasses.replace(cfg, vocab_size=meta["model_vocab_size"])

spec = resolve_tokenizer_spec(
    path=tokenizer_path_for(meta["tokenizer_id"]),   # $LCM_DATA_ROOT/... or --tokenizer
    document_separator=entry["separator"],
    model_vocab_size=meta["model_vocab_size"],
    tokenizer_id=meta["tokenizer_id"],
)

if spec.sha256 != meta["tokenizer_sha256"]:
    raise ValueError(...)   # tokenizer file changed since the corpus was built
if spec.document_separator_id != meta["separator_id"]:
    raise ValueError(...)
if spec.max_token_id != meta["token_id_max"]:
    raise ValueError(...)

# The tokenizer must not be able to emit an id outside the model's matrices,
# or train/qwen_lm.py:176 gathers out of bounds.
if spec.max_token_id >= spec.model_vocab_size:
    raise ValueError(...)

# Spans are dataset identity, not a derived cache (§4). validate_spans does the
# hash check, the endpoint checks, and the strict-partition check.
spans = validate_spans(spans_path, meta)

tokens = np.memmap(data_path, dtype=np.dtype(meta["dtype"]), mode="r",
                   shape=(meta["n_tokens"],))

# Structure does not imply semantics: a builder that wrote wrong-but-contiguous
# offsets passes every partition check above. Verify the boundary set itself.
if np.any(tokens[spans[:, 1] - 1] != meta["separator_id"]):
    raise ValueError(...)   # some span does not end on the separator

if full_verify:
    # The separator must be a unique boundary marker: exactly once per
    # document, and nowhere else.
    if int(np.count_nonzero(tokens == meta["separator_id"])) != meta["n_docs"]:
        raise ValueError(...)
```

Three layers, each catching what the others cannot: the **hash** catches a
foreign or stale file; the **partition checks** catch a builder that wrote bad
offsets into a file whose own hash is perfectly valid; and the **separator
check** catches a boundary set that is structurally continuous but not actually
on document boundaries. `tokens[spans[:, 1] - 1]` is cheap despite being scattered
reads, because the positions are sorted and the memmap pages are read in order.

`full_verify` is a run mode, not a heuristic: production training runs enable it,
quick local starts may disable it and rely on the always-on size invariant. The
separator-count check sits behind it because counting over the whole corpus is a
full pass.

Resolving the path from `tokenizer_id` (rather than only accepting `--tokenizer`)
is what makes the hash check automatic: a run cannot skip verification by
omitting a flag.

`LCMConfig.vocab_size` means the *tensor extent* from here on, and is the row
count of `E`. It is the one place where "vocab size" legitimately means
`model_vocab_size` rather than a tokenizer property.

There is no warning path. Any run whose tokenizer identity cannot be verified is
aborted, because no active-channel metric from such a run is interpretable.

### 4.2 Tied table and active-backend extent invariant

§4.1 proves registry ↔ dataset metadata agreement. Two things remain unproven:
that the tied table was built at the metadata's extent, and that the active
backend's tensors agree with it.

**Tied table**, unconditionally — after `init_cog_params` returns, on both the
fresh and the resume path:

```python
E = params["encoder"]["embed"]
if E.shape != (meta["model_vocab_size"], cfg.d_model):
    raise ValueError(...)
```

This is an assertion, not a derivation. §4.1 sets `cfg.vocab_size` from metadata
and the encoder is initialized from it, so the shapes *should* agree — but this
edge is load-bearing, and the resume path can carry a checkpoint built under a
different config. A load-bearing invariant must not rest on "should".

**Active backend**, only when the backend exposes a vocabulary extent. Qwen does,
so this runs under `use_qwen`; a student-only run has no such extent and must not
be forced to load a Qwen artifact to start. In `_load_qwen_checkpoint`
(`train/cog_train.py:124-143`), where `params['qwen']` is populated:

```python
if use_qwen:
    qwen = params["qwen"]
    qwen_embed_vocab = qwen["model.embed_tokens.weight"].shape[0]
    lm_weight = qwen.get("lm_head.weight", qwen["model.embed_tokens.weight"])
    qwen_logit_vocab = lm_weight.shape[0]

    if qwen_embed_vocab != meta["model_vocab_size"]:
        raise ValueError(...)
    if qwen_logit_vocab != meta["model_vocab_size"]:
        raise ValueError(...)
```

The `lm_head` fallback mirrors `train/qwen_lm.py:220`, where the head is
weight-tied to the embedding when `lm_head.weight` is absent. An `lm_head`-less
checkpoint with matching embedding rows is valid and must pass.

The full invariant. Every edge holds, or training stops:

```
max_token_id
  <  dataset.model_vocab_size
  == E.shape[0]                              (always)
  == qwen embed_tokens.shape[0]              (use_qwen only)
  == qwen lm_head.shape[0]                   (use_qwen only)
```

Each edge closes a different hole: the first prevents an out-of-bounds gather at
`train/qwen_lm.py:176`; the second keeps the passive readout aligned with the
tied table; the last two catch a swapped or differently-sized Qwen checkpoint
that would silently invalidate the corpus.

The conditionality is what makes §2's "vocabulary ownership decoupled from the
active-channel backend" true rather than aspirational: metadata alone suffices to
size `E`, and a backend is only asked to justify itself if it has a vocabulary to
justify.

## 5. Changes by file

### 5.1 `train/data.py`

- Remove module-level `TOKENIZER_PATH` / `MMAP_PATH` / `MMAP_SHAPE_PATH` as
  implicit globals; paths become explicit parameters.
- `tokenize_and_mmap(...)` takes a `TokenizerSpec`. It writes
  `article_tokens + [spec.document_separator_id]` per article, dropping the
  `[BOS] ... [EOS]` construction at `:101-103`, `:125-139`, `:185-186`, `:208-215`.
- Encode with `tokenizer.encode(text, add_special_tokens=False)`, then append
  `[document_separator_id]` explicitly. Corpus framing belongs to this pipeline
  and must not depend on whatever post-processor the tokenizer ships.
- Reject any article whose encoded payload already contains the separator, so the
  separator stays a *unique* boundary marker — the property §4.1's
  `count(separator) == n_docs` check asserts:

  ```python
  if spec.document_separator_id in article_ids:
      raise ValueError("separator occurred inside document payload")
  ```

  `<|endoftext|>` as literal article text is vanishingly unlikely in zhwiki, but
  it must fail loudly rather than silently manufacture a phantom document
  boundary.
- Token-count estimation per line becomes `len(encoded.ids) + 1`.
- Write the full metadata object from §4 (`:145-146`).
- `WikiDataIter.__init__` (`:242-252`) reads `dtype` from metadata instead of the
  hardcoded `np.uint16` at `:251`; `TextLineIter._build` (`:215`) likewise.
- `build_dataset` (`:280-317`) threads the spec through.

Uniform separator rule changes the legacy 30k format as well (no leading BOS).
Accepted: the legacy corpus is not rebuilt in this change, and no valid
checkpoint depends on it.

#### 5.1.1 Document spans

The build additionally emits `zhwiki_qwen_docs.npy`: an `(n_docs, 2)` int64 array
of `[start, end)` offsets, one row per article.

A span **includes its trailing separator**, so `tokens[end-1] == separator_id`.
This is deliberate: it lets a window end on the separator, which is how the model
learns to *predict* `<|endoftext|>` — the boundary that just became semantically
load-bearing. Defining the span to exclude the separator would silently make the
separator unpredictable.

One consequence to accept: under document-bounded sampling the separator is never
*in context*, only ever a predicted target. A window holding a separator as a
non-final token would necessarily cross documents, which §5.1.2 forbids. The model
therefore learns "documents end here" from document-final content, but never sees
`<|endoftext|>` followed by a fresh document inside one window. This is inherent
to the P0 constraint, not an oversight.

#### 5.1.2 Document-bounded sampling (P0)

`WikiDataIter` currently draws `start = randint(0, n_tokens - N - 1)` from the
whole array (`:257-267`), so a window can straddle a boundary and pair article A's
context with article B's generation. Standard causal-LM packing tolerates this,
because each token's loss depends only on its own prefix. **C' cannot.**

`z_q = f(context)` is a global summary, and `active_z_margin`
(`train/cog_train.py:459-461`) *forces* the generation segment to depend on it.
If that segment is an unrelated article, the requirement is unsatisfiable and the
hinge injects noise directly into the cognitive state — the original bug's failure
mode arriving by a different route.

Sampling becomes document-bounded:

```python
lengths = spans[:, 1] - spans[:, 0]
usable  = lengths >= (N + 1)                 # N inputs + 1 shifted target
w       = np.where(usable, lengths - N, 0)   # valid start positions per document
```

Windows are drawn from `[start, end - (N + 1)]` within one span, and documents are
sampled with probability proportional to `w` rather than uniformly — so every
valid *window* is equally likely, not every *document*.

Articles with `L < N + 1` are skipped at iteration time, not filtered at build
time: `N` comes from `--cog-seq` and varies per run, so a build-time filter would
have to guess it. Their tokens stay on disk, which is acceptable — they are short
by construction.

#### 5.1.3 Atomic build

The build writes temporary paths and renames only once every artifact is complete
and flushed:

```
zhwiki_qwen.dat.tmp
zhwiki_qwen_docs.tmp.npy
zhwiki_qwen_shape.tmp.json
```

Note the temp-name ordering for the `.npy`: `np.save("x.npy.tmp", arr)` silently
appends `.npy` and writes `x.npy.tmp.npy`. Either name the temp file
`*.tmp.npy`, or pass a file object — `with open(tmp, "wb") as f: np.save(f, spans)`.

`token_data_sha256` is accumulated incrementally while the token bytes are
written, so pinning the `.dat` costs nothing extra at build time and requires no
second pass over a multi-gigabyte file.

Rename the two data files, then write **metadata last**. An interrupted build
cannot then leave a new `.dat` paired with old spans and new metadata. If the
process dies between the data renames and the metadata write, startup raises on
the hash check instead of training on inconsistent boundaries — failing closed is
the point.

The builder must also never emit a partial trailing article. When `max_tokens`
truncation would cut mid-article, that article is dropped and `n_tokens` shortened
to the previous separator, so §4.1's partition invariant holds exactly and every
span still ends with its separator.

Durability hardening, optional and orthogonal to model correctness: `fsync` each
temp file before its rename and `fsync` the containing directory after, so the
rename ordering survives power loss and not merely a process crash. Correctness
rests on the §4.1 startup checks either way; this only narrows the window in which
they fire.

### 5.2 `train/config.py`

`vocab_size: int = 30000` (`:10`) stays as a default but is no longer
authoritative under cognitive training — it is replaced from dataset metadata in
§4.1. `LCMConfig` is a frozen dataclass, so this is done with
`dataclasses.replace`, not `__post_init__` mutation.

`use_qwen` (`:23`) keeps its current meaning and gains no vocabulary implication.

### 5.3 `train/cog_train.py` — tied readout

One trainable token table. With `E = params['encoder']['embed']`, shape
`(model_vocab_size, d)` — the full tensor extent, so it lines up with Qwen's
151936-wide logits:

```
encoder:  E[token_ids]
passive:  logits = z_q @ E.T
```

- Delete the `W_out` initialization at `:187`.
- Delete the `W_out` resume fallback at `:168-169`.
- Rewrite the passive readout at `:399`:
  `p_logits = jnp.einsum('bsd,vd->bsv', z_qs, p['encoder']['embed'])`
- `init_cog_params` docstring and any remaining `W_out` references updated.
- No `W_out` optimizer leaf exists, so no second Adam state.

### 5.4 `train/cog_train.py` — export

The C engine itself is unchanged (`infer/lcm.h` has no vocabulary constant; the
engine is latent-space `LCM_D` only). The **export and the Python inference
wrapper** are what change.

- `:786-787` derive `d` and `V` from `params['W_out']`. Source them from
  `params['encoder']['embed'].shape` instead, or the export raises `KeyError`.
- `:792` writes `'vocab_size': V` into the exported `config.json`; it now
  reports the tied table's vocabulary, which after this change is Qwen's.
- `:843` the `decoder.bin` fallback writes `W_out`. It writes `E.T` instead,
  preserving the existing `(d, V)` orientation the inference loader expects.

`E` is already written into `encoder.bin` first (`:817`), so `decoder.bin = E.T`
is **disk duplication only** — not a second training parameter and not a second
Adam state. Phase 1 keeps it deliberately, to avoid widening this fix.

- `:895-898` copies `data/tokenizer.json` into the checkpoint by probing hardcoded
  candidate paths. It becomes the resolved spec's path, so a checkpoint carries
  the tokenizer it was actually trained with.

`save_cog_checkpoint(params, output_dir, step, self_state)` has no way to reach
the spec, so the signature changes:

```python
@dataclasses.dataclass(frozen=True)
class RunIdentity:
    tokenizer_spec: TokenizerSpec
    dataset_meta: dict


def save_cog_checkpoint(params, output_dir, step, *, run_identity, cfg=None,
                        self_state=None):
    ...
```

`run_identity` is **required**, not optional. If it can be omitted, an
implementer will omit it, and the export will fall back to probing a global
`data/tokenizer.json` — precisely the bug class this spec exists to kill. A
missing argument should surface as a `TypeError`, not as a silently wrong
checkpoint. `tokenizer_spec` is the load-bearing field; `dataset_meta` rides
along so the checkpoint also records the corpus identity it was trained on.

## 6. Storage layout

All data moves off the root filesystem, which is currently full. Root keeps the
repo and the environment only.

```
/home/DslsDZC/data/lcm/          # /dev/sdc3, btrfs compress=zstd:1
├── raw/
│   └── zhwiki_cleaner.txt
├── tokenizers/
│   └── qwen2.5-0.5b/
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── mmap/
│   ├── zhwiki_qwen.dat
│   ├── zhwiki_qwen_shape.json
│   └── zhwiki_qwen_docs.npy
├── models/
│   └── qwen2.5-0.5b/
│       ├── config.json          # source of model_vocab_size (§4.1)
│       └── qwen_params.npz
└── checkpoints/
```

The repo keeps config and small files; paths point to this tree explicitly.

Environment (fish):

```fish
set -x HF_HOME                  /home/DslsDZC/data/hf
set -x HF_HUB_CACHE             /home/DslsDZC/data/hf/hub
set -x JAX_COMPILATION_CACHE_DIR /home/DslsDZC/data/jax-cache
set -x TMPDIR                   /home/DslsDZC/data/tmp
```

Sizing: `uint32` doubles bytes versus `uint16` at equal token count. The token
count itself will also move, because a domain-trained 30k BPE may be more
efficient on zhwiki than Qwen's general-purpose 152k vocabulary. Budget
headroom above `2 x current size` before running the build. The spans array is
negligible by comparison (`n_docs x 16` bytes).

Note that `n_tokens` overstates the *training* tokens: document-bounded sampling
(§5.1.2) discards windows near boundaries and skips articles shorter than `N + 1`
entirely. Effective tokens are lower, and the shortfall grows as `--cog-seq`
rises.

If `/dev/sdc3` is a spinning disk, that is acceptable: this machine does
preprocessing and saving. Cloud training on the 5090 should upload the finished
mmap to instance-local SSD/NVMe first. Do not let `WikiDataIter` do random reads
against a remote-mounted HDD.

## 7. Verification

Tests (all require a working shell, currently blocked — see §9):

1. `TokenizerSpec` resolution: `max_token_id` is computed with
   `with_added_tokens=True`; `dtype` is derived from `max_token_id`, not from
   `model_vocab_size`; separator resolves by name; `sha256` is stable. The 65535
   boundary needs a synthetic tokenizer — Qwen sits far above it, so the real
   file cannot exercise `uint16`.
2. Corpus round-trip: build a 3-article corpus, read it back through
   `WikiDataIter`, assert ids are identical and that `dtype` came from metadata
   rather than being re-derived.
3. Metadata guard: tamper `tokenizer_sha256`, then `model_vocab_size`, then
   `token_id_max`, then `token_data_sha256`, then `n_tokens` — each must raise,
   not warn, at cog-train startup. The `n_tokens` case must be caught by the
   always-on size invariant, with no full read of the corpus. Also assert the
   `max_token_id < model_vocab_size` invariant fires when violated.
4. Separator: exactly one separator per article, no leading BOS; an article whose
   payload contains the separator is rejected at build time; and every span ends
   on the separator, `tokens[spans[:, 1] - 1] == separator_id`.
5. Tied readout: `'W_out' not in params`; `E.shape == (model_vocab_size, d)`;
   `einsum('bsd,vd->bsv', z, E)` equals `z @ E.T`.
6. Export: `decoder.bin` has shape `(d, model_vocab_size)` and equals the leading
   `E` block of `encoder.bin`, transposed; exported `config.json` reports
   `model_vocab_size`; omitting `run_identity` raises `TypeError`.
7. Document-bounded sampling: draw a large number of random batches and assert
   every window lies inside a single span —
   `not crosses_document_boundary(start, N + 1)` — with the independent
   cross-check that the window `[s, s+N]` holds at most one separator and, if
   present, it is the final element. Also assert no document with `L < N + 1` is
   ever sampled, and that the empirical document frequency is proportional to
   `L - N`.

   The separator lands in the *target*, never in `inputs`: a window's highest
   input index is `s + N - 1`, and §5.1.2 permits `s <= end - N - 1`, so
   `s + N - 1 <= end - 2`, while the separator sits at `end - 1`. An input window
   shaped `[A B C SEP]` is therefore not producible under the sampling rule, and
   "no separator in `inputs`" holds. The at-most-once / final-element form above
   is the stronger assertion and is preferred, because it constrains the whole
   `N+1` window rather than only its input half.
8. Qwen tensor invariant, four cases:
   - A: `embed_tokens` rows != `meta["model_vocab_size"]` → raise.
   - B: explicit `lm_head.weight` present, rows != metadata → raise.
   - C: `lm_head.weight` absent, `embed_tokens` rows == metadata → **pass**,
     through the tied fallback that mirrors `train/qwen_lm.py:220`.
   - D: `embed_tokens` rows == `lm_head` rows == metadata → pass.

   Case C is the one that must not be inverted: asserting that an `lm_head`-less
   checkpoint raises would force breaking legitimate weight-tied Qwen
   checkpoints, which are exactly what the fallback exists to support.
9. Spans identity: tamper `document_spans_sha256`, then `n_docs`, then the spans
   contents — each must raise. Separately, a spans array that is valid but not a
   strict partition must raise on the partition check rather than on the hash.
   Finally, a spans array that **is** a valid partition but whose boundaries do
   not land on separators must raise on the separator check — the case structure
   alone provably cannot catch.
10. Backend conditionality: a stub run with `use_qwen=False` and no Qwen
    `config.json` on disk must start successfully, proving §4.1 does not demand a
    Qwen artifact. A run whose `E` has the wrong row count must raise on §4.2's
    tied-table assertion, on both the fresh and the resume path.

The load-bearing checks are (3), (7), (9), and (10). Without (3), the next
tokenizer or corpus change reproduces this bug silently. Without (7), correctness
of the token space buys nothing, because the cognitive state is still supervised
against unrelated text. Without (9), a stale or wrong-boundary spans file
restores (7)'s failure without tripping any other guard. Without (10), the
decoupling §2 promises is untested, and the `E` assertion — the one edge that
ties metadata to the actual tied table — is never exercised.

## 8. Rollout

- No migration. Existing checkpoints are discarded (accepted decision).
- Phase 1 (this spec): export writes `decoder.bin = E.T` for compatibility.
- Phase 2 (deferred, out of scope): the inference loader constructs the tied
  readout directly from the encoder embedding and the duplicate file is dropped.
  Do not do this in the same change.

## 9. Open items

- **Root filesystem is full.** The shell is unusable, so nothing here can be run,
  tested, or committed until space is freed. The harness appears to use a fixed
  `/tmp/claude-1000` path, so setting `TMPDIR` may not unblock it.
- **Qwen tokenizer is not on disk.** `checkpoints/qwen_model/config.json` is an
  HTTP redirect body, and `tokenizer_config.json` is absent. A clean download of
  `tokenizer.json` + `tokenizer_config.json` is a hard prerequisite for the
  corpus build.
- **`max_token_id` is unmeasured.** The tokenizer file is not on this machine, so
  `token_id_max` in §4's example is illustrative. It is read from the file at
  resolve time, which is the point — but the first build must confirm it stays
  below `model_vocab_size`. The invariant is asserted in §4.1, since a tokenizer
  emitting an id at or above 151936 would gather out of bounds at
  `train/qwen_lm.py:176`.
- **Separator name must exist in the downloaded file.** Resolution is by name, so
  the id cannot go stale and a rename surfaces as a resolution error. Qwen2.5-0.5B
  base sets both `bos_token_id` and `eos_token_id` to 151643, which is
  `<|endoftext|>`, so the name should resolve; the values quoted here come from
  the model cards, not from a file on this machine.
- **`gen_head`.** The from-scratch path (`init_cog_params:172-187`) never creates
  it, so a fresh Qwen run has none; the export's `gen_head` branch (`:833-841`)
  and the legacy wanted-list (`:77-79`) are dead on this path. Confirm nothing
  else assumes it before removing anything.
- **Final token count** is unknown until the build runs; see §6 sizing.
