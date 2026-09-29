"""LCM data pipeline — zhwiki preprocessing + streaming data loader.

Pipeline:
  1. extract_text() — stream XML, extract article text, save to .txt
  2. train_tokenizer() — train BPE tokenizer (vocab_size=30000)
  3. tokenize_and_mmap() — tokenize all text, write memmap array
  4. WikiDataIter — training iterator yielding (inputs, targets) batches

Usage:
    from train.data import build_dataset, WikiDataIter

    # One-time preprocessing
    build_dataset(cfg, force=False)

    # Training iterator
    data_iter = WikiDataIter(data_path, tokenizer_path, B=16, N=512)
    for inputs, targets in data_iter:
        ...
"""

import hashlib
import os
import sys
import json
import numpy as np
from tqdm import tqdm

from train.dataset_meta import save_dataset_meta
from train.tokenizer_spec import sha256_file

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEXT_PATH = os.path.join(PROJECT_ROOT, "zhwiki_cleaner.txt")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
MMAP_PATH = os.path.join(DATA_DIR, "zhwiki_tokens.dat")
MMAP_SHAPE_PATH = os.path.join(DATA_DIR, "zhwiki_tokens_shape.json")
# ── Step 1: BPE tokenizer training ─────────────────────────────────────────

def train_tokenizer(text_path: str = TEXT_PATH,
                    tokenizer_path: str = TOKENIZER_PATH,
                    vocab_size: int = 30000):
    """Train BPE tokenizer on extracted wiki text.

    Args:
        text_path: Path to extracted .txt (one article per line).
        tokenizer_path: Output path for trained tokenizer JSON.
        vocab_size: Vocabulary size (must match LCMConfig.vocab_size).
    """
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
    except ImportError:
        print("ERROR: tokenizers not installed. Run:")
        print("  pip install tokenizers")
        sys.exit(1)

    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=5,
        show_progress=True,
    )

    tokenizer.train([text_path], trainer)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path} (vocab_size={vocab_size})")

    # Quick test
    test_text = "北京是中国的首都"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"  Test: '{test_text}' → {encoded.ids[:10]}... → '{decoded}'")
    return tokenizer


# ── Step 3: Tokenize + memory-map ──────────────────────────────────────────

def tokenize_and_mmap(text_path: str = TEXT_PATH,
                      tokenizer_path: str = TOKENIZER_PATH,
                      mmap_path: str = MMAP_PATH,
                      shape_path: str = MMAP_SHAPE_PATH,
                      max_tokens: int = 0):
    """Tokenize all text and write as memory-mapped uint16 array.

    Args:
        text_path: Extracted .txt file (one article per line).
        tokenizer_path: Trained tokenizer JSON.
        mmap_path: Output .dat file for token array.
        shape_path: Output JSON with (n_tokens,) shape.
        max_tokens: 0 = all tokens.

    Returns:
        n_tokens: Total number of tokens.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    # First pass: count total tokens
    total_tokens = 0
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting tokens"):
            encoded = tokenizer.encode(line.strip())
            # +2 for BOS and EOS tokens
            total_tokens += len(encoded.ids) + 2
            if max_tokens > 0 and total_tokens >= max_tokens:
                break
    if max_tokens > 0:
        total_tokens = min(total_tokens, max_tokens)

    print(f"Total tokens: {total_tokens:,}")

    # Create memory-mapped array
    os.makedirs(os.path.dirname(mmap_path), exist_ok=True)
    tokens_mmap = np.memmap(mmap_path, dtype=np.uint16, mode='w+', shape=(total_tokens,))

    # Second pass: tokenize and write
    pos = 0
    bos_arr = np.array([bos_id], dtype=np.uint16)
    eos_arr = np.array([eos_id], dtype=np.uint16)

    with open(text_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Tokenizing"):
            if max_tokens > 0 and pos >= max_tokens:
                break
            encoded = tokenizer.encode(line.strip())
            ids = np.array(encoded.ids, dtype=np.uint16)
            # Write BOS + ids + EOS
            remaining = total_tokens - pos
            seg = np.concatenate([bos_arr, ids, eos_arr])
            if len(seg) > remaining:
                seg = seg[:remaining]
            tokens_mmap[pos:pos + len(seg)] = seg
            pos += len(seg)

    tokens_mmap.flush()

    # Save shape metadata
    with open(shape_path, 'w') as f:
        json.dump({'n_tokens': int(total_tokens)}, f)

    print(f"Token array saved: {mmap_path} ({total_tokens:,} tokens, "
          f"{os.path.getsize(mmap_path) / 1e9:.2f} GB)")
    return total_tokens


# ── Qwen-path corpus builder ───────────────────────────────────────────────
#
# Supersedes tokenize_and_mmap for the cognitive-training corpus. Two
# differences that matter:
#
#   * One pass, not two. The old builder tokenized the whole corpus once to
#     count, then again to write. On zhwiki that is the entire cost of the build
#     paid twice.
#   * Framing is `article + one separator`, with no synthetic BOS, and the
#     builder refuses to emit a partial trailing article. Both are what let
#     "every span ends on the separator" and "spans form a strict partition"
#     hold exactly rather than approximately.

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


# ── Step 4: Data iterator ──────────────────────────────────────────────────

class TextLineIter:
    """Online tokenizing data iterator for .txt or .jsonl input.

    Tokenizes text on the fly, accumulates into a token buffer,
    then yields random slices as (inputs, targets) batches.
    Slower than WikiDataIter but skips the preprocessing step.
    """

    def __init__(self, text_path: str, tokenizer_path: str = TOKENIZER_PATH,
                 B: int = 16, N: int = 512, max_tokens: int = 0):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.B = B
        self.N = N
        self._tokens = self._build(text_path, max_tokens)

    def _build(self, path, max_tokens):
        from tqdm import tqdm
        import json as _json
        # Detect jsonl
        _is_jsonl = False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        _is_jsonl = isinstance(_json.loads(line), dict)
                    except Exception:
                        _is_jsonl = False
                    break
        bos = self.tokenizer.token_to_id("[BOS]")
        eos = self.tokenizer.token_to_id("[EOS]")

        # Count lines
        n_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_lines += 1

        # Tokenize into a list
        all_ids = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="   tokenizing text", total=n_lines, unit="line"):
                line = line.strip()
                if not line:
                    continue
                if _is_jsonl:
                    obj = _json.loads(line)
                    line = obj.get("text", obj.get("content", ""))
                    if not line:
                        continue
                ids = self.tokenizer.encode(line).ids
                all_ids.append(bos)
                all_ids.extend(ids)
                all_ids.append(eos)
                if max_tokens > 0 and len(all_ids) >= max_tokens:
                    all_ids = all_ids[:max_tokens]
                    break

        arr = np.array(all_ids, dtype=np.uint16)
        print(f"   {len(arr):,} tokens loaded from {path}")
        return arr

    def __iter__(self):
        return self

    def __next__(self):
        max_start = len(self._tokens) - self.N - 1
        starts = np.random.randint(0, max_start, size=self.B)
        batch = np.stack([self._tokens[s:s + self.N] for s in starts])
        inputs = batch.astype(np.int32)
        targets = np.stack([
            self._tokens[s + 1:s + self.N + 1] for s in starts
        ]).astype(np.int32)
        return inputs, targets

class WikiDataIter:
    """Fast random-access data iterator over memory-mapped tokens.

    Yields (inputs, targets) tuples of shape (B, N).
    targets = inputs shifted left by 1 (autoregressive LM).

    Uses a random read head per sample for maximum efficiency —
    no sequential scan, no shuffling needed.
    """

    def __init__(self, mmap_path: str = MMAP_PATH,
                 shape_path: str = MMAP_SHAPE_PATH,
                 B: int = 16, N: int = 512):
        with open(shape_path) as f:
            meta = json.load(f)
        self.n_tokens = meta['n_tokens']
        self.B = B
        self.N = N
        # Memory-map the token array (read-only)
        self.tokens = np.memmap(mmap_path, dtype=np.uint16, mode='r',
                                shape=(self.n_tokens,))

    def __iter__(self):
        return self

    def __next__(self):
        """Return next batch: (inputs, targets) each (B, N)."""
        # Random offsets: ensure each segment fits within the array
        max_start = self.n_tokens - self.N - 1
        starts = np.random.randint(0, max_start, size=self.B)

        batch = np.stack([self.tokens[s:s + self.N] for s in starts])
        inputs = batch.astype(np.int32)
        targets = np.stack([
            self.tokens[s + 1:s + self.N + 1] for s in starts
        ]).astype(np.int32)

        return inputs, targets

    def get_num_batches(self, tokens_per_step: int = None):
        """Return approximate number of batches in one epoch."""
        if tokens_per_step is None:
            tokens_per_step = self.B * self.N
        return self.n_tokens // tokens_per_step


# ── Convenience: build everything ──────────────────────────────────────────

def build_dataset(cfg=None, force: bool = False, max_tokens: int = 0):
    """Run full preprocessing pipeline: tokenizer → mmap.

    Skips steps where output already exists (unless force=True).

    Args:
        cfg: Optional LCMConfig (used for vocab_size).
        force: Re-run even if output files exist.
        max_tokens: 0 = all tokens.

    Returns:
        (n_tokens, tokenizer_path, mmap_path)
    """
    vocab_size = cfg.vocab_size if cfg is not None else 30000

    # Step 1: Train tokenizer
    if not os.path.exists(TOKENIZER_PATH) or force:
        print("=" * 60)
        print("Step 2: Training BPE tokenizer...")
        print("=" * 60)
        train_tokenizer(vocab_size=vocab_size)
    else:
        print(f"[SKIP] {TOKENIZER_PATH} exists")

    # Step 3: Tokenize + mmap
    if not os.path.exists(MMAP_PATH) or not os.path.exists(MMAP_SHAPE_PATH) or force:
        print("=" * 60)
        print("Step 3: Tokenizing and creating memory-mapped array...")
        print("=" * 60)
        tokenize_and_mmap(max_tokens=max_tokens)
    else:
        print(f"[SKIP] {MMAP_PATH} exists")

    # Load metadata
    with open(MMAP_SHAPE_PATH) as f:
        meta = json.load(f)

    return meta['n_tokens'], TOKENIZER_PATH, MMAP_PATH


def count_tokens(tokenizer_path: str = TOKENIZER_PATH,
                 text_path: str = TEXT_PATH,
                 sample: str = "北京是中国的首都"):
    """Quick tokenizer sanity check."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode(sample)
    print(f"  Input: '{sample}'")
    print(f"  Tokens: {encoded.tokens}")
    print(f"  IDs: {encoded.ids}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Decoded: '{tokenizer.decode(encoded.ids)}'")
    return tokenizer


if __name__ == "__main__":
    # Run full pipeline
    from train.config import LCMConfig
    cfg = LCMConfig()
    build_dataset(cfg, force=False)
