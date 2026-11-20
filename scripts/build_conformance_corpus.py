#!/usr/bin/env python3
"""Build the compliant ≤100-step conformance corpus.

The old ``data/zhwiki_tokens.dat`` carries only ``n_tokens`` in its shape file;
``resolve_dataset_identity`` rightly rejects it. This script produces the three
files the smoke consumes: ``.dat``, ``_shape.json`` and ``_docs.npy``.

It also installs the tokenizer at the registry path
``$LCM_DATA_ROOT/tokenizers/bpe30k/tokenizer.json`` (copy only; the source file
is never modified), so the default ``resolve_dataset_identity`` path works.

Run with the project Python, e.g.:

    JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python \
        scripts/build_conformance_corpus.py
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Make the repo root importable when run as ``python scripts/...``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.data import build_corpus  # noqa: E402
from train.dataset_meta import load_dataset_meta  # noqa: E402
from train.tokenizer_spec import resolve_tokenizer_spec  # noqa: E402


def _extract(source, target, target_bytes, max_lines):
    lines = []
    n = 0
    with open(source, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            text = line.strip()
            if not text:
                continue
            lines.append(text)
            n += len(text.encode("utf-8"))
            if target_bytes and n >= target_bytes:
                break
    if not lines:
        raise SystemExit(f"{source} produced no non-empty lines")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines), n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="/home/DslsDZC/LCM/zhwiki_final.txt")
    p.add_argument("--tokenizer", default="/home/DslsDZC/LCM/data/tokenizer.json")
    p.add_argument("--data-root", default=os.environ.get(
        "LCM_DATA_ROOT", "/home/DslsDZC/data/lcm"))
    p.add_argument("--output-dir", default=(
        "/home/DslsDZC/data/tmp/opencode/conformance_corpus"))
    p.add_argument("--name", default="conformance")
    p.add_argument("--target-bytes", type=int, default=220_000)
    p.add_argument("--max-lines", type=int, default=2000)
    args = p.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"source not found: {source}")
    tokenizer_src = Path(args.tokenizer)
    if not tokenizer_src.exists():
        raise SystemExit(f"tokenizer not found: {tokenizer_src}")

    registry_path = Path(args.data_root) / "tokenizers" / "bpe30k" / "tokenizer.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.resolve() != tokenizer_src.resolve():
        shutil.copy2(tokenizer_src, registry_path)
    print(f"[corpus] tokenizer -> {registry_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "corpus.txt"
    n_lines, n_bytes = _extract(
        source, text_path, args.target_bytes, args.max_lines)
    print(f"[corpus] text: {n_lines} lines, {n_bytes} bytes -> {text_path}")

    spec = resolve_tokenizer_spec(
        path=str(registry_path),
        document_separator="[EOS]",
        model_vocab_size=30000,
        tokenizer_id="bpe30k",
    )
    data_path = output_dir / f"{args.name}.dat"
    shape_path = output_dir / f"{args.name}_shape.json"
    spans_path = output_dir / f"{args.name}_docs.npy"
    meta = build_corpus(
        str(text_path), spec, str(data_path), str(shape_path),
        str(spans_path), max_tokens=0)
    meta = load_dataset_meta(str(shape_path))
    print(f"[corpus] tokens={meta['n_tokens']} docs={meta['n_docs']} "
          f"dtype={meta['dtype']} separator={meta['separator_id']}")
    print(f"[corpus] data   -> {data_path}")
    print(f"[corpus] shape  -> {shape_path}")
    print(f"[corpus] spans  -> {spans_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
