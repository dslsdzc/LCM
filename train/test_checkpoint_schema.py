"""Checkpoint/export schema v2: 32-byte header, manifest, canonical hash.

R10 freezes the schema: old 24-byte headers are rejected, not migrated.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_checkpoint_schema.py -v
"""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from train.bin_format import (
    FORMAT_VERSION, HEADER_FMT_V2, HEADER_SIZE_V2,
    canonical_sha256, save_manifest, load_manifest, read_bin_v2,
)


def test_header_v2_roundtrip(tmp_path):
    # Literal 32, not derived from the format string: a format edit must not
    # silently redefine the frozen header size.
    assert HEADER_SIZE_V2 == 32
    hdr = struct.pack(HEADER_FMT_V2, 512, 256, 1, 0, 1.0, FORMAT_VERSION, 0, 0)
    assert len(hdr) == 32
    assert HEADER_SIZE_V2 == len(hdr)
    assert struct.unpack(HEADER_FMT_V2, hdr)[5] == 2


def test_canonical_sha256_is_order_independent():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((4,), dtype=np.float32)
    h1 = canonical_sha256({'b': b, 'a': a})
    h2 = canonical_sha256({'a': a, 'b': b})
    assert h1 == h2


def test_canonical_sha256_is_shape_sensitive():
    """Same bytes, different shape → different hash (shape is part of it)."""
    flat = np.arange(6, dtype=np.float32)
    assert canonical_sha256({'a': flat}) != canonical_sha256(
        {'a': flat.reshape(2, 3)})


def test_manifest_roundtrip(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_manifest(str(tmp_path), {'a': a})
    m = load_manifest(str(tmp_path))
    assert m['format_version'] == FORMAT_VERSION
    entry = m['tensors']['a']
    assert entry['shape'] == [2, 3]
    assert entry['dtype'] == 'float32'
    assert entry['length'] == int(np.prod(entry['shape'])) * 4
    assert entry['sha256'] == canonical_sha256({'a': a})


def test_manifest_rejects_non_float32(tmp_path):
    with pytest.raises(TypeError, match='float32'):
        save_manifest(str(tmp_path), {'a': np.zeros(2, dtype=np.float64)})


def test_load_manifest_rejects_bad_format_version(tmp_path):
    save_manifest(str(tmp_path), {'a': np.zeros(2, dtype=np.float32)})
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest['format_version'] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="old checkpoint format"):
        load_manifest(str(tmp_path))


def test_old_header_rejected(tmp_path):
    p = tmp_path / "old.bin"
    p.write_bytes(b'\x00' * 24)
    with pytest.raises(ValueError, match="old checkpoint format"):
        read_bin_v2(str(p))


def test_truncated_v2_header_is_not_reported_as_old(tmp_path):
    p = tmp_path / "truncated.bin"
    p.write_bytes(b'\x00' * 28)
    with pytest.raises(ValueError, match="truncated v2 header"):
        read_bin_v2(str(p))


def test_engine_requires_config_format_v2(tmp_path):
    """LCMInferEngine refuses a checkpoint whose config.json predates v2."""
    import lcm

    (tmp_path / "config.json").write_text(json.dumps({"d_model": 8}))
    with pytest.raises(SystemExit, match="format_version"):
        lcm.LCMInferEngine(str(tmp_path), liblcm_path=str(tmp_path / "no.so"))


def test_load_checkpoint_rejects_v1_config(tmp_path):
    """checkpoint.load_checkpoint stops before the missing-file WARN path."""
    from train.checkpoint import load_checkpoint

    (tmp_path / "config.json").write_text(json.dumps({"d_model": 8}))
    with pytest.raises(ValueError, match="old checkpoint format"):
        load_checkpoint(str(tmp_path), load_opt=False)


def test_bin_format_and_lcm_import_without_jax():
    """The C inference path must not depend on JAX (schema module is JAX-free)."""
    import subprocess
    import sys

    code = ("import sys; sys.modules['jax'] = None; "
            "import train.bin_format, lcm; "
            "assert lcm._HAS_JAX is False")
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=str(Path(__file__).parents[1]))
