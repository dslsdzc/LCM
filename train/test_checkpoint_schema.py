"""Checkpoint/export schema v2: 32-byte header, manifest, canonical hash.

R10 freezes the schema: old 24-byte headers are rejected, not migrated.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_checkpoint_schema.py -v
"""
import numpy as np
import pytest
from train.checkpoint import (
    FORMAT_VERSION, HEADER_FMT_V2, HEADER_SIZE_V2,
    canonical_sha256, save_manifest, load_manifest,
)


def test_header_v2_roundtrip(tmp_path):
    import struct
    hdr = struct.pack(HEADER_FMT_V2, 512, 256, 1, 0, 1.0, FORMAT_VERSION, 0)
    assert len(hdr) == HEADER_SIZE_V2
    assert struct.unpack(HEADER_FMT_V2, hdr)[5] == 2


def test_canonical_sha256_is_order_independent():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((4,), dtype=np.float32)
    h1 = canonical_sha256({'b': b, 'a': a})
    h2 = canonical_sha256({'a': a, 'b': b})
    assert h1 == h2


def test_manifest_roundtrip(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_manifest(str(tmp_path), {'a': a})
    m = load_manifest(str(tmp_path))
    assert m['format_version'] == FORMAT_VERSION
    assert m['tensors']['a']['shape'] == [2, 3]
    assert m['tensors']['a']['dtype'] == 'float32'


def test_old_header_rejected(tmp_path):
    p = tmp_path / "old.bin"
    p.write_bytes(b'\x00' * 24)
    from train.checkpoint import read_bin_v2
    with pytest.raises(ValueError, match="old checkpoint format"):
        read_bin_v2(str(p))


def test_engine_requires_config_format_v2(tmp_path):
    """LCMInferEngine refuses a checkpoint whose config.json predates v2."""
    import json
    import lcm

    (tmp_path / "config.json").write_text(json.dumps({"d_model": 8}))
    with pytest.raises(SystemExit, match="format_version"):
        lcm.LCMInferEngine(str(tmp_path), liblcm_path=str(tmp_path / "no.so"))
