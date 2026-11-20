"""Checkpoint binary format (schema v2) — stdlib + numpy only.

Deliberately free of JAX imports: ``lcm.py`` can run the C inference path
without JAX, so its codebook reader must not drag in ``train.checkpoint``.

Header layout (32 bytes)::

    <iiiifIII
    M(int32), d(int32), n_layers(int32), type(int32), c(float32),
    format_version(uint32), reserved(uint32), checksum(uint32)

Data is a float32 payload; frozen modules (gvalue/danger) append a 32-byte
SHA-256. Schema v1 (24-byte header) is rejected, never migrated (R10).
"""
import hashlib
import json
import os
import struct
import zlib

import numpy as np

HEADER_FMT_V2 = '<iiiifIII'
HEADER_SIZE_V2 = struct.calcsize(HEADER_FMT_V2)
FORMAT_VERSION = 2
OLD_FORMAT_ERROR = "old checkpoint format; re-export required"
MANIFEST_NAME = "manifest.json"

# Frozen contract (R10): the format is 32 bytes, not whatever calcsize says
# after an accidental edit.
assert HEADER_SIZE_V2 == 32, HEADER_SIZE_V2


def _compute_checksum(data):
    """CRC32 of data section (uint32)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def _canonical_tensor_bytes(array):
    """Canonical serialization of one tensor (R10).

    float32 little-endian, with the shape as an int32[ndim] prefix. No names,
    paths or timestamps: two runs that produced the same weights hash equal.
    """
    a = np.ascontiguousarray(np.asarray(array), dtype='<f4')
    prefix = np.asarray(a.ndim, dtype='<i4').tobytes()
    prefix += np.asarray(a.shape, dtype='<i4').tobytes()
    return prefix + a.tobytes()


def canonical_sha256(tensors):
    """SHA-256 over tensors in name order, each in canonical form."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(_canonical_tensor_bytes(tensors[name]))
    return h.hexdigest()


def _normalise_file_ref(ref):
    """Normalise a ``file_map`` value to ``(filename, byte_offset)``.

    Accepted forms: ``"x.bin"``, ``("x.bin", 32)``, or
    ``{"file": "x.bin", "offset": 32}``. The offset is the byte position of
    the tensor payload inside the file (32 for header-bearing v2 codebooks).
    """
    if isinstance(ref, str):
        return ref, 0
    if isinstance(ref, (tuple, list)) and len(ref) == 2:
        return str(ref[0]), int(ref[1])
    if isinstance(ref, dict) and 'file' in ref:
        return str(ref['file']), int(ref.get('offset', 0))
    raise TypeError(f"bad file_map entry {ref!r}; expected str/(name, off)/dict")


def save_manifest(output_dir, tensors, file_map=None):
    """Write manifest.json describing every tensor in canonical form.

    Tensors must already be float32: silently casting a float64 tensor would
    record a dtype the caller did not have.

    ``canonical_offset`` always locates the tensor in the canonical float32
    stream (name-sorted, concatenated) used by ``canonical_sha256``.
    ``canonical_sha256`` itself stays name-sorted and independent of files.

    ``file_map`` is optional and maps a logical tensor name to the ``.bin``
    file (and byte offset) that actually stores its payload. When supplied,
    the entry gains a ``file`` field and ``offset`` becomes the byte offset
    inside that file (Ruling 9); ``canonical_offset`` keeps the stream
    position. Tensors without a mapping keep ``offset == canonical_offset``.
    """
    entries = {}
    canonical_offset = 0
    for name in sorted(tensors):
        source = np.asarray(tensors[name])
        if source.dtype != np.float32:
            raise TypeError(
                f"manifest tensor {name!r} must be float32, got {source.dtype}")
        a = np.ascontiguousarray(source, dtype='<f4')
        length = a.nbytes
        entry = {
            'name': name,
            'shape': [int(x) for x in a.shape],
            'dtype': 'float32',
            'offset': canonical_offset,
            'canonical_offset': canonical_offset,
            'length': length,
            'sha256': hashlib.sha256(_canonical_tensor_bytes(a)).hexdigest(),
        }
        if file_map is not None and name in file_map:
            filename, file_offset = _normalise_file_ref(file_map[name])
            entry['file'] = filename
            entry['offset'] = file_offset
        entries[name] = entry
        canonical_offset += length
    manifest = {'format_version': FORMAT_VERSION, 'tensors': entries}
    path = os.path.join(output_dir, MANIFEST_NAME)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  [CKPT] {MANIFEST_NAME} ({len(entries)} tensors)")


def load_manifest(output_dir):
    """Read manifest.json, rejecting anything that is not schema v2."""
    path = os.path.join(output_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        raise ValueError(f"{path} missing; {OLD_FORMAT_ERROR}")
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError(f"{path}: {OLD_FORMAT_ERROR}")
    return manifest


def read_bin_v2(path, has_hash=False):
    """Read a schema-v2 .bin file → (header_dict, float32 array).

    A bare 24-byte v1 header is reported as old format; a partial 25..31 byte
    header is reported as a truncated v2 header. Neither is ever migrated.
    ``has_hash`` strips the trailing 32-byte SHA-256 used by the frozen
    gvalue/danger codebooks before checksumming.
    """
    with open(path, 'rb') as f:
        hdr_bytes = f.read(HEADER_SIZE_V2)
        if len(hdr_bytes) <= 24:
            raise ValueError(f"{OLD_FORMAT_ERROR} ({path})")
        if len(hdr_bytes) < HEADER_SIZE_V2:
            raise ValueError(
                f"truncated v2 header ({len(hdr_bytes)} of "
                f"{HEADER_SIZE_V2} bytes); re-export required ({path})")
        (M, d, n_layers, cb_type, c, version, _reserved,
         stored_crc) = struct.unpack(HEADER_FMT_V2, hdr_bytes)
        if version != FORMAT_VERSION:
            raise ValueError(f"{OLD_FORMAT_ERROR} ({path})")
        data_bytes = f.read()

    if has_hash:
        if len(data_bytes) < 32:
            raise ValueError(f"{path}: truncated SHA-256 suffix; re-export required")
        data_bytes = data_bytes[:-32]

    actual_crc = _compute_checksum(data_bytes)
    if actual_crc != stored_crc:
        raise ValueError(
            f"CRC mismatch in {path}: stored={stored_crc:#010x}, "
            f"actual={actual_crc:#010x} — file is corrupt or was written "
            f"by an old writer; re-export the checkpoint")

    header = {'M': M, 'd': d, 'n_layers': n_layers, 'type': cb_type,
              'c': c, 'format_version': version}
    return header, np.frombuffer(data_bytes, dtype=np.float32)
