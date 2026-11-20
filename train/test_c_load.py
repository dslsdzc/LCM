"""Focused tests for the schema-v2 full-lattice loader (task 7).

Builds a synthetic v2 checkpoint (manifest + .bin files) with *different*
layer markers, so a loader that reads only layer 0 or only block 0 fails.

Run:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_c_load.py -v
"""
import json
import struct
import zlib

import numpy as np
import pytest

from train.bin_format import (
    FORMAT_VERSION, HEADER_FMT_V2, HEADER_SIZE_V2, SELF_LAYOUT_TAG,
    canonical_sha256, load_manifest, save_manifest, tensor_sha256,
)
from lcm import load_lattice_memories


D = 8
M_TOP = 5
M_FINE = 4
N_FINE = 2
M_SPARSE = 6
M_LR = 4            # deliberately != d: M_lr comes from U[0].shape[0]
N_LR = 3
RANKS = (1, 2, 4)
R_MAX = 4
M_MAN = 3
T_DIM = 2
M_BIND = 4
N_BIND = 3
M_CONTRAST = 3
N_CONTRAST = 2
N_LATTICES = 6
N_SELF = 3


def _write_header_bin(path, M, d, n_layers, payload):
    data = np.ascontiguousarray(payload, dtype=np.float32).tobytes()
    hdr = struct.pack(HEADER_FMT_V2, int(M), int(d), int(n_layers), 1, 1.0,
                      FORMAT_VERSION, 0, zlib.crc32(data) & 0xFFFFFFFF)
    path.write_bytes(hdr + data)


def _cfg():
    return {
        'format_version': FORMAT_VERSION,
        'd_model': D,
        'M_top': M_TOP,
        'M_fine': M_FINE,
        'n_hrq_layers': N_FINE,
        'M_sparse': M_SPARSE,
        'M_lr': M_LR,
        'n_lr_layers': N_LR,
        'ranks': list(RANKS),
        'r_max': R_MAX,
        'M_man': M_MAN,
        't_dim': T_DIM,
        'M_bind': M_BIND,
        'n_bind_layers': N_BIND,
        'M_contrast': M_CONTRAST,
        'n_contrast_layers': N_CONTRAST,
        'n_lattices': N_LATTICES,
        'n_self_codes': N_SELF + 1,
        'n_self_rows': N_SELF + 1,
        'n_self_modes': N_SELF,
    }


def _build_fixture(tmp_path, ranks=RANKS):
    """Write a complete synthetic export and return the expected arrays."""
    exp = {}
    tensors = {}
    file_map = {}

    # HRQ: top + all fine layers.
    hrq_top = np.zeros((M_TOP, D), dtype=np.float32)
    hrq_top[:, 0] = 10.0
    hrq_fine = np.zeros((N_FINE, M_FINE, D), dtype=np.float32)
    for l in range(N_FINE):
        hrq_fine[l, :, 0] = 20.0 + l
    exp['hrq_top'] = hrq_top
    exp['hrq_fine'] = hrq_fine
    tensors['hrq_codebook'] = np.concatenate(
        [hrq_top.ravel(), hrq_fine.ravel()])
    _write_header_bin(tmp_path / 'hrq_codebook.bin', M_TOP, D, N_FINE,
                      tensors['hrq_codebook'])
    file_map['hrq_codebook'] = ('hrq_codebook.bin', HEADER_SIZE_V2)

    # Sparse: C then the zero_vec row (header M includes both).
    sparse_C = np.zeros((M_SPARSE, D), dtype=np.float32)
    sparse_C[:, 0] = 30.0
    sparse_zero = np.zeros((1, D), dtype=np.float32)
    sparse_zero[0, 1] = 0.125
    exp['sparse_C'] = sparse_C
    exp['sparse_zero'] = sparse_zero
    tensors['sparse_codebook'] = np.concatenate(
        [sparse_C.ravel(), sparse_zero.ravel()])
    _write_header_bin(tmp_path / 'sparse_codebook.bin', M_SPARSE + 1, D, 1,
                      tensors['sparse_codebook'])
    file_map['sparse_codebook'] = ('sparse_codebook.bin', HEADER_SIZE_V2)

    # LowRank: ragged U_0..U_{n-1} then shared V (d, r_max).
    parts = []
    U = np.zeros((len(ranks), M_LR, R_MAX), dtype=np.float32)
    for k, r_k in enumerate(ranks):
        U_k = np.zeros((M_LR, r_k), dtype=np.float32)
        U_k[:, 0] = 40.0 + k
        U[k, :, :r_k] = U_k
        parts.append(U_k.ravel())
    V = np.zeros((D, R_MAX), dtype=np.float32)
    V[:, 0] = 0.5
    V[:, 1] = 0.25
    parts.append(V.ravel())
    exp['lowrank_U'] = U
    exp['lowrank_V'] = V
    tensors['lowrank_codebook'] = np.concatenate(parts).astype(np.float32)
    (tmp_path / 'lowrank_codebook.bin').write_bytes(
        tensors['lowrank_codebook'].tobytes())
    file_map['lowrank_codebook'] = ('lowrank_codebook.bin', 0)

    # Manifold: C then T flattened as (M, d, t_dim).
    man_C = np.zeros((M_MAN, D), dtype=np.float32)
    man_C[:, 0] = 50.0
    man_T = np.zeros((M_MAN, D, T_DIM), dtype=np.float32)
    man_T[:, 0, 0] = 0.75
    man_T[:, 1, 1] = 0.5
    exp['manifold_C'] = man_C
    exp['manifold_T'] = man_T
    tensors['manifold_codebook'] = np.concatenate(
        [man_C.ravel(), man_T.ravel()])
    _write_header_bin(tmp_path / 'manifold_codebook.bin', M_MAN, D, 1,
                      tensors['manifold_codebook'])
    file_map['manifold_codebook'] = ('manifold_codebook.bin', HEADER_SIZE_V2)

    # Binding: [layer][key, val, bind].
    bind_blocks = np.zeros((N_BIND, 3, M_BIND, D), dtype=np.float32)
    for l in range(N_BIND):
        for family in range(3):
            bind_blocks[l, family, :, 0] = 60.0 + 10.0 * family + l
    exp['bind_key'] = bind_blocks[:, 0]
    exp['bind_val'] = bind_blocks[:, 1]
    exp['bind_bind'] = bind_blocks[:, 2]
    tensors['bind_codebook'] = bind_blocks.ravel()
    (tmp_path / 'bind_codebook.bin').write_bytes(
        tensors['bind_codebook'].tobytes())
    file_map['bind_codebook'] = ('bind_codebook.bin', 0)

    # Contrast: all C_a layers, then all C_b layers.
    C_a = np.zeros((N_CONTRAST, M_CONTRAST, D), dtype=np.float32)
    C_b = np.zeros((N_CONTRAST, M_CONTRAST, D), dtype=np.float32)
    for l in range(N_CONTRAST):
        C_a[l, :, 0] = 100.0 + l
        C_b[l, :, 0] = 200.0 + l
    exp['contrast_a'] = C_a
    exp['contrast_b'] = C_b
    tensors['contrast_codebook'] = np.concatenate(
        [C_a.ravel(), C_b.ravel()])
    (tmp_path / 'contrast_codebook.bin').write_bytes(
        tensors['contrast_codebook'].tobytes())
    file_map['contrast_codebook'] = ('contrast_codebook.bin', 0)

    # Route / fusion.
    route_C = np.zeros((N_LATTICES, D), dtype=np.float32)
    route_C[:, 0] = 0.1 * np.arange(N_LATTICES)
    route_W = np.zeros((D, N_LATTICES), dtype=np.float32)
    route_W[0, :] = 1.0
    exp['route_C'] = route_C
    exp['route_W'] = route_W
    tensors['route_codebook'] = np.concatenate(
        [route_C.ravel(), route_W.ravel()])
    (tmp_path / 'route_codebook.bin').write_bytes(
        tensors['route_codebook'].tobytes())
    file_map['route_codebook'] = ('route_codebook.bin', 0)

    fusion_alpha = np.linspace(0.5, 1.0, N_LATTICES).astype(np.float32)
    fusion_ln_scale = np.ones(D, dtype=np.float32) * 2.0
    fusion_ln_bias = np.zeros(D, dtype=np.float32)
    exp['fusion_alpha'] = fusion_alpha
    exp['fusion_ln_scale'] = fusion_ln_scale
    exp['fusion_ln_bias'] = fusion_ln_bias
    tensors['fusion'] = np.concatenate(
        [fusion_alpha, fusion_ln_scale, fusion_ln_bias])
    (tmp_path / 'fusion.bin').write_bytes(tensors['fusion'].tobytes())
    file_map['fusion'] = ('fusion.bin', 0)

    # R22 value scalars, fixed order.
    layout = [
        ('hrq', 'top', M_TOP), ('hrq', 'fine', M_FINE),
        ('sparse', None, M_SPARSE), ('lowrank', None, M_LR),
        ('manifold', None, M_MAN),
        ('binding', 'key', M_BIND), ('binding', 'val', M_BIND),
        ('binding', 'bind', M_BIND),
        ('contrast', 'a', M_CONTRAST), ('contrast', 'b', M_CONTRAST),
    ]
    value_scalars = {'hrq': {}, 'binding': {}, 'contrast': {}}
    vs_parts = []
    for i, (name, family, m) in enumerate(layout):
        arr = np.full(m, i + 1.0, dtype=np.float32)
        vs_parts.append(arr)
        if family is None:
            value_scalars[name] = arr
        else:
            value_scalars[name][family] = arr
    exp['value_scalars'] = value_scalars
    tensors['value_scalars'] = np.concatenate(vs_parts)
    _write_header_bin(tmp_path / 'value_scalars.bin', len(layout), D, 1,
                      tensors['value_scalars'])
    file_map['value_scalars'] = ('value_scalars.bin', HEADER_SIZE_V2)

    # Self: C_self rows (core + modes) + R15 layout tag + bias.
    self_core = np.full(D, 0.375, dtype=np.float32)
    self_modes = np.zeros((N_SELF, D), dtype=np.float32)
    for j in range(N_SELF):
        self_modes[j, :] = 0.01 * (j + 1)
    exp['self_C'] = np.concatenate([self_core[None, :], self_modes])
    exp['self_tau'] = SELF_LAYOUT_TAG
    exp['self_bias'] = 0.125
    tensors['self_lattice'] = np.concatenate([
        self_core, self_modes.ravel(),
        np.asarray([exp['self_tau'], exp['self_bias']], dtype=np.float32),
    ])
    (tmp_path / 'self_lattice.bin').write_bytes(
        tensors['self_lattice'].tobytes())
    file_map['self_lattice'] = ('self_lattice.bin', 0)

    # Binding A_k / A_v.
    A_k = np.zeros((R_MAX, D), dtype=np.float32)
    A_v = np.zeros((R_MAX, D), dtype=np.float32)
    A_k[0, 0] = 1.0
    A_v[0, 1] = 1.0
    exp['bind_A_k'] = A_k
    exp['bind_A_v'] = A_v
    tensors['binding_proj'] = np.concatenate([A_k.ravel(), A_v.ravel()])
    (tmp_path / 'binding_proj.bin').write_bytes(
        tensors['binding_proj'].tobytes())
    file_map['binding_proj'] = ('binding_proj.bin', 0)

    save_manifest(str(tmp_path), tensors, file_map=file_map)
    return exp


def test_manifest_file_offsets(tmp_path):
    """Ruling 9: file + offset, canonical hash/offset remain name-sorted."""
    a = np.arange(6, dtype=np.float32)
    save_manifest(str(tmp_path), {'a': a},
                  file_map={'a': ('a.bin', HEADER_SIZE_V2)})
    manifest = load_manifest(str(tmp_path))
    entry = manifest['tensors']['a']
    assert entry['file'] == 'a.bin'
    assert entry['offset'] == HEADER_SIZE_V2
    assert entry['canonical_offset'] == 0
    assert entry['sha256'] == canonical_sha256({'a': a})


def test_manifest_pickle_entries_have_no_offset(tmp_path):
    """Pickle-stored params must not claim a meaningless byte offset."""
    a = np.arange(2, dtype=np.float32)
    save_manifest(
        str(tmp_path), {'a': a, 'params.x': a},
        file_map={'a': ('a.bin', 0),
                  'params.x': {'file': 'cog_params.pkl',
                               'storage': 'pickle'}})
    manifest = load_manifest(str(tmp_path))
    pickled = manifest['tensors']['params.x']
    assert pickled['file'] == 'cog_params.pkl'
    assert pickled['storage'] == 'pickle'
    assert pickled['offset'] is None


def test_loads_lowrank_all_layers(tmp_path):
    exp = _build_fixture(tmp_path)
    mem = load_lattice_memories(str(tmp_path), _cfg())

    assert mem['lowrank_U'].shape == (N_LR, M_LR, R_MAX)
    assert mem['lowrank_V'].shape == (D, R_MAX)
    assert list(mem['lowrank_ranks']) == list(RANKS)
    for k in range(N_LR):
        r_k = RANKS[k]
        np.testing.assert_array_equal(
            mem['lowrank_U'][k, :, :r_k], exp['lowrank_U'][k, :, :r_k])
        assert np.all(mem['lowrank_U'][k, :, r_k:] == 0), \
            f"layer {k} must be zero-padded past rank {r_k}"
    np.testing.assert_array_equal(mem['lowrank_V'], exp['lowrank_V'])


def test_lowrank_two_ranks_ignores_stale_n_lr_layers(tmp_path):
    """ranks is the file-split source of truth, not the n_lr_layers hint."""
    ranks = (1, 4)
    exp = _build_fixture(tmp_path, ranks=ranks)
    cfg = _cfg()
    cfg['ranks'] = list(ranks)
    cfg['n_lr_layers'] = 3          # deliberately stale (2 U layers on disk)
    mem = load_lattice_memories(str(tmp_path), cfg)

    assert mem['lowrank_U'].shape == (2, M_LR, R_MAX)
    assert list(mem['lowrank_ranks']) == [1, 4]
    np.testing.assert_array_equal(mem['lowrank_U'][0, :, :1],
                                  exp['lowrank_U'][0, :, :1])
    np.testing.assert_array_equal(mem['lowrank_U'][1, :, :4],
                                  exp['lowrank_U'][1, :, :4])


def test_lowrank_ranks_must_be_strictly_increasing(tmp_path):
    """b.md §4.3: the rank sequence increases to r_max."""
    ranks = (4, 1)
    _build_fixture(tmp_path, ranks=ranks)
    cfg = _cfg()
    cfg['ranks'] = list(ranks)
    try:
        load_lattice_memories(str(tmp_path), cfg)
    except ValueError as exc:
        assert 'strictly increasing' in str(exc)
    else:
        raise AssertionError("non-increasing ranks were accepted")


def test_loads_binding_and_contrast_all_layers(tmp_path):
    exp = _build_fixture(tmp_path)
    mem = load_lattice_memories(str(tmp_path), _cfg())

    for key, name in [('bind_key', 'bind_key'), ('bind_val', 'bind_val'),
                      ('bind_bind', 'bind_bind')]:
        assert mem[key].shape == (N_BIND, M_BIND, D)
        np.testing.assert_array_equal(mem[key], exp[name])
        # Family separation: the first element of layer 2 is family/layer
        # specific, so a swapped val/bind block is not silently accepted.
        assert mem[key][2, 0, 0] == exp[name][2, 0, 0]

    assert mem['contrast_a'].shape == (N_CONTRAST, M_CONTRAST, D)
    assert mem['contrast_b'].shape == (N_CONTRAST, M_CONTRAST, D)
    np.testing.assert_array_equal(mem['contrast_a'], exp['contrast_a'])
    np.testing.assert_array_equal(mem['contrast_b'], exp['contrast_b'])
    assert mem['contrast_a'][1, 0, 0] != mem['contrast_a'][0, 0, 0]
    assert mem['contrast_b'][1, 0, 0] != mem['contrast_a'][1, 0, 0]

    np.testing.assert_array_equal(mem['bind_A_k'], exp['bind_A_k'])
    np.testing.assert_array_equal(mem['bind_A_v'], exp['bind_A_v'])
    # Task 9: binding_mem_t.r_max = the shared low-rank V column count.
    assert mem['binding_r_max'] == R_MAX


def test_loads_hrq_sparse_manifold_route_fusion_self(tmp_path):
    exp = _build_fixture(tmp_path)
    mem = load_lattice_memories(str(tmp_path), _cfg())

    assert mem['hrq_top'].shape == (M_TOP, D)
    assert mem['hrq_fine'].shape == (N_FINE, M_FINE, D)
    np.testing.assert_array_equal(mem['hrq_top'], exp['hrq_top'])
    np.testing.assert_array_equal(mem['hrq_fine'], exp['hrq_fine'])

    assert mem['sparse_C'].shape == (M_SPARSE, D)
    assert mem['sparse_zero'].shape == (1, D)
    np.testing.assert_array_equal(mem['sparse_C'], exp['sparse_C'])
    np.testing.assert_array_equal(mem['sparse_zero'], exp['sparse_zero'])

    assert mem['manifold_C'].shape == (M_MAN, D)
    assert mem['manifold_T'].shape == (M_MAN, D, T_DIM)
    np.testing.assert_array_equal(mem['manifold_C'], exp['manifold_C'])
    np.testing.assert_array_equal(mem['manifold_T'], exp['manifold_T'])

    np.testing.assert_array_equal(mem['route_C'], exp['route_C'])
    np.testing.assert_array_equal(mem['route_W'], exp['route_W'])
    np.testing.assert_array_equal(mem['fusion_alpha'], exp['fusion_alpha'])
    np.testing.assert_array_equal(mem['fusion_ln_scale'],
                                  exp['fusion_ln_scale'])
    np.testing.assert_array_equal(mem['fusion_ln_bias'],
                                  exp['fusion_ln_bias'])
    assert set(mem['fusion']) == {'alpha', 'ln_scale', 'ln_bias'}
    np.testing.assert_array_equal(mem['fusion']['alpha'],
                                  exp['fusion_alpha'])

    assert mem['self_C'].shape == (1 + N_SELF, D)
    np.testing.assert_array_equal(mem['self_C'], exp['self_C'])
    # C fusion_mem_t.M_self counts mode rows, excluding the core row.
    assert mem['self_M'] == N_SELF
    assert mem['self_rows'] == N_SELF + 1
    assert mem['self_M'] + 1 == mem['self_C'].shape[0]
    assert mem['self_tau'] == exp['self_tau']
    assert mem['self_bias'] == exp['self_bias']

    for name, family, m in [
            ('hrq', 'top', M_TOP), ('hrq', 'fine', M_FINE),
            ('sparse', None, M_SPARSE), ('lowrank', None, M_LR),
            ('manifold', None, M_MAN), ('binding', 'key', M_BIND),
            ('binding', 'val', M_BIND), ('binding', 'bind', M_BIND),
            ('contrast', 'a', M_CONTRAST), ('contrast', 'b', M_CONTRAST)]:
        arr = (mem['value_scalars'][name] if family is None
               else mem['value_scalars'][name][family])
        expected = (exp['value_scalars'][name] if family is None
                    else exp['value_scalars'][name][family])
        assert arr.shape == (m,), f"value_scalars[{name}][{family}]"
        # Distinct per-leaf values: a wrong order or a zeroed loader fails.
        np.testing.assert_array_equal(
            arr, expected, err_msg=f"value_scalars[{name}][{family}]")


def test_truncated_lowrank_is_rejected(tmp_path):
    _build_fixture(tmp_path)
    path = tmp_path / 'lowrank_codebook.bin'
    path.write_bytes(path.read_bytes()[:-4])
    # The manifest length is longer than the file now: a short read, never a
    # silent zero tail.
    try:
        load_lattice_memories(str(tmp_path), _cfg())
    except ValueError as exc:
        assert 'short read' in str(exc) or 'truncated' in str(exc)
    else:
        raise AssertionError("truncated lowrank_codebook.bin was accepted")


def test_manifest_tensor_sha256_rejects_tampered_payload(tmp_path):
    """R10: a mutated .bin payload must fail the per-tensor sha256 check."""
    _build_fixture(tmp_path)
    path = tmp_path / 'hrq_codebook.bin'
    data = bytearray(path.read_bytes())
    # Flip one bit in the first payload float, after the 32-byte header.
    data[HEADER_SIZE_V2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256 mismatch|tampered|corrupt"):
        load_lattice_memories(str(tmp_path), _cfg())


def test_manifest_missing_sha256_is_rejected(tmp_path):
    """A v2 manifest without a per-tensor hash is not silently trusted."""
    _build_fixture(tmp_path)
    manifest = load_manifest(str(tmp_path))
    manifest['tensors']['hrq_codebook'].pop('sha256')
    (tmp_path / 'manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match="no sha256|re-export"):
        load_lattice_memories(str(tmp_path), _cfg())


def test_pre_r15_self_lattice_is_rejected(tmp_path):
    """Ruling 35: same-size old core+modes+tau+bias files must not load."""
    _build_fixture(tmp_path)
    path = tmp_path / 'self_lattice.bin'
    data = np.fromfile(path, dtype=np.float32)
    data[-2] = 1.0  # old tau_self slot, not SELF_LAYOUT_TAG
    data.tofile(path)
    # A real pre-R15 file would have a manifest hash matching the old bytes.
    # Refresh it so the loader reaches the R15 layout-tag rejection instead of
    # failing first on the (also correct) sha256 check.
    manifest = load_manifest(str(tmp_path))
    manifest['tensors']['self_lattice']['sha256'] = tensor_sha256(data)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="R15|layout tag|migrated"):
        load_lattice_memories(str(tmp_path), _cfg())


# ── Frozen safety codebooks: file SHA, manifest SHA, shape guards ───────────

def _write_safety_bin(path, M, d, pos, neg, sha=None):
    """Write a v2 safety file: header + two float32 halves + 32-byte SHA."""
    from train.bin_format import _compute_checksum

    data = np.ascontiguousarray(pos, dtype=np.float32).tobytes()
    data += np.ascontiguousarray(neg, dtype=np.float32).tobytes()
    crc = _compute_checksum(data)
    if sha is None:
        import hashlib
        sha = hashlib.sha256(data).digest()
    path.write_bytes(
        struct.pack(HEADER_FMT_V2, int(M), int(d), 1, 2, 1.0,
                    FORMAT_VERSION, 0, crc) + data + sha)
    return sha.hex()


def _safety_halves(M, d, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(M, d)).astype(np.float32)
    neg = rng.normal(size=(M, d)).astype(np.float32)
    assert not np.array_equal(pos, neg)
    return pos, neg


def test_safety_codebook_valid_load_and_digest(tmp_path):
    import lcm

    pos, neg = _safety_halves(4, 8)
    digest = _write_safety_bin(tmp_path / "gvalue_codebook.bin",
                               4, 8, pos, neg)
    data, M, d, header = lcm.load_codebook_for_c(
        str(tmp_path), "gvalue_codebook.bin", return_header=True)
    assert (M, d) == (4, 8)
    assert data.size == 2 * M * d
    assert header["sha256"] == digest
    lcm.validate_safety_codebooks(
        data[:M * d], data[M * d:], M,
        np.zeros((1, 8)), np.ones((1, 8)), 1)


def test_safety_codebook_rejects_tampered_trailing_sha(tmp_path):
    import lcm

    pos, neg = _safety_halves(4, 8)
    bad = bytes(32)  # not the payload digest
    _write_safety_bin(tmp_path / "gvalue_codebook.bin", 4, 8, pos, neg, sha=bad)
    with pytest.raises(ValueError, match="SHA-256 suffix mismatch"):
        lcm.load_codebook_for_c(str(tmp_path), "gvalue_codebook.bin")


def test_safety_codebook_rejects_truncated_payload(tmp_path):
    import lcm

    pos, neg = _safety_halves(4, 8)
    # Header claims 4x8 per half, file carries one byte fewer (unaligned).
    full = np.concatenate([pos.ravel(), neg.ravel()]).astype(np.float32)
    data = full.tobytes()[:-1]
    import hashlib
    from train.bin_format import _compute_checksum
    path = tmp_path / "gvalue_codebook.bin"
    path.write_bytes(
        struct.pack(HEADER_FMT_V2, 4, 8, 1, 2, 1.0, FORMAT_VERSION, 0,
                    _compute_checksum(data)) + data
        + hashlib.sha256(data).digest())
    with pytest.raises(ValueError, match="not float32-aligned"):
        lcm.load_codebook_for_c(str(tmp_path), "gvalue_codebook.bin")

    # Also a float-aligned short file: M*d header does not describe the data.
    trunc = full[:-2]
    data = trunc.tobytes()
    path.write_bytes(
        struct.pack(HEADER_FMT_V2, 4, 8, 1, 2, 1.0, FORMAT_VERSION, 0,
                    _compute_checksum(data)) + data
        + hashlib.sha256(data).digest())
    with pytest.raises(ValueError, match="truncated or the header"):
        lcm.load_codebook_for_c(str(tmp_path), "gvalue_codebook.bin")


def test_safety_codebook_rejects_wrong_d_and_zero_M(tmp_path):
    import lcm

    pos, neg = _safety_halves(4, 8)
    path = tmp_path / "gvalue_codebook.bin"
    _write_safety_bin(path, 4, 4, pos, neg)  # header d=4, payload 4x8 halves
    with pytest.raises(ValueError, match="truncated or the header"):
        lcm.load_codebook_for_c(str(tmp_path), "gvalue_codebook.bin")

    _write_safety_bin(path, 0, 8,
                      np.zeros((0, 8)), np.zeros((0, 8)))
    with pytest.raises(ValueError, match="invalid safety codebook header"):
        lcm.load_codebook_for_c(str(tmp_path), "gvalue_codebook.bin")

    with pytest.raises(SystemExit, match="requires exactly 4"):
        lcm.validate_safety_codebooks(
            np.zeros(8), np.ones(8), 0,
            np.zeros((1, 8)), np.ones((1, 8)), 1)
    with pytest.raises(SystemExit, match="danger header has M=0"):
        lcm.validate_safety_codebooks(
            np.zeros(32), np.ones(32), 4,
            np.zeros((0, 8)), np.zeros((0, 8)), 0)


def test_safety_codebook_verifies_manifest_sha(tmp_path):
    import lcm

    pos, neg = _safety_halves(4, 8)
    _write_safety_bin(tmp_path / "gvalue_codebook.bin", 4, 8, pos, neg)
    tensors = {"gvalue_codebook": np.concatenate(
        [pos.ravel(), neg.ravel()]).astype(np.float32)}
    save_manifest(str(tmp_path), tensors,
                  file_map={"gvalue_codebook": ("gvalue_codebook.bin",
                                                HEADER_SIZE_V2)})
    lcm.load_codebook_for_c(str(tmp_path), "gvalue_codebook.bin")

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["tensors"]["gvalue_codebook"]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sha256 mismatch against manifest"):
        lcm.load_codebook_for_c(str(tmp_path), "gvalue_codebook.bin")


def test_stage1_save_gvalue_roundtrips_through_inference_loader(tmp_path):
    """The legacy Stage-1 saver must write distinct anchors, not C_pos.copy()."""
    import lcm
    from train.checkpoint import _save_gvalue
    from train.gvalue import GValueCodebook, make_global_value_vectors

    d = 8
    C_pos, C_neg = make_global_value_vectors(d)
    gv = GValueCodebook(C_pos, C_neg)
    path = tmp_path / "gvalue_codebook.bin"
    digest = _save_gvalue(gv, str(path), d)

    data, M, loaded_d, header = lcm.load_codebook_for_c(
        str(tmp_path), "gvalue_codebook.bin", return_header=True)
    assert (M, loaded_d) == (4, d)
    assert data.size == 2 * M * d
    assert header["sha256"] == digest.hex()
    pos = data[:M * d].reshape(M, d)
    neg = data[M * d:].reshape(M, d)
    assert not np.array_equal(pos, neg), "legacy saver still writes C_neg=C_pos"
    lcm.validate_safety_codebooks(pos, neg, M, pos, neg + 1.0, M)
