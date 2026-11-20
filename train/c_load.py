"""Full lattice-parameter loader for the C inference engine (schema v2, R10).

C never parses JSON (Ruling 3). This module reads ``manifest.json``, uses each
logical tensor's ``file``/``offset``/``length`` fields to locate the payload,
and returns numpy arrays in the exact layouts ``infer/lcm.h``'s v3 structs
expect. Every returned array is caller-owned, C-contiguous float32; C only
binds pointers to them.

Deliberately JAX-free: ``lcm.py`` can run the C inference path without JAX.

Layer layouts (matching the training/export side):
    HRQ        top (M_top, d) then fine layers (n_fine, M_fine, d)
    Sparse     C (M_sparse, d) then zero_vec (1, d) appended in the file
    LowRank    U_k (M_lr, r_k) per layer, then shared V (d, r_max)
    Manifold   C (M_man, d) then T (M_man, d, t_dim)
    Binding    per layer: key (M_bind, d), val, bind
    Contrast   all C_a layers then all C_b layers
"""
import os

import numpy as np

from train.bin_format import load_manifest


def _require(cfg, key):
    if key not in cfg:
        raise KeyError(f"config.json is missing {key!r}; re-export required")
    return cfg[key]


def _read_tensor(ckpt_dir, manifest, name):
    """Read one logical tensor via its manifest file mapping.

    ``offset`` is the byte offset inside ``file`` (Ruling 9); ``length`` is
    the float32 payload size. The manifest's shape must agree with length.
    """
    entry = manifest['tensors'].get(name)
    if entry is None:
        raise ValueError(
            f"manifest.json has no tensor {name!r}; re-export required")
    file_name = entry.get('file')
    if not file_name:
        raise ValueError(
            f"manifest tensor {name!r} has no 'file' mapping; re-export "
            f"required (schema v2 needs file/offset)")
    if file_name.endswith('.pkl'):
        raise ValueError(
            f"manifest tensor {name!r} lives in {file_name}, not a .bin "
            f"file; the C loader only reads flat float32 files")

    length = int(entry['length'])
    if length % 4:
        raise ValueError(f"{name!r}: length {length} is not float32-aligned")
    path = os.path.join(ckpt_dir, file_name)
    with open(path, 'rb') as f:
        f.seek(int(entry['offset']))
        raw = f.read(length)
    if len(raw) != length:
        raise ValueError(
            f"{path}: short read for {name!r} "
            f"({len(raw)} of {length} bytes); checkpoint is truncated")
    arr = np.frombuffer(raw, dtype='<f4')
    expected = int(np.prod(entry['shape'], dtype=np.int64)) if entry['shape'] else 1
    if arr.size != expected:
        raise ValueError(
            f"{name!r}: {arr.size} floats in {file_name} but shape "
            f"{entry['shape']} needs {expected}")
    return np.ascontiguousarray(arr.reshape(entry['shape']), dtype=np.float32)


def _value_scalar_layout(cfg):
    """R22 order. Must match train.checkpoint._value_scalar_specs."""
    return [
        ('hrq', 'top', int(_require(cfg, 'M_top'))),
        ('hrq', 'fine', int(_require(cfg, 'M_fine'))),
        ('sparse', None, int(_require(cfg, 'M_sparse'))),
        ('lowrank', None, int(_require(cfg, 'M_lr'))),
        ('manifold', None, int(_require(cfg, 'M_man'))),
        ('binding', 'key', int(_require(cfg, 'M_bind'))),
        ('binding', 'val', int(_require(cfg, 'M_bind'))),
        ('binding', 'bind', int(_require(cfg, 'M_bind'))),
        ('contrast', 'a', int(_require(cfg, 'M_contrast'))),
        ('contrast', 'b', int(_require(cfg, 'M_contrast'))),
    ]


def load_lattice_memories(ckpt_dir, cfg):
    """Load every lattice/tensor the C engine needs from an exported ckpt.

    Args:
        ckpt_dir: directory containing ``manifest.json`` and the ``.bin``
            files it points at.
        cfg: the exported ``config.json`` as a dict (shape facts only; no
            JAX import).

    Returns a dict with:
        hrq_top, hrq_fine, sparse_C, sparse_zero,
        lowrank_U, lowrank_V, lowrank_ranks,
        manifold_C, manifold_T,
        bind_key, bind_val, bind_bind, bind_A_k, bind_A_v,
        contrast_a, contrast_b,
        route_C, route_W,
        fusion_alpha, fusion_ln_scale, fusion_ln_bias,
        value_scalars, self_C, self_core, self_modes, self_tau, self_bias.

    Raises ValueError on a missing/mismatched tensor instead of silently
    falling back to first-layer or zero data.
    """
    ckpt_dir = str(ckpt_dir)
    manifest = load_manifest(ckpt_dir)
    d = int(_require(cfg, 'd_model'))
    if d <= 0:
        raise ValueError(f"config.json d_model={d} is not positive")

    # ── HRQ: top + n_fine fine layers ──────────────────────────────────────
    M_top = int(_require(cfg, 'M_top'))
    M_fine = int(_require(cfg, 'M_fine'))
    n_fine = int(cfg.get('n_hrq_layers', 0))
    hrq = _read_tensor(ckpt_dir, manifest, 'hrq_codebook').ravel()
    expected = M_top * d + n_fine * M_fine * d
    if hrq.size != expected:
        raise ValueError(
            f"hrq_codebook has {hrq.size} floats; expected {expected} "
            f"(M_top={M_top}, n_fine={n_fine}, M_fine={M_fine}, d={d})")
    hrq_top = np.ascontiguousarray(hrq[:M_top * d].reshape(M_top, d))
    hrq_fine = np.ascontiguousarray(
        hrq[M_top * d:].reshape(n_fine, M_fine, d))

    # ── Sparse: C then zero_vec ────────────────────────────────────────────
    M_sparse = int(_require(cfg, 'M_sparse'))
    sparse = _read_tensor(ckpt_dir, manifest, 'sparse_codebook').ravel()
    if sparse.size == (M_sparse + 1) * d:
        sparse_C = np.ascontiguousarray(
            sparse[:M_sparse * d].reshape(M_sparse, d))
        sparse_zero = np.ascontiguousarray(
            sparse[M_sparse * d:].reshape(1, d))
    elif sparse.size == M_sparse * d:
        # Older v2 exports omitted the zero row. C must not invent one: the
        # LFQ origin is exactly zero, but flag it explicitly rather than
        # silently mixing a padded row into C.
        raise ValueError(
            "sparse_codebook is missing the appended zero_vec row; "
            "re-export required (R2 needs zero_vec)")
    else:
        raise ValueError(
            f"sparse_codebook has {sparse.size} floats; expected "
            f"{(M_sparse + 1) * d} (M_sparse={M_sparse} + zero_vec, d={d})")

    # ── LowRank: ragged U_k blocks then shared V ──────────────────────────
    M_lr = int(_require(cfg, 'M_lr'))
    r_max = int(_require(cfg, 'r_max'))
    n_lr = int(cfg.get('n_lr_layers', 0))
    ranks = cfg.get('ranks')
    if ranks is None:
        if n_lr <= 0:
            raise ValueError("config.json has neither ranks nor n_lr_layers")
        ranks = [r_max] * n_lr
    ranks = [int(r) for r in ranks]
    if n_lr and len(ranks) != n_lr:
        raise ValueError(
            f"config.json ranks has {len(ranks)} entries but "
            f"n_lr_layers={n_lr}")
    n_lr = len(ranks)
    lowrank = _read_tensor(ckpt_dir, manifest, 'lowrank_codebook').ravel()
    lowrank_U = np.zeros((n_lr, M_lr, r_max), dtype=np.float32)
    pos = 0
    for k, r_k in enumerate(ranks):
        if r_k < 1 or r_k > r_max:
            raise ValueError(f"ranks[{k}]={r_k} outside [1, r_max={r_max}]")
        n = M_lr * r_k
        if pos + n > lowrank.size:
            raise ValueError(
                f"lowrank_codebook is truncated at layer {k} "
                f"(needs {pos + n} floats, has {lowrank.size})")
        lowrank_U[k, :, :r_k] = lowrank[pos:pos + n].reshape(M_lr, r_k)
        pos += n
    v_flat = lowrank[pos:]
    if v_flat.size == d * r_max:
        lowrank_V = np.ascontiguousarray(v_flat.reshape(d, r_max))
    elif v_flat.size == d * d:
        # Tolerate a full (d, d) shared base: only the leading r_max columns
        # are used by the rank-k reconstruction.
        lowrank_V = np.ascontiguousarray(v_flat.reshape(d, d)[:, :r_max])
    else:
        raise ValueError(
            f"lowrank_codebook tail has {v_flat.size} floats; expected "
            f"d*r_max={d * r_max} (or d*d)")

    # ── Manifold: C then T ─────────────────────────────────────────────────
    M_man = int(_require(cfg, 'M_man'))
    t_dim = int(_require(cfg, 't_dim'))
    manifold = _read_tensor(ckpt_dir, manifest, 'manifold_codebook').ravel()
    expected = M_man * d + M_man * d * t_dim
    if manifold.size != expected:
        raise ValueError(
            f"manifold_codebook has {manifold.size} floats; expected "
            f"{expected} (M_man={M_man}, d={d}, t_dim={t_dim})")
    manifold_C = np.ascontiguousarray(manifold[:M_man * d].reshape(M_man, d))
    manifold_T = np.ascontiguousarray(
        manifold[M_man * d:].reshape(M_man, d, t_dim))

    # ── Binding: per layer key/val/bind ────────────────────────────────────
    M_bind = int(_require(cfg, 'M_bind'))
    n_bind = int(_require(cfg, 'n_bind_layers'))
    bind = _read_tensor(ckpt_dir, manifest, 'bind_codebook').ravel()
    expected = n_bind * 3 * M_bind * d
    if bind.size != expected:
        raise ValueError(
            f"bind_codebook has {bind.size} floats; expected {expected} "
            f"(n_bind_layers={n_bind}, M_bind={M_bind}, d={d})")
    bind_blocks = bind.reshape(n_bind, 3, M_bind, d)
    bind_key = np.ascontiguousarray(bind_blocks[:, 0])
    bind_val = np.ascontiguousarray(bind_blocks[:, 1])
    bind_bind = np.ascontiguousarray(bind_blocks[:, 2])

    # ── Contrast: all C_a layers, then all C_b layers ─────────────────────
    M_contrast = int(_require(cfg, 'M_contrast'))
    n_contrast = int(_require(cfg, 'n_contrast_layers'))
    contrast = _read_tensor(ckpt_dir, manifest, 'contrast_codebook').ravel()
    expected = 2 * n_contrast * M_contrast * d
    if contrast.size != expected:
        raise ValueError(
            f"contrast_codebook has {contrast.size} floats; expected "
            f"{expected} (n_contrast_layers={n_contrast}, "
            f"M_contrast={M_contrast}, d={d})")
    half = n_contrast * M_contrast * d
    contrast_a = np.ascontiguousarray(
        contrast[:half].reshape(n_contrast, M_contrast, d))
    contrast_b = np.ascontiguousarray(
        contrast[half:].reshape(n_contrast, M_contrast, d))

    # ── Binding projections ───────────────────────────────────────────────
    proj = _read_tensor(ckpt_dir, manifest, 'binding_proj').ravel()
    expected = 2 * r_max * d
    if proj.size != expected:
        raise ValueError(
            f"binding_proj has {proj.size} floats; expected {expected} "
            f"(r_max={r_max}, d={d})")
    bind_A_k = np.ascontiguousarray(proj[:r_max * d].reshape(r_max, d))
    bind_A_v = np.ascontiguousarray(proj[r_max * d:].reshape(r_max, d))

    # ── Routing + fusion ──────────────────────────────────────────────────
    n_lattices = int(_require(cfg, 'n_lattices'))
    route = _read_tensor(ckpt_dir, manifest, 'route_codebook').ravel()
    expected = n_lattices * d + d * n_lattices
    if route.size != expected:
        raise ValueError(
            f"route_codebook has {route.size} floats; expected {expected} "
            f"(n_lattices={n_lattices}, d={d})")
    route_C = np.ascontiguousarray(
        route[:n_lattices * d].reshape(n_lattices, d))
    route_W = np.ascontiguousarray(
        route[n_lattices * d:].reshape(d, n_lattices))

    fusion = _read_tensor(ckpt_dir, manifest, 'fusion').ravel()
    expected = n_lattices + 2 * d
    if fusion.size != expected:
        raise ValueError(
            f"fusion has {fusion.size} floats; expected {expected} "
            f"(n_lattices={n_lattices}, 2*d={2 * d})")
    fusion_alpha = np.ascontiguousarray(fusion[:n_lattices])
    fusion_ln_scale = np.ascontiguousarray(
        fusion[n_lattices:n_lattices + d])
    fusion_ln_bias = np.ascontiguousarray(
        fusion[n_lattices + d:n_lattices + 2 * d])

    # ── R22 value scalars (nested, fixed order) ───────────────────────────
    vs_layout = _value_scalar_layout(cfg)
    vs_total = sum(m for _, _, m in vs_layout)
    vs_flat = _read_tensor(ckpt_dir, manifest, 'value_scalars').ravel()
    if vs_flat.size != vs_total:
        raise ValueError(
            f"value_scalars has {vs_flat.size} floats; expected {vs_total} "
            f"(R22 nested layout); old flat checkpoints are not migrated")
    value_scalars = {'hrq': {}, 'binding': {}, 'contrast': {}}
    pos = 0
    for name, family, m in vs_layout:
        arr = np.ascontiguousarray(vs_flat[pos:pos + m])
        if family is None:
            value_scalars[name] = arr
        else:
            value_scalars[name][family] = arr
        pos += m

    # ── Self lattice: core + modes + 2 scalars (R15) ──────────────────────
    n_self = int(_require(cfg, 'n_self_codes'))
    self_flat = _read_tensor(ckpt_dir, manifest, 'self_lattice').ravel()
    expected = d + n_self * d + 2
    if self_flat.size != expected:
        raise ValueError(
            f"self_lattice has {self_flat.size} floats; expected {expected} "
            f"(d={d}, n_self_codes={n_self})")
    pos = 0
    self_core = np.ascontiguousarray(self_flat[pos:pos + d])
    pos += d
    self_modes = np.ascontiguousarray(
        self_flat[pos:pos + n_self * d].reshape(n_self, d))
    pos += n_self * d
    self_tau = float(self_flat[pos])
    self_bias = float(self_flat[pos + 1])
    # C_self[row 0] is the frozen identity anchor; modes follow (R15).
    self_C = np.ascontiguousarray(
        np.concatenate([self_core[None, :], self_modes], axis=0))

    return {
        'd': d,
        'n_lattices': n_lattices,
        'hrq_top': hrq_top,
        'hrq_fine': hrq_fine,
        'sparse_C': sparse_C,
        'sparse_zero': sparse_zero,
        'lowrank_U': lowrank_U,
        'lowrank_V': lowrank_V,
        'lowrank_ranks': np.asarray(ranks, dtype=np.int32),
        'manifold_C': manifold_C,
        'manifold_T': manifold_T,
        'bind_key': bind_key,
        'bind_val': bind_val,
        'bind_bind': bind_bind,
        'bind_A_k': bind_A_k,
        'bind_A_v': bind_A_v,
        'contrast_a': contrast_a,
        'contrast_b': contrast_b,
        'route_C': route_C,
        'route_W': route_W,
        'fusion_alpha': fusion_alpha,
        'fusion_ln_scale': fusion_ln_scale,
        'fusion_ln_bias': fusion_ln_bias,
        'value_scalars': value_scalars,
        'self_C': self_C,
        'self_core': self_core,
        'self_modes': self_modes,
        'self_tau': self_tau,
        'self_bias': self_bias,
    }
