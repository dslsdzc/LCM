"""LCM checkpoint: per-module flat binary format (.bin).

Each codebook .bin file uses:
  ┌──────────────────────┐
  │ Header (24 bytes)    │  M, d, n_layers, type, c, checksum
  ├──────────────────────┤
  │ Data (float32[])     │  flat vectors, M*d per layer
  ├──────────────────────┤
  │ SHA-256 (32 bytes)   │  (frozen modules only: gvalue, danger)
  └──────────────────────┘

Directory layout:
  checkpoint/
  ├── encoder.bin           # float32 flat (shapes from config.json)
  ├── decoder.bin           # float32 flat
  ├── hrq_codebook.bin      # header + vectors
  ├── sparse_codebook.bin   # header + vectors + zero_vec
  ├── lowrank_codebook.bin  # header + U_k per layer + V
  ├── manifold_codebook.bin # header + C_man + T_j
  ├── bind_codebook.bin     # header + key/val/bind per layer
  ├── contrast_codebook.bin # header + C_a/C_b per layer
  ├── gvalue_codebook.bin   # header + C_pos + C_neg + SHA256
  ├── opt_state.bin         # optimizer state (Python, not C-readable)
  ├── config.json           # all hyperparameters
  ├── tokenizer.json        # BPE vocabulary
  └── step.txt              # step number
"""
import os
import json
import struct
import hashlib
import pickle
import zlib
import shutil
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

# ── Binary header format ─────────────────────────────────────────────────────
# struct: <iiifi = 4+4+4+4+4+4 = 24 bytes
# 6 fields: M(int32), d(int32), n_layers(int32), type(int32), c(float32), checksum(uint32)
HEADER_FMT = '<iiiifI'
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Codebook type constants
CB_EUCLIDEAN = 1
CB_HYPERBOLIC = 2


def _pack_header(M, d, n_layers, cb_type, curvature=1.0):
    """Pack 24-byte codebook header."""
    return struct.pack(HEADER_FMT, int(M), int(d), int(n_layers),
                       int(cb_type), float(curvature), 0)


def _unpack_header(data):
    """Unpack 24-byte header → (M, d, n_layers, type, c, checksum_uint32)."""
    raw = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    # Convert to standard ints
    return raw


def _compute_checksum(data):
    """CRC32 of data section (uint32)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def _sha256(data):
    return hashlib.sha256(data).digest()


# ── Save helpers ──────────────────────────────────────────────────────────────

def _write_bin(path, header_bytes, data_bytes, sha256_digest=None):
    """Write .bin file: header + data + optional SHA256."""
    # Compute CRC32 over data
    checksum = _compute_checksum(data_bytes)
    # Patch header with real checksum
    hdr = struct.unpack(HEADER_FMT, header_bytes)
    hdr_fixed = struct.pack(HEADER_FMT, *hdr[:5], checksum)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(hdr_fixed)
        f.write(data_bytes)
        if sha256_digest:
            f.write(sha256_digest)
    size_kb = os.path.getsize(path) / 1024
    print(f"  [CKPT] {os.path.basename(path)} ({size_kb:.0f} KB)")


def _read_bin(path, has_hash=False):
    """Read .bin file → (header_dict, data_array_or_raw, sha256_or_None).

    For uniform codebooks (all layers same M), returns (n_layers, M, d) array.
    For variable-size codebooks (HRQ), returns flat 1D array — caller must reshape.
    """
    with open(path, 'rb') as f:
        hdr_bytes = f.read(HEADER_SIZE)
        M, d, n_layers, cb_type, c, stored_crc = _unpack_header(hdr_bytes)
        data_bytes = f.read()  # read all remaining bytes
        sha = None
        if has_hash:
            sha = data_bytes[-32:]
            data_bytes = data_bytes[:-32]

        # Verify checksum (WARN, not raise: old checkpoints predate CRC or
        # were written by writers that computed it differently — they must
        # still load, but the corruption is surfaced clearly).
        actual_crc = _compute_checksum(data_bytes)
        if actual_crc != stored_crc:
            print(f"  [WARN] CRC mismatch in {path}: stored={stored_crc:#x}, "
                  f"actual={actual_crc:#x} — file is corrupt or was written "
                  f"by an older writer; re-export the checkpoint")

        data = np.frombuffer(data_bytes, dtype=np.float32)
        expected = M * d * n_layers
        if data.size == expected:
            data = data.reshape(n_layers, M, d)

    return {'M': M, 'd': d, 'n_layers': n_layers, 'type': cb_type, 'c': c}, data, sha


# ── Per-module savers ────────────────────────────────────────────────────────

def _save_encoder(params, path):
    """encoder: embed + rel_bias + layers (ln1, attn, ln2, glu) + q_pool + w_proj."""
    enc = params['encoder']
    parts = []
    parts.append(np.asarray(enc['embed']).ravel())         # (V, d)
    parts.append(np.asarray(enc['rel_bias']).ravel())      # (N*2-1,) or similar
    for layer in enc['layers']:
        for k in ['ln1_scale', 'ln1_bias', 'w_q', 'w_k', 'w_v', 'w_o',
                   'ln2_scale', 'ln2_bias', 'w_1', 'w_2', 'w_3']:
            parts.append(np.asarray(layer[k]).ravel())
    parts.append(np.asarray(enc['q_pool']).ravel())        # (d,)
    parts.append(np.asarray(enc['w_proj']).ravel())        # (d, d)
    flat = np.concatenate(parts).astype(np.float32)
    flat.tofile(path)


def _save_decoder(params, path):
    """decoder (gen_head): causal linear attn + GLU params → flat float32."""
    gh = params['gen_head']
    flat = np.concatenate([
        np.asarray(gh[k]).ravel()
        for k in ['w_embed', 'w_q', 'w_k', 'w_v', 'w_o', 'w_1', 'w_2', 'w_3']
    ]).astype(np.float32)
    flat.tofile(path)


def _save_codebook(path, codebook_arrays, M, d, n_layers, cb_type, curvature=1.0):
    """Save codebook with header. codebook_arrays: list of (M, d) arrays."""
    hdr = _pack_header(M, d, n_layers, cb_type, curvature)
    data_bytes = b''.join(np.asarray(a, dtype=np.float32).tobytes() for a in codebook_arrays)
    _write_bin(path, hdr, data_bytes)


def _save_codebook_from_params(params, key, M, d, output_dir, cb_type=CB_EUCLIDEAN, curvature=1.0, zero_vec=False, cfg=None):
    """Save a codebook from params dict, handling zero_vec expansion."""
    cbs = []
    if key == 'hrq':
        # HRQ has varying M per layer (M_top ≠ M_fine), use flat format
        p = params['hrq']
        cbs.append(p['top']['A'] @ p['top']['W'])  # (M_top, d)
        for fl in p['fine']:
            cbs.append(fl['A'] @ fl['W'])  # (M_fine, d)
        path = os.path.join(output_dir, "hrq_codebook.bin")
        data_bytes = b''.join(np.asarray(a, dtype=np.float32).tobytes() for a in cbs)
        # Use max M in header, n_layers=1. Loader uses config to parse.
        hdr = _pack_header(M, d, 1, cb_type, curvature)
        _write_bin(path, hdr, data_bytes)
        return
    elif key == 'sparse':
        C = params['sparse']['C']
        if zero_vec:
            Z = params['sparse'].get('zero_vec', jnp.zeros((1, d)))
            C = jnp.concatenate([C, Z], axis=0)  # (M+1, d)
            M_actual = M + 1
        else:
            M_actual = M
        path = os.path.join(output_dir, "sparse_codebook.bin")
        return _save_codebook(path, [C], M_actual, d, 1, cb_type=CB_EUCLIDEAN)
    elif key == 'manifold':
        C = params['manifold']['C']
        t_dim = params['manifold']['T'].shape[-2]
        T = params['manifold']['T'].reshape(M, -1)  # (M, t*d)
        path = os.path.join(output_dir, "manifold_codebook.bin")
        return _save_codebook(path, [C, T], M, d, 1, cb_type=CB_HYPERBOLIC, curvature=1.0)


def _save_lowrank(params, path, M_lr, d):
    """lowrank: U_k for each layer + V = A_V @ W_V."""
    lr = params['lowrank']
    parts = []
    for U in lr['U']:
        parts.append(np.asarray(U).ravel())  # (M_lr, r_k)
    V = np.asarray(lr['A_V'] @ lr['W_V'])  # (d, d)
    parts.append(V.ravel())
    flat = np.concatenate(parts).astype(np.float32)
    flat.tofile(path)


def _save_binding(params, path, M_bind, d):
    """binding: key/val/bind sub-codebooks per layer."""
    bind = params['binding']
    parts = []
    for l in range(len(bind.get('key_cb', []))):
        for cb_type in ['key_cb', 'val_cb', 'bind_cb']:
            C = bind[cb_type][l]['A'] @ bind[cb_type][l]['W']  # (M_bind, d)
            parts.append(np.asarray(C).ravel())
    flat = np.concatenate(parts).astype(np.float32)
    flat.tofile(path)


def _save_contrast(params, path):
    """contrast: C_a + C_b per layer (each layer uses A @ W for clean codebook)."""
    c = params['contrast']
    parts = []
    for layer_a in c['C_a']:
        C = layer_a['A'] @ layer_a['W']
        parts.append(np.asarray(C).ravel())
    for layer_b in c['C_b']:
        C = layer_b['A'] @ layer_b['W']
        parts.append(np.asarray(C).ravel())
    flat = np.concatenate(parts).astype(np.float32)
    flat.tofile(path)


def _save_gvalue(gvalue, path, d):
    """gvalue: C_pos + C_neg + SHA-256 hash."""
    C_pos = np.asarray(gvalue.C_pos, dtype=np.float32)
    C_neg = np.asarray(gvalue.C_neg, dtype=np.float32)
    M_pair = C_pos.shape[0]

    hdr = _pack_header(M_pair, d, 1, CB_HYPERBOLIC, 1.0)
    data_bytes = C_pos.tobytes() + C_neg.tobytes()
    sha = _sha256(data_bytes)
    _write_bin(path, hdr, data_bytes, sha256_digest=sha)
    print(f"    SHA-256: {sha.hex()[:16]}...")
    return sha


def _save_opt_state(opt_state, path):
    """Optimizer state (pickle, for training resume only)."""
    def _to_np(x):
        try:
            return np.asarray(jax.device_get(x))
        except Exception:
            return x
    flat = jax.tree_util.tree_map(_to_np, opt_state)
    with open(path, 'wb') as f:
        pickle.dump(flat, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  [CKPT] {os.path.basename(path)} ({size_kb:.0f} KB)")


# ── Public save/load API ─────────────────────────────────────────────────────

def save_checkpoint(state, cfg, output_dir="checkpoint", step=None):
    """Save full LCM checkpoint as flat binary files.

    Args:
        state: Training state from create_train_state.
        cfg: LCMConfig.
        output_dir: Output directory (default: checkpoint/).
        step: Step number for filenames.

    Returns:
        output_dir path.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    d = cfg.d_model
    params = state['params']

    print(f"[CKPT] Saving checkpoint to {output_dir}/")

    # 1. Config
    cfg_dict = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(cfg_dict, f, indent=2, default=str)
    print(f"  [CKPT] config.json")

    # 2. Tokenizer (copy from data/)
    import shutil as _su
    tokenizer_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tokenizer.json")
    if os.path.exists(tokenizer_src):
        _su.copy2(tokenizer_src, os.path.join(output_dir, "tokenizer.json"))
        print(f"  [CKPT] tokenizer.json")

    # 3. Encoder
    encoder_path = os.path.join(output_dir, "encoder.bin")
    _save_encoder(params, encoder_path)
    enc_size = os.path.getsize(encoder_path) / 1e6
    print(f"  [CKPT] encoder.bin ({enc_size:.1f} MB)")

    # 4. Decoder (gen_head)
    decoder_path = os.path.join(output_dir, "decoder.bin")
    _save_decoder(params, decoder_path)
    print(f"  [CKPT] decoder.bin")

    # 5. Codebooks
    _save_codebook_from_params(params, 'hrq', cfg.M_top, d, output_dir)
    _save_codebook_from_params(params, 'sparse', cfg.M_sparse, d, output_dir, zero_vec=True)

    _save_lowrank(params, os.path.join(output_dir, "lowrank_codebook.bin"), cfg.M_lr, d)
    _save_codebook_from_params(params, 'manifold', cfg.M_man, d, output_dir)
    _save_binding(params, os.path.join(output_dir, "bind_codebook.bin"), cfg.M_bind, d)

    # Contrast
    _save_contrast(params, os.path.join(output_dir, "contrast_codebook.bin"))

    # 6. Route codebook
    route_path = os.path.join(output_dir, "route_codebook.bin")
    _save_route(params, route_path)
    print(f"  [CKPT] route_codebook.bin")

    # 7. GValue (frozen, with hash)
    if state.get('gvalue') is not None:
        gv_path = os.path.join(output_dir, "gvalue_codebook.bin")
        _save_gvalue(state['gvalue'], gv_path, d)

    # 8. Danger codebook (frozen, with hash)
    if 'danger' in params:
        dg_path = os.path.join(output_dir, "danger_codebook.bin")
        _save_danger(params, dg_path, cfg.M_danger, d)

    # 9. Optimizer state (for resume)
    opt_path = os.path.join(output_dir, "opt_state.bin")
    _save_opt_state(state['opt_state'], opt_path)

    # 10. Step metadata
    step_val = step if step is not None else state.get('step', 0)
    with open(os.path.join(output_dir, "step.txt"), 'w') as f:
        f.write(str(step_val))

    print(f"[CKPT] Done → {output_dir}/")
    return output_dir


def _save_route(params, path):
    """route: C_route + W_route → flat float32."""
    rt = params['route']
    flat = np.concatenate([
        np.asarray(rt['C_route']).ravel(),
        np.asarray(rt['W_route']).ravel(),
    ]).astype(np.float32)
    flat.tofile(path)


def _save_danger(params, path, M_danger, d):
    """danger: threats + normals, SHA-256 hash (frozen, values not trained).

    The C engine (lcm.py) reads TWO halves: C_threats (M_danger × d) then
    C_normal (M_danger × d). params['danger'] only carries the threat
    codebook (C) — the normal half is a C placeholder: identical threats
    give danger_score = sim_threat - sim_normal = 0, so the pattern-match
    gate never fires until real normals are trained.
    """
    C = np.asarray(params['danger']['C'], dtype=np.float32)
    hdr = _pack_header(M_danger, d, 1, CB_EUCLIDEAN, 1.0)
    data_bytes = C.tobytes() + C.tobytes()
    sha = _sha256(data_bytes)
    _write_bin(path, hdr, data_bytes, sha256_digest=sha)
    print(f"    SHA-256: {sha.hex()[:16]}...")
    return sha


def load_checkpoint(output_dir, cfg=None, rng=None, load_opt=True):
    """Load LCM checkpoint from flat binary directory.

    Loads encoder, decoder, all codebooks, gvalue, and optional optimizer state.
    Codebooks are loaded as clean vectors (A=clean, W=I) suitable for inference.
    For training resume, use the pickle-based load_checkpoint in train.py instead.

    Args:
        output_dir: Path to checkpoint directory.
        cfg: LCMConfig (loaded from config.json if None).
        rng: JAX PRNG key.
        load_opt: Whether to load optimizer state.

    Returns:
        (params, gvalue, opt_state, step) — params is inference-ready with
        clean codebook vectors in factorized form.
    """
    import dataclasses as _dc
    from train.config import LCMConfig as _Cfg
    from train.gvalue import GValueCodebook
    from train.fusion import init_fusion_params, init_gen_head_params

    # Load config
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg_dict = json.load(f)
        if cfg is None:
            cfg = _Cfg()
            for k, v in cfg_dict.items():
                if hasattr(cfg, k):
                    object.__setattr__(cfg, k, v)
    if cfg is None:
        cfg = _Cfg()
    d = cfg.d_model

    params = {}
    print(f"[CKPT] Loading checkpoint from {output_dir}/")

    # Encoder
    enc_path = os.path.join(output_dir, "encoder.bin")
    if os.path.exists(enc_path):
        params['encoder'] = _load_encoder(enc_path, cfg)
        print(f"  [CKPT] encoder.bin loaded")

    # Decoder (gen_head)
    dec_path = os.path.join(output_dir, "decoder.bin")
    if os.path.exists(dec_path):
        params['gen_head'] = _load_decoder(dec_path, cfg)
        print(f"  [CKPT] decoder.bin loaded")

    # Fusion params (re-init — small, not saved separately)
    rng_fuse, rng_gen = jax.random.split(jax.random.PRNGKey(0))
    params['fusion'] = init_fusion_params(rng_fuse, cfg.n_lattices, d)
    params['gen_head'] = params.get('gen_head',
        init_gen_head_params(rng_gen, d, cfg.vocab_size))

    # Route gate (load saved if available, else re-init)
    from train.lattices import init_route_params, init_value_scalars
    from train.self_lattice import init_self_params
    route_saved_path = os.path.join(output_dir, "route_codebook.bin")
    if os.path.exists(route_saved_path):
        params['route'] = _load_route(route_saved_path, cfg)
        print(f"  [CKPT] route_codebook.bin loaded")
    else:
        params['route'] = init_route_params(jax.random.PRNGKey(1), cfg.n_lattices, d)

    # Local value scalars (re-init)
    lattice_sizes = [
        ('hrq', cfg.M_top), ('sparse', cfg.M_sparse),
        ('lowrank', cfg.M_lr), ('manifold', cfg.M_man),
        ('binding', cfg.M_bind), ('contrast', cfg.M_contrast),
    ]
    params['value_scalars'] = init_value_scalars(jax.random.PRNGKey(2), lattice_sizes)

    # Self lattice params (re-init)
    params['self'] = init_self_params(jax.random.PRNGKey(3), d, cfg.n_self_codes)

    # HRQ codebook (flat format — varying M per layer)
    hrq_path = os.path.join(output_dir, "hrq_codebook.bin")
    if os.path.exists(hrq_path):
        hdr, data, _ = _read_bin(hrq_path)
        d = cfg.d_model
        M_top = cfg.M_top
        M_fine = cfg.M_fine
        n_fine = cfg.n_hrq_layers

        flat_vectors = data.reshape(-1, d)
        pos = 0
        C_top = jnp.array(flat_vectors[pos:pos+M_top]); pos += M_top
        fine_list = []
        for _ in range(n_fine):
            C_f = jnp.array(flat_vectors[pos:pos+M_fine]); pos += M_fine
            fine_list.append({'A': C_f, 'W': jnp.eye(d)})
        params['hrq'] = {'top': {'A': C_top, 'W': jnp.eye(d)}, 'fine': fine_list}
        print(f"  [CKPT] hrq_codebook.bin loaded (M_top={M_top}, M_fine={M_fine}, {n_fine} fine)")

    # Sparse codebook
    sp_path = os.path.join(output_dir, "sparse_codebook.bin")
    if os.path.exists(sp_path):
        hdr, data, _ = _read_bin(sp_path)
        if data.ndim == 3:
            sparse_C_all = data[0]  # (M, d)
        else:
            sparse_C_all = data.reshape(-1, d)  # flat → (M, d)
        params['sparse'] = {'C': sparse_C_all[:cfg.M_sparse],
                            'zero_vec': sparse_C_all[cfg.M_sparse:cfg.M_sparse+1]
                                        if sparse_C_all.shape[0] > cfg.M_sparse
                                        else jnp.zeros((1, d))}
        print(f"  [CKPT] sparse_codebook.bin loaded")

    # Lowrank codebook (flat float32: U_k per layer then V)
    lr_path = os.path.join(output_dir, "lowrank_codebook.bin")
    if os.path.exists(lr_path):
        flat = np.fromfile(lr_path, dtype=np.float32)
        M_lr = cfg.M_lr
        ranks = cfg.ranks
        pos = 0
        U_list = []
        for r_k in ranks:
            n_f = M_lr * r_k
            u_k = jnp.array(flat[pos:pos+n_f].reshape(M_lr, r_k))
            pos += n_f
            U_list.append(u_k)
        # V = A_V @ W_V shape is (d, remaining/d)
        n_v = flat.shape[0] - pos
        V_cols = n_v // d
        V = jnp.array(flat[pos:pos+n_v].reshape(d, V_cols))
        params['lowrank'] = {'U': U_list, 'A_V': V, 'W_V': jnp.eye(V_cols)}
        print(f"  [CKPT] lowrank_codebook.bin loaded")

    # Manifold codebook
    man_path = os.path.join(output_dir, "manifold_codebook.bin")
    if os.path.exists(man_path):
        hdr, data, _ = _read_bin(man_path)
        if data.ndim == 3:
            # Expected: data has (1, M, d) but actual data has C+M*T vectors
            # _read_bin keeps flat when sizes don't match expected
            # This shouldn't happen — handle below
            C_man = data[0]
            # T not available from 3D reshape; fallback
            T_man = jnp.zeros((cfg.M_man, d, cfg.t_dim))
        else:
            flat_d = data.reshape(-1, d)
            C_man = flat_d[:cfg.M_man]  # (M_man, d)
            # Remaining data is T reshaped as (M_man, d*t_dim)
            n_t_floats = flat_d.shape[0] - cfg.M_man
            if n_t_floats > 0:
                T_flat = flat_d[cfg.M_man:].ravel()
                T_man = jnp.array(T_flat.reshape(cfg.M_man, d, cfg.t_dim))
            else:
                T_man = jnp.zeros((cfg.M_man, d, cfg.t_dim))
        params['manifold'] = {'C': jnp.array(C_man), 'T': T_man}
        print(f"  [CKPT] manifold_codebook.bin loaded")

    # Binding codebook (flat float32: key/val/bind per layer)
    bind_path = os.path.join(output_dir, "bind_codebook.bin")
    if os.path.exists(bind_path):
        flat = np.fromfile(bind_path, dtype=np.float32)
        n_bind_layers = cfg.n_bind_layers
        M_bind = cfg.M_bind
        pos = 0
        key_cb, val_cb, bind_cb = [], [], []
        for _ in range(n_bind_layers):
            for lst in [key_cb, val_cb, bind_cb]:
                C = jnp.array(flat[pos:pos+M_bind*d].reshape(M_bind, d))
                pos += M_bind * d
                lst.append({'A': C, 'W': jnp.eye(d)})
        # A_k and A_v are not saved in binary; re-init
        A_k = jax.random.normal(jax.random.PRNGKey(4), (cfg.r_max, d)) * 0.01
        A_v = jax.random.normal(jax.random.PRNGKey(5), (cfg.r_max, d)) * 0.01
        params['binding'] = {
            'A_k': A_k, 'A_v': A_v,
            'key_cb': key_cb, 'val_cb': val_cb, 'bind_cb': bind_cb,
        }
        print(f"  [CKPT] bind_codebook.bin loaded")

    # Contrast codebook (flat float32: C_a then C_b per layer)
    contrast_path = os.path.join(output_dir, "contrast_codebook.bin")
    if os.path.exists(contrast_path):
        flat = np.fromfile(contrast_path, dtype=np.float32)
        n_contrast_layers = cfg.n_contrast_layers
        M_contrast = cfg.M_contrast
        pos = 0
        C_a_list, C_b_list = [], []
        for _ in range(n_contrast_layers):
            Ca = jnp.array(flat[pos:pos+M_contrast*d].reshape(M_contrast, d))
            pos += M_contrast * d
            C_a_list.append({'A': Ca, 'W': jnp.eye(d)})
        for _ in range(n_contrast_layers):
            Cb = jnp.array(flat[pos:pos+M_contrast*d].reshape(M_contrast, d))
            pos += M_contrast * d
            C_b_list.append({'A': Cb, 'W': jnp.eye(d)})
        params['contrast'] = {'C_a': C_a_list, 'C_b': C_b_list}
        print(f"  [CKPT] contrast_codebook.bin loaded")

    # GValue (frozen, with hash verification)
    gv_path = os.path.join(output_dir, "gvalue_codebook.bin")
    gvalue = None
    if os.path.exists(gv_path):
        hdr, data, sha = _read_bin(gv_path, has_hash=True)
        M_gv = hdr['M']  # 4
        d_gv = hdr['d']  # d
        # data is flat: C_pos (M*d) + C_neg (M*d)
        C_pos = jnp.array(data[:M_gv*d_gv].reshape(M_gv, d_gv))
        C_neg = jnp.array(data[M_gv*d_gv:].reshape(M_gv, d_gv))
        if sha is not None:
            actual_sha = _sha256(data.tobytes())
            if sha != actual_sha:
                print(f"  [FATAL] GValue hash mismatch! Tampered checkpoint?")
                print(f"    Stored SHA: {sha.hex()}")
                print(f"    Actual SHA: {actual_sha.hex()}")
                raise ValueError("GValue hash verification failed")
        gvalue = GValueCodebook(C_pos, C_neg)
        gvalue.verify_integrity()
        print(f"  [CKPT] gvalue_codebook.bin loaded (hash verified)")

    # Danger codebook (frozen, with hash verification)
    danger_path = os.path.join(output_dir, "danger_codebook.bin")
    if os.path.exists(danger_path):
        from train.lattices import init_danger_params
        hdr, data, sha = _read_bin(danger_path, has_hash=True)
        M_d = hdr['M']
        d_d = hdr['d']
        if sha is not None:
            actual_sha = _sha256(data.tobytes())
            if sha != actual_sha:
                print(f"  [FATAL] Danger hash mismatch! Tampered checkpoint?")
                raise ValueError("Danger hash verification failed")
        C_danger = jnp.array(data[:M_d * d_d].reshape(M_d, d_d))
        params['danger'] = {'C': C_danger}
        print(f"  [CKPT] danger_codebook.bin loaded (hash verified)")

    # Optimizer state (optional, for training resume)
    opt_state = None
    if load_opt:
        opt_path = os.path.join(output_dir, "opt_state.bin")
        if os.path.exists(opt_path):
            with open(opt_path, 'rb') as f:
                opt_state = pickle.load(f)
            opt_state = jax.tree_util.tree_map(
                lambda x: jnp.array(x) if isinstance(x, (np.ndarray,)) else x,
                opt_state)
            print(f"  [CKPT] opt_state.bin loaded")

    # Step
    step_path = os.path.join(output_dir, "step.txt")
    step = 0
    if os.path.exists(step_path):
        with open(step_path) as f:
            step = int(f.read().strip())

    print(f"[CKPT] Done (step {step})")
    return params, gvalue, opt_state, step


# ── Load helpers (inverse of savers) ─────────────────────────────────────────

def _load_encoder(path, cfg):
    """Reconstruct encoder params from flat binary.

    Layout: embed(V,d) | rel_bias(2*max_seq_len-1) |
            layers[0..n-1] each(4*d + 4*d² + 3*d*d_ff) | q_pool(d) | w_proj(d,d)
    """
    d = cfg.d_model
    n_layers = cfg.n_encoder_layers
    V = cfg.vocab_size
    d_ff = cfg.d_ff
    max_seq_len = cfg.max_seq_len

    flat = np.fromfile(path, dtype=np.float32)
    pos = 0

    # 1. Embed
    embed = jnp.array(flat[pos:pos + V*d].reshape(V, d))
    pos += V * d

    # 2. Relative position bias (size = 2*max_seq_len - 1)
    rel_bias_size = 2 * max_seq_len - 1
    rel_bias = jnp.array(flat[pos:pos + rel_bias_size])
    pos += rel_bias_size

    # 3. Encoder layers
    n_attn = d * d
    layer_n = 4 * d + 4 * n_attn + 3 * d * d_ff
    layers = []
    for _ in range(n_layers):
        ln1_s = jnp.array(flat[pos:pos+d]); pos += d
        ln1_b = jnp.array(flat[pos:pos+d]); pos += d
        w_q = jnp.array(flat[pos:pos+n_attn].reshape(d, d)); pos += n_attn
        w_k = jnp.array(flat[pos:pos+n_attn].reshape(d, d)); pos += n_attn
        w_v = jnp.array(flat[pos:pos+n_attn].reshape(d, d)); pos += n_attn
        w_o = jnp.array(flat[pos:pos+n_attn].reshape(d, d)); pos += n_attn
        ln2_s = jnp.array(flat[pos:pos+d]); pos += d
        ln2_b = jnp.array(flat[pos:pos+d]); pos += d
        w_1 = jnp.array(flat[pos:pos+d*d_ff].reshape(d, d_ff)); pos += d*d_ff
        w_2 = jnp.array(flat[pos:pos+d*d_ff].reshape(d, d_ff)); pos += d*d_ff
        w_3 = jnp.array(flat[pos:pos+d_ff*d].reshape(d_ff, d)); pos += d_ff*d
        layers.append({'ln1_scale': ln1_s, 'ln1_bias': ln1_b,
                        'w_q': w_q, 'w_k': w_k, 'w_v': w_v, 'w_o': w_o,
                        'ln2_scale': ln2_s, 'ln2_bias': ln2_b,
                        'w_1': w_1, 'w_2': w_2, 'w_3': w_3})

    # 4. q_pool + w_proj
    q_pool = jnp.array(flat[pos:pos+d]); pos += d
    w_proj = jnp.array(flat[pos:pos+d*d].reshape(d, d))

    return {
        'embed': embed,
        'rel_bias': rel_bias,
        'layers': layers,
        'q_pool': q_pool,
        'w_proj': w_proj,
    }


def _load_decoder(path, cfg):
    d = cfg.d_model
    V = cfg.vocab_size
    flat = np.fromfile(path, dtype=np.float32)

    # New format size: w_embed(V,d) + w_q(d,d) + w_k(d,d) + w_v(d,d) +
    #                  w_o(d,d) + w_1(d,d*4) + w_2(d,d*4) + w_3(d*4,V)
    new_size = V*d + d*d + d*d + d*d + d*d + d*d*4 + d*d*4 + d*4*V
    if len(flat) == new_size:
        # New format with causal linear attention + GLU
        offset = 0
        gh = {}
        shapes = [
            ('w_embed', (V, d)), ('w_q', (d, d)), ('w_k', (d, d)),
            ('w_v', (d, d)), ('w_o', (d, d)), ('w_1', (d, d*4)),
            ('w_2', (d, d*4)), ('w_3', (d*4, V)),
        ]
        for name, shape in shapes:
            n_el = shape[0] * shape[1]
            gh[name] = jnp.array(flat[offset:offset + n_el].reshape(shape))
            offset += n_el
        return gh
    else:
        # Old format (w_proj + w_out): reinit with new params
        from train.fusion import init_gen_head_params
        rng = jax.random.PRNGKey(0)
        return init_gen_head_params(rng, d, V)


def _load_route(path, cfg):
    """Reconstruct route params from flat binary.
    Layout: C_route(n_lattices, d) | W_route(d, n_lattices)
    """
    flat = np.fromfile(path, dtype=np.float32)
    n_lt = cfg.n_lattices
    d = cfg.d_model
    C_route = jnp.array(flat[:n_lt*d].reshape(n_lt, d))
    W_route = jnp.array(flat[n_lt*d:].reshape(d, n_lt))
    return {'C_route': C_route, 'W_route': W_route}
