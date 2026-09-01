"""Export cog_train checkpoint → C inference engine format (encoder.bin, decoder.bin, codebooks)."""

import hashlib
import json
import os
import pickle
import struct
import sys
import zlib

import numpy as np

# Match checkpoint.py 24-byte header format: <iiiifI = M,d,n_layers,type,curvature,crc
HEADER_FMT = '<iiiifI'
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def _pack_header(M, d, n_layers, cb_type, curvature=1.0):
    return struct.pack(HEADER_FMT, int(M), int(d), int(n_layers), int(cb_type), float(curvature), 0)


def _write_cb(path, arrays, cb_type=1, curvature=1.0):
    """Write codebook .bin file: 24-byte header + data + CRC."""
    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    M = arrays[0].shape[0]
    d = arrays[0].shape[1]
    n_layers = len(arrays)
    data_bytes = b''.join(a.tobytes() for a in arrays)
    crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
    hdr = struct.pack(HEADER_FMT, int(M), int(d), int(n_layers), int(cb_type), float(curvature), crc)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(data_bytes)


def export(ckpt_dir: str, out_dir: str, data_dir: str = "data"):
    """Export cog_train checkpoint to C inference format.

    Args:
        ckpt_dir: Directory containing cog_params.pkl.
        out_dir: Output directory for inference checkpoint.
        data_dir: Directory containing tokenizer.json.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Load params ────────────────────────────────────────────────────
    ckpt_path = os.path.join(ckpt_dir, "cog_params.pkl")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    params = ckpt["params"] if "params" in ckpt else ckpt
    step = ckpt.get("step", 0)
    print(f"[EXPORT] Loading params from {ckpt_path} (step {step})")

    d_model = params["W_out"].shape[0]
    vocab_size = params["W_out"].shape[1]

    # ── config.json ────────────────────────────────────────────────────
    cfg = {
        "d_model": d_model,
        "vocab_size": vocab_size,
        "max_seq_len": 512,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_lattices": 6,
        "d_ff": int(1.5 * d_model),
        # codebook sizes (hardcoded defaults — override if needed)
        "M_top": params.get("hrq", {}).get("top", {}).get("A", np.zeros((512, 1))).shape[0],
        "M_fine": 256,
        "n_hrq_layers": len(params.get("hrq", {}).get("fine", [])),
        "M_sparse": params.get("sparse", {}).get("C", np.zeros((512, 1))).shape[0],
        "M_lr": params.get("lowrank", {}).get("A_V", np.zeros((256, 1))).shape[0],
        "n_lr_layers": 3,
        "M_man": params.get("manifold", {}).get("C", np.zeros((512, 1))).shape[0],
        "t_dim": 4,
        "M_bind": 512,
        "n_bind_layers": 3,
        "M_contrast": 512,
        "n_contrast_layers": 3,
        "r_max": 8,
        "n_value_pairs": 4,
        "M_danger": 256,
        "n_self_codes": params.get("self", {}).get("modes", np.zeros((64, 1))).shape[0],
        "max_inference_steps": 32,
        "convergence_tol": 1e-3,
        "entropy_threshold": 2.0,
        "tau_route": 0.5,
        "beta_vq": 0.25,
        "gamma_sparse": 0.99,
        "gamma_man": 0.99,
        "gamma_bind": 0.99,
    }
    # Fill actual sizes from params
    hrq_top = params.get("hrq", {}).get("top", {})
    if hrq_top:
        cfg["M_top"] = hrq_top.get("A", np.zeros((1, 1))).shape[0] if "A" in hrq_top else 512
    hrq_fine = params.get("hrq", {}).get("fine", [])
    if hrq_fine:
        cfg["n_hrq_layers"] = len(hrq_fine)
        cfg["M_fine"] = hrq_fine[0].get("A", np.zeros((1, 1))).shape[0] if "A" in hrq_fine[0] else 256
    sparse = params.get("sparse", {})
    if sparse and "C" in sparse:
        cfg["M_sparse"] = sparse["C"].shape[0]
    lowrank = params.get("lowrank", {})
    if lowrank and "A_V" in lowrank:
        cfg["M_lr"] = lowrank["A_V"].shape[0]
    manifold = params.get("manifold", {})
    if manifold and "C" in manifold:
        cfg["M_man"] = manifold["C"].shape[0]
    binding = params.get("binding", {})
    if binding and "key_cb" in binding:
        cfg["n_bind_layers"] = len(binding["key_cb"])
    contrast = params.get("contrast", {})
    if contrast and "C_a" in contrast:
        cfg["n_contrast_layers"] = len(contrast["C_a"])
    self_p = params.get("self", {})
    if self_p and "modes" in self_p:
        cfg["n_self_codes"] = self_p["modes"].shape[0]

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[EXPORT] config.json → {out_dir}/")

    # ── encoder.bin ────────────────────────────────────────────────────
    enc = params.get("encoder", {})
    if not enc:
        print("[EXPORT] WARN: no encoder params found.  Generating dummy.")
        n_layers = cfg["n_encoder_layers"]
        d = cfg["d_model"]
        d_ff = cfg["d_ff"]
        V = cfg["vocab_size"]
        enc = {
            "embed": np.random.randn(V, d).astype(np.float32) * 0.02,
            "rel_bias": np.random.randn(2 * cfg["max_seq_len"] - 1).astype(np.float32) * 0.01,
            "layers": [
                {
                    "ln1_scale": np.ones(d, dtype=np.float32),
                    "ln1_bias": np.zeros(d, dtype=np.float32),
                    "w_q": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "w_k": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "w_v": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "w_o": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
                    "ln2_scale": np.ones(d, dtype=np.float32),
                    "ln2_bias": np.zeros(d, dtype=np.float32),
                    "w_1": np.random.randn(d, d_ff).astype(np.float32) * (d ** -0.5),
                    "w_2": np.random.randn(d, d_ff).astype(np.float32) * (d ** -0.5),
                    "w_3": np.random.randn(d_ff, d).astype(np.float32) * (d_ff ** -0.5),
                }
                for _ in range(n_layers)
            ],
            "q_pool": np.random.randn(d).astype(np.float32) * 0.01,
            "w_proj": np.random.randn(d, d).astype(np.float32) * (d ** -0.5),
        }

    def _to_np(x):
        return np.array(x) if hasattr(x, "numpy") else x

    enc_np = {k: _to_np(v) for k, v in enc.items()}
    parts = [enc_np["embed"].ravel()]
    parts.append(enc_np["rel_bias"].ravel())
    for layer in enc_np["layers"]:
        for key in ["ln1_scale", "ln1_bias", "w_q", "w_k", "w_v", "w_o",
                     "ln2_scale", "ln2_bias", "w_1", "w_2", "w_3"]:
            parts.append(_to_np(layer[key]).ravel())
    parts.append(enc_np["q_pool"].ravel())
    parts.append(enc_np["w_proj"].ravel())

    encoder_flat = np.concatenate(parts).astype(np.float32)
    encoder_flat.tofile(os.path.join(out_dir, "encoder.bin"))
    print(f"[EXPORT] encoder.bin ({encoder_flat.nbytes / 1e6:.1f} MB) → {out_dir}/")

    # ── decoder.bin ────────────────────────────────────────────────────
    gen_head = params.get("gen_head", {})
    if gen_head and "w_embed" in gen_head:
        # new format decoder
        d = cfg["d_model"]
        V = cfg["vocab_size"]
        parts = []
        parts.append(_to_np(gen_head["w_embed"]).ravel())
        for key in ["w_q", "w_k", "w_v", "w_o"]:
            parts.append(_to_np(gen_head[key]).ravel())
        for key in ["w_1", "w_2"]:
            parts.append(_to_np(gen_head[key]).ravel())
        parts.append(_to_np(gen_head["w_3"]).ravel())
        decoder_flat = np.concatenate(parts).astype(np.float32)
    elif "W_out" in params:
        # cog format: bare W_out (d×V) → loads as 'cog' with a linear
        # readout (z @ W_out), matching save_cog_checkpoint. The old
        # identity+w_proj concatenation loaded as 'old' format, which
        # applies an ELU the training loss never saw.
        W_out = _to_np(params["W_out"])
        decoder_flat = W_out.ravel().astype(np.float32)
    else:
        print("[EXPORT] WARN: no decoder/W_out found.  Writing dummy.")
        decoder_flat = np.random.randn(d_model * d_model + d_model * vocab_size).astype(np.float32)

    decoder_flat.tofile(os.path.join(out_dir, "decoder.bin"))
    print(f"[EXPORT] decoder.bin ({decoder_flat.nbytes / 1e6:.1f} MB) → {out_dir}/")

    # ── codebooks ──────────────────────────────────────────────────────
    def _simvq(simvq):
        return _to_np(simvq['A']) @ _to_np(simvq['W'])

    # HRQ: one flat file, all layers concat (header + data)
    hrq = params.get('hrq', {})
    if hrq and 'top' in hrq:
        layers_hrq = [_simvq(hrq['top'])]
        for fb in hrq.get('fine', []):
            layers_hrq.append(_simvq(fb))
        _write_cb(os.path.join(out_dir, "hrq_codebook.bin"), layers_hrq, cb_type=1)

    # Sparse: header + C
    sparse = params.get('sparse', {})
    if sparse and 'C' in sparse:
        _write_cb(os.path.join(out_dir, "sparse_codebook.bin"), [_to_np(sparse['C'])], cb_type=1)

    # LowRank: flat file (U_0..U_k + V), no header
    lr = params.get('lowrank', {})
    if lr and 'A_V' in lr and 'W_V' in lr and 'U' in lr:
        V_lr = _to_np(lr['A_V']) @ _to_np(lr['W_V'])
        parts_lr = []
        for U in lr['U']:
            parts_lr.append(_to_np(U).ravel())  # U itself (M_lr, r_k) not the product
        parts_lr.append(V_lr.ravel())           # (d, d)
        np.concatenate(parts_lr).astype(np.float32).tofile(
            os.path.join(out_dir, "lowrank_codebook.bin"))

    # Manifold: header + C + T
    manifold = params.get('manifold', {})
    if manifold and 'C' in manifold and 'T' in manifold:
        C_m = _to_np(manifold['C'])
        T_m = _to_np(manifold['T'].reshape(C_m.shape[0], -1))
        _write_cb(os.path.join(out_dir, "manifold_codebook.bin"), [C_m, T_m], cb_type=2)

    # Binding: single flat file (key_0, val_0, bind_0, key_1, ...)
    binding = params.get('binding', {})
    if binding and 'key_cb' in binding:
        parts_bind = []
        for l in range(len(binding['key_cb'])):
            for cb_type_name in ['key_cb', 'val_cb', 'bind_cb']:
                C = _simvq(binding[cb_type_name][l])
                parts_bind.append(C.ravel())
        np.concatenate(parts_bind).astype(np.float32).tofile(
            os.path.join(out_dir, "bind_codebook.bin"))

    # Contrast: single flat file (C_a_0..C_a_n, C_b_0..C_b_n)
    contrast = params.get('contrast', {})
    if contrast and 'C_a' in contrast:
        parts_ct = []
        for ca in contrast['C_a']:
            parts_ct.append(_simvq(ca).ravel())
        for cb in contrast['C_b']:
            parts_ct.append(_simvq(cb).ravel())
        np.concatenate(parts_ct).astype(np.float32).tofile(
            os.path.join(out_dir, "contrast_codebook.bin"))

    print(f"[EXPORT] Codebooks → {out_dir}/")

    # ── tokenizer.json (copy) ──────────────────────────────────────────
    import shutil
    tok_src = os.path.join(data_dir, "tokenizer.json")
    tok_dst = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_src):
        shutil.copy2(tok_src, tok_dst)
        print(f"[EXPORT] tokenizer.json → {out_dir}/")
    else:
        print(f"[EXPORT] WARN: tokenizer.json not found at {tok_src}")

    # ── gvalue codebooks ───────────────────────────────────────────────
    from train.gvalue import make_global_value_vectors
    C_pos, C_neg = make_global_value_vectors(d_model)
    C_p = np.asarray(C_pos, dtype=np.float32) if hasattr(C_pos, 'numpy') else np.array(C_pos, dtype=np.float32)
    # Placeholder: identical halves → the C engine's margin check never fires
    # (see cog_train export). Real anchors are NOT trained yet.
    C_n = C_p.copy()
    data_gv = C_p.tobytes() + C_n.tobytes()
    sha_gv = hashlib.sha256(data_gv).digest()
    M_gv = C_p.shape[0]
    crc_gv = zlib.crc32(data_gv) & 0xFFFFFFFF
    hdr_gv = struct.pack(HEADER_FMT, int(M_gv), int(d_model), 1, 2, 1.0, crc_gv)
    with open(os.path.join(out_dir, "gvalue_codebook.bin"), "wb") as f:
        f.write(hdr_gv)
        f.write(data_gv)
        f.write(sha_gv)
    print(f"[EXPORT] gvalue_codebook.bin → {out_dir}/")

    # danger codebook (dummy placeholder, values NOT trained; the danger
    # lattice is not part of cognitive training yet). Identical halves with a
    # fixed seed → danger_score ≡ 0 → no spurious blocks, deterministic export.
    # Matches checkpoint._save_danger and cog_train's export.
    M_danger = cfg.get('M_danger', 256)
    np.random.seed(0)
    danger_t = np.random.randn(M_danger, d_model).astype(np.float32) * 0.02
    danger_n = danger_t.copy()
    data_danger = danger_t.tobytes() + danger_n.tobytes()
    sha_danger = hashlib.sha256(data_danger).digest()
    crc_danger = zlib.crc32(data_danger) & 0xFFFFFFFF
    hdr_danger = struct.pack(HEADER_FMT, int(M_danger), int(d_model), 1, 2, 1.0, crc_danger)
    with open(os.path.join(out_dir, 'danger_codebook.bin'), 'wb') as f:
        f.write(hdr_danger)
        f.write(data_danger)
        f.write(sha_danger)

    print(f"[EXPORT] Done → {out_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export cog_train checkpoint → C inference format")
    p.add_argument("ckpt_dir", help="Path to cog_train checkpoint directory")
    p.add_argument("-o", "--out", default=None, help="Output directory (default: ckpt_dir + _infer)")
    p.add_argument("--data-dir", default="data", help="Data directory (for tokenizer.json)")
    args = p.parse_args()
    out = args.out or args.ckpt_dir.rstrip("/") + "_infer"
    export(args.ckpt_dir, out, args.data_dir)