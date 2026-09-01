# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
#
# LCM accelerated routines — Cython hot-path for cache matching, edge counting,
# gen_head forward, incremental encoder, and categorical sampling.
#
# Build with: python setup.py build_ext --inplace
# Pure-Python fallbacks live in the respective callers.

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt, fabs

# ---------------------------------------------------------------------------
# popcount for 64-bit integers (portable C fallback)
# ---------------------------------------------------------------------------

cdef inline int popcount64(long long x) nogil:
    cdef int c = 0
    while x:
        c += 1
        x &= x - 1
    return c


# ---------------------------------------------------------------------------
# match_euclidean_cy
#
# Scan flat buffer, compute squared L2, pick K smallest, distance-weighted
# fusion of corresponding z_next vectors.
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def match_euclidean_cy(
        np.ndarray[float, ndim=2]  z_cur_buf not None,   # (N, D)
        np.ndarray[float, ndim=2]  z_next_buf not None,  # (N, D)
        np.ndarray[np.uint8_t, ndim=1] valid not None,   # (N,)
        np.ndarray[float, ndim=1]  query not None,        # (D,)
        int K):
    cdef int N = z_cur_buf.shape[0]
    cdef int D = z_cur_buf.shape[1]
    cdef int i, j, nv, ai, K_actual
    cdef float diff, d
    cdef double wsum, w

    # scratch arrays
    cdef np.ndarray[float, ndim=1] dists = np.empty(N, dtype=np.float32)
    cdef np.ndarray[np.intp_t, ndim=1] idxs = np.empty(N, dtype=np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] top

    nv = 0
    for i in range(N):
        if not valid[i]:
            continue
        d = 0.0
        for j in range(D):
            diff = z_cur_buf[i, j] - query[j]
            d += diff * diff
        dists[nv] = d
        idxs[nv] = i
        nv += 1

    if nv == 0:
        return None

    K_actual = K if K < nv else nv
    if K_actual <= 0:
        return None
    top = np.argpartition(dists[:nv], K_actual - 1)[:K_actual]
    top = top[np.argsort(dists[:nv][top])]

    cdef np.ndarray[float, ndim=1] z_pred = np.zeros(D, dtype=np.float32)
    wsum = 0.0
    for i in range(K_actual):
        d = dists[top[i]] + 1e-8
        w = 1.0 / d
        wsum += w
        ai = idxs[top[i]]
        for j in range(D):
            z_pred[j] += w * z_next_buf[ai, j]

    if wsum > 0:
        for j in range(D):
            z_pred[j] /= wsum
    return z_pred


# ---------------------------------------------------------------------------
# match_hamming_cy
#
# Hamming-distance filter on sig, then Euclidean distance on z_cur, then
# K-NN weighted fusion of z_next.
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def match_hamming_cy(
        np.ndarray[float, ndim=2]  z_cur_buf not None,
        np.ndarray[float, ndim=2]  z_next_buf not None,
        np.ndarray[np.int64_t, ndim=1] sig_buf not None,
        np.ndarray[np.uint8_t, ndim=1] valid not None,
        np.ndarray[float, ndim=1]  query not None,
        long long sig,
        int max_dist,
        int K):
    cdef int N = z_cur_buf.shape[0]
    cdef int D = z_cur_buf.shape[1]
    cdef int i, j, nv, ai, K_actual
    cdef float diff, d
    cdef long long xor_, hd
    cdef double wsum, w

    cdef np.ndarray[float, ndim=1] dists = np.empty(N, dtype=np.float32)
    cdef np.ndarray[np.intp_t, ndim=1] idxs = np.empty(N, dtype=np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] top

    nv = 0
    for i in range(N):
        if not valid[i]:
            continue
        xor_ = sig_buf[i] ^ sig
        hd = popcount64(xor_)
        if hd <= max_dist:
            d = 0.0
            for j in range(D):
                diff = z_cur_buf[i, j] - query[j]
                d += diff * diff
            dists[nv] = d
            idxs[nv] = i
            nv += 1

    if nv == 0:
        return None

    K_actual = K if K < nv else nv
    if K_actual <= 0:
        return None
    top = np.argpartition(dists[:nv], K_actual - 1)[:K_actual]
    top = top[np.argsort(dists[:nv][top])]

    cdef np.ndarray[float, ndim=1] z_pred = np.zeros(D, dtype=np.float32)
    wsum = 0.0
    for i in range(K_actual):
        d = dists[top[i]] + 1e-8
        w = 1.0 / d
        wsum += w
        ai = idxs[top[i]]
        for j in range(D):
            z_pred[j] += w * z_next_buf[ai, j]

    if wsum > 0:
        for j in range(D):
            z_pred[j] /= wsum
    return z_pred


# ---------------------------------------------------------------------------
# match_exact_cy
#
# Exact sig filter + Euclidean argmin. Returns (z_next, distance) or None.
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def match_exact_cy(
        np.ndarray[float, ndim=2]  z_cur_buf not None,
        np.ndarray[float, ndim=2]  z_next_buf not None,
        np.ndarray[np.int64_t, ndim=1] sig_buf not None,
        np.ndarray[np.uint8_t, ndim=1] valid not None,
        np.ndarray[float, ndim=1]  query not None,
        long long sig):
    cdef int N = z_cur_buf.shape[0]
    cdef int D = z_cur_buf.shape[1]
    cdef int i, j, best_i = -1
    cdef float diff, d, best_d = 1e38

    for i in range(N):
        if not valid[i] or sig_buf[i] != sig:
            continue
        d = 0.0
        for j in range(D):
            diff = z_cur_buf[i, j] - query[j]
            d += diff * diff
        if d < best_d:
            best_d = d
            best_i = i

    if best_i < 0:
        return None
    return z_next_buf[best_i].copy(), best_d


# ---------------------------------------------------------------------------
# count_edges_cy
#
# Count internal / total causal edges within a sliding window.
# Returns (n_internal, n_total).
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def count_edges_cy(
        np.ndarray[np.int64_t, ndim=1] effect_steps not None,
        np.ndarray[np.uint8_t, ndim=1] internal_flags not None,
        np.ndarray[np.uint8_t, ndim=1] valid not None,
        int window,
        int current_step):
    cdef int N = len(effect_steps)
    cdef int cutoff = current_step - window
    cdef int n_internal = 0, n_total = 0
    cdef int i

    for i in range(N):
        if not valid[i]:
            continue
        if effect_steps[i] >= cutoff:
            n_total += 1
            if internal_flags[i]:
                n_internal += 1
    return n_internal, n_total


# ─── PRNG for sampling (xorshift32, module-local) ─────────────────────────────

cdef unsigned int _rng_state = 2463534242
cdef unsigned int _rng_seeded = 0

cdef inline float _rand_f32() nogil:
    """Uniform float32 in (0, 1] using xorshift32."""
    global _rng_state
    _rng_state ^= _rng_state << 13
    _rng_state ^= _rng_state >> 17
    _rng_state ^= _rng_state << 5
    return (<float>_rng_state + 1.0) / 4294967296.0  # (0, 1]


def init_rng_cy(unsigned int seed):
    """Seed the module-level PRNG (call once at startup; later calls no-op).

    Guarded: repeated calls must NOT reset the state — reseeding with 0 would
    put xorshift32 into its absorbing state and make _rand_f32() constant.
    Seed 0 is mapped to a nonzero default for the same reason.
    """
    global _rng_state, _rng_seeded
    if not _rng_seeded:
        _rng_state = seed if seed != 0 else 2463534242
        _rng_seeded = 1


# ─── sample_categorical_cy — fused top‑k + softmax + CDF walk ────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_categorical_cy(
        np.ndarray[float, ndim=1] logits,
        float temperature,
        int top_k):
    """Sample token ID from logits with temperature + top-k.

    Fuses softmax (exp → sum → normalize) and CDF-walk sampling into a single
    pass over V.  Uses O(V) argpartition instead of O(V log V) full sort.
    Uses module-local xorshift32 PRNG — no per-call Generator allocation.
    """
    cdef int V = logits.shape[0]
    cdef int i, k, idx
    cdef float inv_temp = 1.0 / (temperature if temperature > 1e-8 else 1e-8)
    cdef float max_val = -1e38
    cdef float threshold, sum_probs, inv_sum, r, cum

    # Apply temperature
    cdef np.ndarray[float, ndim=1] scaled = np.empty(V, dtype=np.float32)
    for i in range(V):
        scaled[i] = logits[i] * inv_temp
        if scaled[i] > max_val:
            max_val = scaled[i]

    # Top-k: find k-th largest logit, zero out rest
    if 0 < top_k < V:
        k = V - top_k  # index of k-th largest in sorted order
        # Partial O(V) selection
        threshold = -np.partition(-scaled, k)[k]
        for i in range(V):
            if scaled[i] < threshold:
                scaled[i] = -1e9

    # Stable softmax: exp(x - max_val)
    sum_probs = 0.0
    for i in range(V):
        scaled[i] = exp(scaled[i] - max_val)
        sum_probs += scaled[i]

    inv_sum = 1.0 / (sum_probs + 1e-30)

    # CDF walk sampling
    r = _rand_f32()
    cum = 0.0
    for i in range(V):
        cum += scaled[i] * inv_sum
        if r < cum:
            return i
    return V - 1  # fallback (should not reach)


# ─── genhead_step_cy — single gen_head forward step ──────────────────────────
#
# Fuses: embed lookup, φ-matmul+ELU, outer+cumsum, attention, GLU.
# Small (d × d) matmuls use manual C loops to avoid BLAS launch overhead.
# The final (d_ff × V) logit projection is left to the caller (@ numpy dot)
# since it dominates FLOPS and BLAS handles it optimally.
#
# Returns (gate_act) for the caller to project to vocabulary.

@cython.boundscheck(False)
@cython.wraparound(False)
def genhead_step_cy(
        np.ndarray[float, ndim=2] w_embed,       # (V, d)
        np.ndarray[float, ndim=2] w_q,            # (d, d)
        np.ndarray[float, ndim=2] w_k,            # (d, d)
        np.ndarray[float, ndim=2] w_v,            # (d, d)
        np.ndarray[float, ndim=2] w_o,            # (d, d)
        np.ndarray[float, ndim=2] w_1,            # (d, d_ff)
        np.ndarray[float, ndim=2] w_2,            # (d, d_ff)
        np.ndarray[float, ndim=1] z_q,            # (d,)
        int last_token_id,
        np.ndarray[float, ndim=2] kv_sum,         # (d, d) — modified in‑place
        np.ndarray[float, ndim=1] k_sum,          # (d,)   — modified in‑place
        int is_first,
        np.ndarray[float, ndim=1] gate_act_out):  # (d_ff,) — output (caller projects to V)
    cdef int d = w_q.shape[0]
    cdef int d_ff = w_1.shape[1]
    cdef int i, j
    cdef float s, q_dot, inv_norm

    # ── Token embed (z_q on first step, lookup thereafter) ──────────────
    cdef np.ndarray[float, ndim=1] tok = np.empty(d, dtype=np.float32)
    if is_first:
        for i in range(d):
            tok[i] = z_q[i]
    else:
        for i in range(d):
            tok[i] = w_embed[last_token_id, i]

    # ── QKV with fused φ (ELU+1) ────────────────────────────────────────
    cdef np.ndarray[float, ndim=1] q = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] k = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] v = np.empty(d, dtype=np.float32)

    # q = φ(tok @ w_q)
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += tok[j] * w_q[j, i]
        q[i] = (exp(s) if s < 0 else s + 1.0)  # ELU(x)+1 = (x<0 ? exp(x) : x+1)

    # k = φ(tok @ w_k)
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += tok[j] * w_k[j, i]
        k[i] = (exp(s) if s < 0 else s + 1.0)

    # v = tok @ w_v
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += tok[j] * w_v[j, i]
        v[i] = s

    # ── Update KV cumsum ─────────────────────────────────────────────────
    for i in range(d):
        k_sum[i] = k_sum[i] + k[i]
        for j in range(d):
            kv_sum[i, j] = kv_sum[i, j] + k[i] * v[j]

    # ── Attention: (q @ kv_sum) / (q · k_sum + 1e-8) ────────────────────
    cdef np.ndarray[float, ndim=1] attn = np.empty(d, dtype=np.float32)
    q_dot = 0.0
    for i in range(d):
        q_dot += q[i] * k_sum[i]
    inv_norm = 1.0 / (q_dot + 1e-8)
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += q[j] * kv_sum[j, i]
        attn[i] = s * inv_norm

    # attn = attn @ w_o
    cdef np.ndarray[float, ndim=1] attn_proj = np.empty(d, dtype=np.float32)
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += attn[j] * w_o[j, i]
        attn_proj[i] = s

    # ── GLU: sigmoid(attn_proj @ w_1) * (attn_proj @ w_2) ────────────────
    cdef np.ndarray[float, ndim=1] gate = np.empty(d_ff, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] act = np.empty(d_ff, dtype=np.float32)

    for i in range(d_ff):
        s = 0.0
        for j in range(d):
            s += attn_proj[j] * w_1[j, i]
        gate[i] = 1.0 / (1.0 + exp(-(s if s > -30.0 else -30.0)))  # sigmoid

    for i in range(d_ff):
        s = 0.0
        for j in range(d):
            s += attn_proj[j] * w_2[j, i]
        act[i] = s

    for i in range(d_ff):
        gate_act_out[i] = gate[i] * act[i]
    # Caller:   logits = gate_act_out @ w_3  (BLAS, large V)


# ─── encoder_recurrent_step_cy — incremental encoder, O(d²) ──────────────────

@cython.boundscheck(False)
@cython.wraparound(False)
def encoder_recurrent_step_cy(
        np.ndarray[float, ndim=2] embed,          # (V, d)
        np.ndarray[float, ndim=1] q_pool,         # (d,)
        np.ndarray[float, ndim=2] w_proj,         # (d, d)
        list layers_kv,                            # list of (d,d) arrays — per‑layer KV cumsums, modified in‑place
        list layers_k,                             # list of (d,) arrays — per‑layer K cumsums, modified in‑place
        np.ndarray[float, ndim=2] pool_kv,         # (d, d) — global pooling cumsum, modified in‑place
        np.ndarray[float, ndim=1] pool_k,          # (d,) — global pooling cumsum, modified in‑place
        list ln1_scales,                           # per‑layer ln1_scale (d,)
        list ln1_biases,
        list ln2_scales,
        list ln2_biases,
        list w_qs,                                  # per‑layer w_q (d,d)
        list w_ks,
        list w_vs,
        list w_os,
        list w_1s,                                  # per‑layer w_1 (d, d_ff)
        list w_2s,
        list w_3s,
        int token_id,
        int n_heads,
        np.ndarray[float, ndim=1] z_out):         # (d,) — output
    """Incremental encoder step: one new token, O(d²) per layer.

    Python wrapper handles dict extraction; this function operates on flat
    typed arrays for maximum speed.  All state arrays are modified in‑place.
    """
    cdef int n_layers = len(w_qs)
    cdef int d = embed.shape[1]
    cdef int d_h = d // n_heads
    cdef int l, i, j, hh, idx, h_start, d_ff
    cdef float s, q_dot, inv_norm, mean, inv_std

    cdef np.ndarray[float, ndim=1] h = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] h_norm = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] q_raw = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] k_raw = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] v_raw = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] attn = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] attn_proj = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] z_tmp = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] k_pool_raw = np.empty(d, dtype=np.float32)
    cdef np.ndarray[float, ndim=1] q_p = np.empty(d, dtype=np.float32)
    # GLU temporaries (re‑allocated per layer since d_ff may vary)
    cdef np.ndarray[float, ndim=1] glu_gate
    cdef np.ndarray[float, ndim=1] glu_act
    cdef np.ndarray[float, ndim=1] glu_out

    # h = embed[token_id]
    for i in range(d):
        h[i] = embed[token_id, i]

    for l in range(n_layers):
        # Per-layer weights (Python list lookups — fine outside typed loops)
        _ln1_s = ln1_scales[l]
        _ln1_b = ln1_biases[l]
        _ln2_s = ln2_scales[l]
        _ln2_b = ln2_biases[l]
        _w_q = w_qs[l]
        _w_k = w_ks[l]
        _w_v = w_vs[l]
        _w_o = w_os[l]
        _w_1 = w_1s[l]
        _w_2 = w_2s[l]
        _w_3 = w_3s[l]
        _kv = layers_kv[l]
        _k_sum_arr = layers_k[l]
        d_ff = _w_1.shape[1]

        glu_gate = np.empty(d_ff, dtype=np.float32)
        glu_act = np.empty(d_ff, dtype=np.float32)
        glu_out = np.empty(d, dtype=np.float32)

        # ── Pre‑LN ───────────────────────────────────────────────────────
        s = 0.0
        for i in range(d):
            s += h[i]
        mean = s / d
        s = 0.0
        for i in range(d):
            s += (h[i] - mean) * (h[i] - mean)
        inv_std = 1.0 / sqrt(s / d + 1e-6)
        for i in range(d):
            h_norm[i] = (h[i] - mean) * inv_std * _ln1_s[i] + _ln1_b[i]

        # ── Multi‑head QKV + φ ───────────────────────────────────────────
        # q = φ(h_norm @ w_q)
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += h_norm[j] * _w_q[j, i]
            q_raw[i] = (exp(s) if s < 0 else s + 1.0)

        # k = φ(h_norm @ w_k)
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += h_norm[j] * _w_k[j, i]
            k_raw[i] = (exp(s) if s < 0 else s + 1.0)

        # v = h_norm @ w_v
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += h_norm[j] * _w_v[j, i]
            v_raw[i] = s

        # ── Per-head cumsum update ───────────────────────────────────────
        # State layout: layers_kv[l] is a (d, d) block-diagonal matrix —
        # head hh occupies rows/cols [hh*d_h, (hh+1)*d_h) (see
        # lcm.py _kv_cumsum_block). layers_k[l] is the (d,) diagonal of that
        # per-head k-sum in the same block order.
        for hh in range(n_heads):
            h_start = hh * d_h
            for i in range(d_h):
                idx = h_start + i
                _k_sum_arr[idx] = _k_sum_arr[idx] + k_raw[idx]
                for j in range(d_h):
                    _kv[h_start + i, h_start + j] = _kv[h_start + i, h_start + j] + k_raw[idx] * v_raw[h_start + j]

        # ── Attention ─────────────────────────────────────────────────────
        for hh in range(n_heads):
            h_start = hh * d_h
            # numerator: q_block @ kv_block
            for i in range(d_h):
                s = 0.0
                for j in range(d_h):
                    s += q_raw[h_start + j] * _kv[h_start + j, h_start + i]
                attn[h_start + i] = s
            # denominator: q_block · k_sum_block
            q_dot = 0.0
            for i in range(d_h):
                q_dot += q_raw[h_start + i] * _k_sum_arr[h_start + i]
            inv_norm = 1.0 / (q_dot + 1e-6)
            for i in range(d_h):
                attn[h_start + i] *= inv_norm

        # attn @ w_o
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += attn[j] * _w_o[j, i]
            attn_proj[i] = s

        # Residual
        for i in range(d):
            h[i] = h[i] + attn_proj[i]

        # ── GLU with pre‑LN ──────────────────────────────────────────────
        s = 0.0
        for i in range(d):
            s += h[i]
        mean = s / d
        s = 0.0
        for i in range(d):
            s += (h[i] - mean) * (h[i] - mean)
        inv_std = 1.0 / sqrt(s / d + 1e-6)

        for i in range(d):
            h_norm[i] = (h[i] - mean) * inv_std * _ln2_s[i] + _ln2_b[i]

        # Gate + act
        for i in range(d_ff):
            s = 0.0
            for j in range(d):
                s += h_norm[j] * _w_1[j, i]
            glu_gate[i] = s / (1.0 + exp(-(s if s > -30.0 else -30.0)))  # SiLU: s * sigmoid(s)

        for i in range(d_ff):
            s = 0.0
            for j in range(d):
                s += h_norm[j] * _w_2[j, i]
            glu_act[i] = s

        # glu_out = (gate * act) @ w_3
        for i in range(d):
            s = 0.0
            for j in range(d_ff):
                s += glu_gate[j] * glu_act[j] * _w_3[j, i]
            glu_out[i] = s

        # Residual
        for i in range(d):
            h[i] = h[i] + glu_out[i]

    # ── Global attention pooling ─────────────────────────────────────────
    for i in range(d):
        k_pool_raw[i] = (exp(h[i]) if h[i] < 0 else h[i] + 1.0)

    for i in range(d):
        pool_k[i] = pool_k[i] + k_pool_raw[i]
        for j in range(d):
            pool_kv[i, j] = pool_kv[i, j] + k_pool_raw[i] * h[j]

    for i in range(d):
        q_p[i] = (exp(q_pool[i]) if q_pool[i] < 0 else q_pool[i] + 1.0)

    q_dot = 0.0
    for i in range(d):
        q_dot += q_p[i] * pool_k[i]
    inv_norm = 1.0 / (q_dot + 1e-6)

    for i in range(d):
        s = 0.0
        for j in range(d):
            s += q_p[j] * pool_kv[j, i]
        z_tmp[i] = s * inv_norm

    for i in range(d):
        s = 0.0
        for j in range(d):
            s += z_tmp[j] * w_proj[j, i]
        z_out[i] = s
