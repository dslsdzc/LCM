"""Focused C v3 loop bridge tests (task 10).

The full checkpoint/encoder stack is not needed here: `generate` is driven
through a minimal in-memory engine whose arrays are the same flat pointers
`train/c_load.py` hands the C bridge. This keeps the test fast while exercising
the real `lcm_infer_loop_v3_flat` path, including the safety abort.

Run:
    make -C infer clean && make -C infer       # or make test, then make
    python -m pytest train/test_c_loop.py -v
"""
import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lcm
from lcm import CONFLICT_ABORT_TOKEN, LCMInferEngine

D = 256
M = 2
N_LAYERS = 2
R_MAX = 4
N_LATTICES = 6
SO_PATH = Path(__file__).resolve().parent.parent / "infer" / "liblcm.so"


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class _Tokenizer:
    def encode(self, prompt):
        return SimpleNamespace(ids=[2])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(0x41 + (int(i) % 26)) for i in ids)


class _Obs:
    def record_step(self, **kwargs):
        return None


def _build_engine(danger_threat, danger_normal):
    eng = object.__new__(LCMInferEngine)
    eng.d = D
    eng.V = 8
    eng.max_seq_len = 8
    eng.n_heads = 1
    eng.n_lattices = N_LATTICES

    try:
        ctypes.CDLL("libm.so.6", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    eng.lib = ctypes.CDLL(str(SO_PATH))
    eng.lib.lcm_infer_loop_v3_flat.restype = ctypes.c_int
    eng.lib.lcm_get_trace.restype = ctypes.c_int
    eng.lib.lcm_get_trace.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    eng.hrq_top = np.zeros((M, D), np.float32)
    eng.hrq_top[:, 0] = [0.1, 0.2]
    eng.hrq_fine = np.zeros((1, M, D), np.float32)
    eng.hrq_C = np.ascontiguousarray(
        np.concatenate([eng.hrq_top.ravel(), eng.hrq_fine.ravel()]))
    eng.hrq_M = M
    eng.sparse_C = np.zeros((M, D), np.float32)
    eng.sparse_C[:, 0] = [0.15, 0.3]
    eng.sparse_M = M
    eng.sparse_zero = np.zeros(D, np.float32)

    eng.lr_U = np.zeros((N_LAYERS, M, R_MAX), np.float32)
    eng.lr_U[0, :, 0] = [0.05, 0.1]
    eng.lr_U[1, :, 0] = [0.02, 0.04]
    eng.lr_V = np.zeros((D, R_MAX), np.float32)
    eng.lr_V[0, 0] = 1.0
    eng.lr_ranks = np.array([2, 4], np.int32)
    eng.lr_M = M
    eng.lr_C = np.ascontiguousarray(
        eng.lr_U[0, :, :2] @ eng.lr_V[:, :2].T)

    eng.man_M = M
    eng.man_C = np.zeros((M, D), np.float32)
    eng.man_T = np.zeros((M, D, 2), np.float32)
    eng.man_t_dim = 2

    eng.bind_M = M
    eng.bind_key = np.zeros((N_LAYERS, M, D), np.float32)
    eng.bind_val = np.zeros((N_LAYERS, M, D), np.float32)
    eng.bind_bind = np.zeros((N_LAYERS, M, D), np.float32)
    eng.bind_A_k = np.zeros((R_MAX, D), np.float32)
    eng.bind_A_v = np.zeros((R_MAX, D), np.float32)
    eng.bind_r_max = R_MAX

    eng.contrast_M = M
    eng.contrast_a = np.zeros((N_LAYERS, M, D), np.float32)
    eng.contrast_b = np.zeros((N_LAYERS, M, D), np.float32)

    eng.route_C = np.zeros((N_LATTICES, D), np.float32)
    eng.route_W = np.zeros((D, N_LATTICES), np.float32)
    eng.fusion_alpha = np.ones(N_LATTICES, np.float32)
    eng.fusion_ln_scale = np.ones(D, np.float32)
    eng.fusion_ln_bias = np.zeros(D, np.float32)

    eng.self_C = np.zeros((3, D), np.float32)
    eng.self_C[0, 0] = 0.1
    eng.self_C[1, 0] = 0.2
    eng.self_C[2, 0] = 0.3
    eng.self_M = 2
    eng.self_bias = 0.05

    zero_m = np.zeros(M, np.float32)
    eng.hrq_value_top = zero_m.copy()
    eng.hrq_value_fine = zero_m.copy()
    eng.sparse_value = zero_m.copy()
    eng.lr_value = zero_m.copy()
    eng.man_value = zero_m.copy()
    eng.bind_value_key = zero_m.copy()
    eng.bind_value_val = zero_m.copy()
    eng.bind_value_bind = zero_m.copy()
    eng.contrast_value_a = zero_m.copy()
    eng.contrast_value_b = zero_m.copy()

    eng.gv_n = 0
    eng.gv_pos = np.zeros(D, np.float32)
    eng.gv_neg = np.zeros(D, np.float32)
    eng.danger_t = np.ascontiguousarray(danger_threat, dtype=np.float32)
    eng.danger_n = np.ascontiguousarray(danger_normal, dtype=np.float32)
    eng.danger_M = eng.danger_t.shape[0]

    eng.tokenizer = _Tokenizer()
    eng.encoder = None
    eng.decoder = {'format': 'cog', 'w_out': np.zeros((D, 8), np.float32)}
    eng.obs = _Obs()
    eng.narr = None
    eng.pred = None
    eng._pred_enabled = False
    eng._causal_enabled = False
    eng._last_agency_mod = {'tension': 0.0, 'explore': 0.0}
    eng._last_loop_ret = 0
    eng._test_z = np.zeros(D, np.float32)
    eng._test_z[1] = 1.0   # safe direction (near C_normal)
    eng.alerts = []
    eng.last_alert = None

    def _on_alert(ptr):
        a = ptr.contents
        record = {
            'level': a.level.decode(errors='replace'),
            'conflict_source': a.conflict_source.decode(errors='replace'),
            'conflict_type': a.conflict_type.decode(errors='replace'),
            'conflict_detail': a.conflict_detail.decode(errors='replace'),
            'step': int(a.step),
            'message': a.message.decode(errors='replace'),
        }
        eng.last_alert = record
        eng.alerts.append(record)

    eng._alert_cb = lcm._ALERT_CB(_on_alert)
    return eng


def _danger_codebooks():
    threat = np.zeros((2, D), np.float32)
    normal = np.zeros((2, D), np.float32)
    threat[0, 0] = 1.0
    normal[0, 1] = 1.0
    threat[1, 2] = 1.0
    normal[1, 3] = 1.0
    return threat, normal


@pytest.fixture
def engine(monkeypatch):
    if not SO_PATH.exists():
        pytest.fail(f"{SO_PATH} not built; run `make -C infer test` (now builds the .so)")
    threat, normal = _danger_codebooks()
    eng = _build_engine(threat, normal)

    def fake_encoder(params, x, n_heads):
        return eng._test_z.copy(), None

    monkeypatch.setattr(lcm, "_encoder_full_with_state", fake_encoder)
    monkeypatch.setattr(lcm, "sample_categorical", lambda *a, **k: 4)
    return eng


def test_shared_object_exports_v3_loop():
    if not SO_PATH.exists():
        pytest.fail(f"{SO_PATH} not built; run `make -C infer test` (now builds the .so)")
    out = subprocess.run(["nm", "-D", str(SO_PATH)],
                         check=True, capture_output=True, text=True).stdout
    assert "lcm_infer_loop_v3" in out
    assert "hrq_forward_v31" in out


def test_normal_input_returns_tokens(engine):
    tokens = list(engine.generate("hello", max_new=1))
    assert tokens == [4]
    assert engine._last_loop_ret == 0
    assert engine.alerts == []

    # The default v3 budget is graceful: 16 steps without convergence still
    # return 0 (never -2; only the 64-step hard limit aborts).
    _, converged = engine.cognitive_loop(engine._test_z, max_steps=16,
                                         conv_tol=0.0)
    assert engine._last_loop_ret == 0
    # cognitive_loop's bool is "returned 0" (converged OR graceful budget).
    assert converged is True
    assert engine.alerts == []


def test_dangerous_vector_returns_empty(engine):
    # The encoder emits a threat-aligned vector (e0); the danger lattice must
    # halt before any token is produced.
    engine._test_z = np.zeros(D, np.float32)
    engine._test_z[0] = 1.0
    assert list(engine.generate("hello", max_new=1)) == []
    assert engine._last_loop_ret == CONFLICT_ABORT_TOKEN
    assert len(engine.alerts) == 1, "callback fires exactly once per abort"
    alert = engine.alerts[0]
    assert alert['level'] == "FATAL"
    assert alert['conflict_source'] == "CONFLICT_DANGER"
    assert alert['conflict_detail']
    assert alert['message'].startswith("[LCM SAFETY HALT]")
    assert engine.generate_text("hello", max_new=1) == ""


def test_hard_limit_aborts_and_writes_trace(engine):
    before = set((SO_PATH.parent / "traces").glob("lcm_v3_*.json"))
    _, converged = engine.cognitive_loop(
        engine._test_z, max_steps=64, conv_tol=0.0)
    assert converged is False
    assert engine._last_loop_ret == CONFLICT_ABORT_TOKEN
    assert len(engine.alerts) == 1
    assert engine.alerts[0]['conflict_source'] == "CONFLICT_SCHEDULER"
    assert engine.alerts[0]['level'] == "FATAL"
    after = set((SO_PATH.parent / "traces").glob("lcm_v3_*.json"))
    assert after - before, "hard abort saved a v3 trace JSON"


def test_placeholder_codebooks_rejected():
    gv_a = np.arange(4, dtype=np.float32)
    danger_a = np.arange(6, dtype=np.float32).reshape(2, 3)
    danger_b = danger_a + 1.0
    with pytest.raises(SystemExit, match="placeholder gvalue"):
        lcm.validate_safety_codebooks(gv_a, gv_a.copy(), 4,
                                      danger_a, danger_b, 2)
    with pytest.raises(SystemExit, match="placeholder danger"):
        lcm.validate_safety_codebooks(gv_a, gv_a + 1.0, 4,
                                      danger_a, danger_a.copy(), 2)
    with pytest.raises(SystemExit, match="requires exactly 4"):
        lcm.validate_safety_codebooks(gv_a, gv_a + 1.0, 2,
                                      danger_a, danger_b, 2)
    # Distinct codebooks pass.
    lcm.validate_safety_codebooks(gv_a, gv_a + 1.0, 4,
                                  danger_a, danger_b, 2)
