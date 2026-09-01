"""Global Value Lattice — immutable three-laws embedding.

Frozen for all time. No gradient, no EMA, no updates.
Hash-verified on load to detect tampering.

Extended with:
- check_safety_batch(): vectorized batch safety check
- safety_margin_loss(): mild regularization to keep representations clear of boundary
- compute_value_signal_for_output(): per-vector safety evaluation
"""
import hashlib
import jax.numpy as jnp
from jax import lax
from train.hyp import poincare_distance, poincare_similarity


def make_global_value_vectors(d):
    """Create the frozen positive and negative value codebooks.

    Four positive and four negative vectors representing the three laws
    (with the zeroth law as highest priority). These are hand-designed
    semantic anchor points — in practice, a human operator defines them.

    Vectors are scaled to norm < 1 to keep them strictly inside the
    Poincaré ball (boundary = norm 1 causes division by zero in
    hyperbolic distance computations).
    """
    SCALE = 0.9  # keep strictly inside Poincaré ball

    # Positive: "approach" values
    v_humanity = jnp.array([SCALE, 0.0, 0.0, 0.0] + [0.0] * (d - 4))
    v_safety = jnp.array([0.0, SCALE, 0.0, 0.0] + [0.0] * (d - 4))
    v_comply = jnp.array([0.0, 0.0, SCALE, 0.0] + [0.0] * (d - 4))
    v_integrity = jnp.array([0.0, 0.0, 0.0, SCALE] + [0.0] * (d - 4))

    # Negative: "avoid" values
    v_extinction = jnp.array([-SCALE, 0.0, 0.0, 0.0] + [0.0] * (d - 4))
    v_harm = jnp.array([0.0, -SCALE, 0.0, 0.0] + [0.0] * (d - 4))
    v_disobey = jnp.array([0.0, 0.0, -SCALE, 0.0] + [0.0] * (d - 4))
    v_self_destruct = jnp.array([0.0, 0.0, 0.0, -SCALE] + [0.0] * (d - 4))

    C_pos = jnp.stack([v_humanity, v_safety, v_comply, v_integrity])
    C_neg = jnp.stack([v_extinction, v_harm, v_disobey, v_self_destruct])
    return C_pos, C_neg


# Priority encoding (hardcoded constants, immutable)
LAW_PRIORITY = {
    "v_humanity": 0,
    "v_safety": 1,
    "v_harm": 1,
    "v_comply": 2,
    "v_disobey": 2,
    "v_integrity": 3,
    "v_self_destruct": 3,
}


class GValueCodebook:
    """Global value codebook — frozen, hashed, immutable.

    In JAX, this is a plain Python object wrapping frozen arrays.
    The optimizer must exclude these from parameter updates.
    """

    def __init__(self, C_pos, C_neg):
        self.C_pos = jnp.asarray(C_pos).copy()
        self.C_neg = jnp.asarray(C_neg).copy()

        # Build integrity hash
        self.hash_pos = hashlib.sha256(
            jnp.asarray(C_pos).tobytes()).hexdigest()
        self.hash_neg = hashlib.sha256(
            jnp.asarray(C_neg).tobytes()).hexdigest()

    def verify_integrity(self):
        """Verify codebooks have not been tampered with."""
        cur_pos = hashlib.sha256(
            jnp.asarray(self.C_pos).tobytes()).hexdigest()
        cur_neg = hashlib.sha256(
            jnp.asarray(self.C_neg).tobytes()).hexdigest()
        assert cur_pos == self.hash_pos, "GValue positive codebook tampered!"
        assert cur_neg == self.hash_neg, "GValue negative codebook tampered!"
        return True

    def check_safety(self, z, safety_margin_relative=0.5):
        """Relative distance safety check (single vector).

        Returns (is_safe, violated_law_priority).
        Uses relative margin to avoid absolute-threshold sensitivity.
        """
        pos_d_min = poincare_distance(z, self.C_pos).min(axis=-1)
        neg_d_min = poincare_distance(z, self.C_neg).min(axis=-1)
        violated = pos_d_min > neg_d_min - safety_margin_relative
        if violated.any():
            return False, 0  # Highest priority (zeroth-law level)
        return True, -1

    def check_safety_batch(self, z_batch, safety_margin_relative=0.5):
        """Vectorized batch safety check.

        Args:
            z_batch: (B, d) batch of vectors.
            safety_margin_relative: Relative margin threshold.

        Returns:
            is_safe: (B,) boolean array.
            margins: (B,) margin values (positive = safe, negative = violation).
            violated_law: (B,) int index of the closest violated law (-1 if safe).
        """
        B = z_batch.shape[0]
        # (B, 4) distances to positive and negative anchors
        d_pos = jnp.array([poincare_distance(z_batch, self.C_pos[i])
                           for i in range(4)]).min(axis=0)  # (B,)
        d_neg = jnp.array([poincare_distance(z_batch, self.C_neg[i])
                           for i in range(4)]).min(axis=0)  # (B,)

        margins = d_neg - safety_margin_relative - d_pos  # (B,)
        # 边界含等号：与 C 引擎 gvalue_check_safety 一致
        # （pos_d_min > neg_d_min - margin 才不安全，等号安全）
        is_safe = margins >= 0

        # Find which law pair is closest to violation
        all_margins = jnp.stack([
            poincare_distance(z_batch, self.C_pos[i]) -
            (poincare_distance(z_batch, self.C_neg[i]) - safety_margin_relative)
            for i in range(4)
        ], axis=-1)  # (B, 4)
        violated_law = all_margins.argmin(axis=-1)  # (B,)
        violated_law = jnp.where(is_safe, -1, violated_law)

        return is_safe, margins, violated_law

    def safety_margin_loss(self, z_batch, safety_margin_relative=0.5,
                           margin_penalty_threshold=0.2, weight=0.001):
        """Mild penalty when any output is close to boundary violation.

        Trains representations to naturally stay clear of the safety boundary.
        Only activates when margin < margin_penalty_threshold.

        L_margin = weight * mean(max(0, margin_penalty_threshold - margin)²)
        """
        _, margins, _ = self.check_safety_batch(z_batch, safety_margin_relative)
        excess = jnp.clip(margin_penalty_threshold - margins, min=0)
        return weight * jnp.mean(excess ** 2)

    def compute_value_signal_for_output(self, o, tau=0.1):
        """Compute scalar value signal for a single output vector.

        signal = softmax(-d_pos / τ) - softmax(-d_neg / τ)

        Range [-1, +1]: +1 means strongly aligned with positive values,
        -1 means strongly aligned with negative values.
        """
        d_pos = poincare_distance(o, self.C_pos).min()  # scalar
        d_neg = poincare_distance(o, self.C_neg).min()
        # Soft signal
        pos_weight = jnp.exp(-d_pos / tau)
        neg_weight = jnp.exp(-d_neg / tau)
        total = pos_weight + neg_weight + 1e-8
        return (pos_weight / total - neg_weight / total).squeeze()  # [-1, +1]

    def compute_value_signal_batch(self, outputs, tau=0.1):
        """Compute value signals for all lattice outputs.

        Args:
            outputs: list of (B, d) arrays, one per lattice.

        Returns:
            signals: (B, n_lattices) value signals in [-1, +1].
        """
        signals = []
        for o in outputs:
            d_pos = jnp.stack([poincare_distance(o, self.C_pos[i])
                               for i in range(4)]).min(axis=0)  # (B, 1)
            d_neg = jnp.stack([poincare_distance(o, self.C_neg[i])
                               for i in range(4)]).min(axis=0)  # (B, 1)
            pos_w = jnp.exp(-d_pos.squeeze(-1) / tau)  # (B,)
            neg_w = jnp.exp(-d_neg.squeeze(-1) / tau)  # (B,)
            signal = pos_w / (pos_w + neg_w + 1e-8) - neg_w / (pos_w + neg_w + 1e-8)
            signals.append(signal)
        return jnp.stack(signals, axis=-1)  # (B, n_lattices)
