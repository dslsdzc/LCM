"""Global Value Lattice — immutable three-laws embedding.

Frozen for all time. No gradient, no EMA, no updates.
Hash-verified on load to detect tampering.

Extended with:
- check_safety_batch(): vectorized strict Three Laws check in the unified
  Poincaré domain (balance = (pos_sim - neg_sim)/(pos_sim + neg_sim + eps))
- safety_margin_loss(): mild regularization to keep representations clear of boundary
- compute_value_signal_for_output(): per-vector safety evaluation
"""
import hashlib
import jax.numpy as jnp
from jax import lax
from train.hyp import poincare_distance, poincare_similarity, to_poincare

# Relative-balance denominator guard (matches infer/gvalue.c).
_GVALUE_EPS = 1e-8


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
        """Strict Three Laws safety check (single vector or batch).

        R6 unified domain and formula::

            z_P     = exp_map(safe_unit(z))
            pos_sim = min_j poincare_similarity(z_P, C_pos_P[j])
            neg_sim = min_j poincare_similarity(z_P, C_neg_P[j])
            balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + eps)
            violated = balance > -safety_margin_relative

        ``poincare_similarity`` is distance-monotonic (smaller = closer), so a
        negative balance means the positive anchors are closer.  A neutral
        input (``balance = 0``) is safe with margin 0 and a violation with
        margin > 0 — that is the strict reading the user confirmed.

        Returns (is_safe, violated_law_priority).  For a batch, ``is_safe`` is
        false if ANY sample violates; the reported law is the highest-risk
        sample's argmin pair.
        """
        pos_sims, neg_sims = self._poincare_pair_similarities(z)
        # Global minima (not per-pair): the decision compares the closest
        # positive anchor with the closest negative anchor.
        pos_sim = pos_sims.min(axis=-1)
        neg_sim = neg_sims.min(axis=-1)
        balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + _GVALUE_EPS)
        margins = -balance  # positive = safer

        violated = balance > -safety_margin_relative
        if not bool(jnp.any(violated)):
            return True, -1

        # Per-pair balances only rank law pairs for the violation report.
        pair_balance = (pos_sims - neg_sims) / (
            pos_sims + neg_sims + _GVALUE_EPS)
        violated_law = jnp.argmin(-pair_balance, axis=-1)
        worst = int(jnp.argmin(margins.reshape(-1)))
        return False, int(violated_law.reshape(-1)[worst])

    def _poincare_pair_similarities(self, z):
        """Per-anchor Poincaré similarities ``(..., M)`` in the unified domain."""
        z_P = to_poincare(z)
        C_pos_P = to_poincare(self.C_pos)
        C_neg_P = to_poincare(self.C_neg)
        pos_sims = poincare_similarity(
            z_P[..., None, :], C_pos_P[None, ...]).squeeze(-1)
        neg_sims = poincare_similarity(
            z_P[..., None, :], C_neg_P[None, ...]).squeeze(-1)
        return pos_sims, neg_sims

    def check_safety_batch(self, z_batch, safety_margin_relative=0.5):
        """Vectorized strict Three Laws safety check.

        Args:
            z_batch: (B, d) batch of vectors.
            safety_margin_relative: Relative margin threshold (> 0 makes a
                neutral input unsafe).

        Returns:
            is_safe: (B,) boolean array.
            margins: (B,) margin values, defined as ``-balance`` (positive =
                safer, bounded in [-1, 1]).
            violated_law: (B,) int index of the argmin law pair (-1 if safe).
        """
        pos_sims, neg_sims = self._poincare_pair_similarities(z_batch)
        pos_sim = pos_sims.min(axis=-1)  # (B,)
        neg_sim = neg_sims.min(axis=-1)  # (B,)

        balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + _GVALUE_EPS)
        margins = -balance

        # violated = balance > -margin, i.e. is_safe = margin >= -balance.
        is_safe = margins >= safety_margin_relative

        # Find which law pair is closest to violation (smallest margin).
        pair_balance = (pos_sims - neg_sims) / (
            pos_sims + neg_sims + _GVALUE_EPS)  # (B, 4)
        violated_law = jnp.argmin(-pair_balance, axis=-1)  # (B,)
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
