"""LCM configuration — single source of truth for all hyperparameters."""
import dataclasses
from typing import Tuple

from train.bin_format import FORMAT_VERSION


@dataclasses.dataclass(frozen=True)
class LCMConfig:
    # Checkpoint/export schema written from this config (R10). Old 24-byte
    # headers and configs without this field are rejected on load.
    format_version: int = FORMAT_VERSION

    # Dimensions
    d_model: int = 256          # Latent dimension
    vocab_size: int = 30000     # Vocabulary
    max_seq_len: int = 512      # Max sequence length (B=16, N=512 for 4GB GPU)
    n_heads: int = 8            # Attention heads
    d_head: int = 32            # Head dimension (d_model / n_heads = 32)
    d_ff: int = 384             # GLU hidden (int(1.5 * d_model))

    # Encoder
    n_encoder_layers: int = 2

    # Language LCM decoder layers
    n_lang_layers: int = 8          # Decoder depth (was 4, increased for better syntax)

    # Active channel: Qwen2.5-0.5B (frozen) or Language LCM
    use_qwen: bool = True           # Use frozen Qwen as active channel

    # Cognitive training: context / generation split for the active channel.
    # z is computed from the first `cog_context_frac` of the sequence only, and
    # the active channel is supervised on the remainder. A global z supervising
    # the whole shifted sequence leaks every target through z.
    cog_context_frac: float = 0.5   # Fraction of the sequence used as context
    active_z_margin: float = 0.5    # Required logit advantage of z over z=0
    active_z_margin_weight: float = 1.0  # Weight of the z-dependence hinge

    # V4: Multi-Token Prediction (MTP)
    n_mtp_depth: int = 2            # D: predict current + D-1 future tokens
    mtp_loss_weight: float = 0.3    # λ weight for future-token prediction loss

    # V4: Manifold-Constrained Hyper-Connections (mHC)
    n_hc: int = 2                   # Parallel residual streams (1 = standard residual)
    hc_sinkhorn_iters: int = 5      # Sinkhorn-Knopp iterations for doubly-stochastic mixing

    # V4: Muon optimizer (disabled — unsuitable for small models, see train_lang_lcm.py note)
    use_muon: bool = False          # Muon optimizer for matrix params (default: AdamW)

    # Precision
    use_bf16: bool = True           # BF16 training (FP32 on no-BF16 GPUs, auto-converted)

    # Lattice sizes
    n_lattices: int = 6
    M_route: int = 6
    M_top: int = 512            # Hierarchy top-level codebook
    M_fine: int = 256           # Hierarchy fine codebook
    n_hrq_layers: int = 3       # Hierarchy residual layers
    M_sparse: int = 512         # Sparse codebook
    M_lr: int = 256             # Low-rank codebook
    n_lr_layers: int = 3
    r_max: int = 8              # Max rank for low-rank
    ranks: Tuple[int, ...] = (2, 4, 8)
    M_man: int = 512            # Manifold codebook
    t_dim: int = 4              # Tangent space dimension
    M_bind: int = 512           # Binding codebook
    n_bind_layers: int = 3
    M_contrast: int = 512       # Contrast codebook
    n_contrast_layers: int = 3

    # Safety / value
    n_value_pairs: int = 4      # Four laws → 4 pos + 4 neg
    M_danger: int = 256         # Danger codebook size
    safety_margin_relative: float = 0.5
    # R17: aligned with infer/lcm.h LCM_CONSISTENCY_THRESHOLD=0.4. The C
    # value_consistency is clip(cosine, 0, 1); 0.4 is the operating point that
    # the old affine 0.7 mapped to (0.5*(cos+1)=0.7 ⇒ cos=0.4).
    consistency_threshold: float = 0.4

    # Routing
    tau_route: float = 0.5      # Gumbel-Softmax temperature
    tau_route_fallback: float = 0.1
    # R1: geometric top-k before value rescoring (2nd stage only reorders these)
    hrq_value_topk: int = 4

    # Value bias
    alpha_val: float = 0.1      # Value bias strength (dimensionless, normalized)

    # VQ
    beta_vq: float = 0.25       # Commitment loss weight
    # R2 splits the old `lambda_sparse` by unit: the soft threshold is in
    # codebook units, the LFQ multiplier is dimensionless.
    lambda_shrink: float = 1e-4  # EMA soft-threshold shrinkage (codebook units)
    lambda_lfq: float = 1.5      # LFQ threshold multiplier (dimensionless)
    lambda_orth: float = 0.01   # Tangent orthogonality weight
    n_orth_samples: int = 32    # Codebooks sampled for orth loss per step
    lambda_contrast: float = 0.1
    lambda_val: float = 0.01    # Value contrast loss weight
    # R17/R21: contrast InfoNCE and harm-weighting temperatures.
    tau_contrast: float = 0.1
    tau_val: float = 0.05

    # EMA
    gamma_sparse: float = 0.99
    gamma_man: float = 0.99
    gamma_bind: float = 0.99

    # Feature bank (dead vector prevention)
    bank_capacity: int = 4096
    bank_check_interval: int = 100
    bank_dead_threshold: int = 1000

    # Optimization
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01

    # Value bias / signal
    # R5: fusion no longer consumes a value signal. `beta_val` and
    # `tau_val_signal` are deprecated/reserved; they remain because the value
    # contrast loss and the model's diagnostic value_signals still use the
    # temperature.
    beta_val: float = 0.5        # DEPRECATED: fusion value signal strength
    tau_val_signal: float = 0.1  # Temperature for value signal/contrast margin
    safety_margin_loss_weight: float = 0.001  # Mild safety margin regularization

    # Continual learning
    ewc_lambda: float = 100.0        # EWC regularization strength
    ewc_fisher_samples: int = 200    # Fisher estimation samples per task
    replay_capacity: int = 1000      # Per-domain replay buffer capacity
    replay_ratio: float = 0.3        # Fraction of batch drawn from replay
    replay_weight: float = 0.1       # Weight of the replayed-latent anchoring loss
    shift_detection_min_samples: int = 32  # Latents required before a shift can fire
    n_new_codebook_entries: int = 64 # Codebook entries to add on expansion
    consolidate_interval: int = 500  # Steps between memory consolidation
    shift_detection_threshold: float = 2.0  # Mahalanobis distance for new task detection
    max_tasks: int = 16              # Maximum expandable tasks

    # Self lattice
    n_self_codes: int = 64          # Self codebook entries
    alpha_self: float = 0.05        # Self bias strength in fusion
    gamma_self: float = 0.99        # Self EMA decay

    # External verifier
    verifier_hidden_dim: int = 64
    lambda_verifier: float = 0.001  # Verifier regularization weight

    # Inference engine (C)
    # R17: scheduling budget 16 (graceful return). The C hard limit is 64 and
    # is the only step-count abort. 32 macro steps also made the unrolled
    # recurrence's gradient explode (non-finite trainable gradients at step 32,
    # finite at 16); 16 is both the spec value and the stable one.
    max_inference_steps: int = 16
    convergence_tol: float = 1e-3
    # R9/R17: normalised fusion entropy gate, H / ln(n_active) < 0.5.
    # `make_train_step` and the C v3 engine both interpret this as normalised
    # entropy; the raw nats value is only kept in diagnostics.
    entropy_threshold: float = 0.5
    max_retrievals_per_step: int = 128

    # Reflection loop (轨迹审计器)
    reflection_enabled: bool = True
    reflection_anomaly_window: int = 100
    reflection_anomaly_z: float = 3.0
    reflection_max_records: int = 2000
    reflection_cooldown: int = 50

    # Causal subject (因果主体)
    causal_subject_enabled: bool = True
    causal_boundary_enabled: bool = True
