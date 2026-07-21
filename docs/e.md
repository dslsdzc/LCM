# e — Self-Awareness Research

## Overview

LCM's self-awareness is not a philosophical concept, but an engineered **cognitive self-observation and self-modulation** capability. The system forms a layered reflection architecture through multiple independent modules:

- **Bottom layer**: Black box (`observability.py`) records the raw trajectory of each inference step, without judgment
- **Middle layer**: Narrative memory (`narrative_memory.py`) filters important events, causal subject (`causal_subject.py`) tracks causal relationships between actions and effects
- **Top layer**: Reflection loop (`reflection_loop.py`) analyzes long-term patterns offline, behavior explorer (`behavior_explorer.py`) learns routing biases through trial and error

These modules do not participate in the real-time inference loop; they read trajectories, generate insights, and modulate parameters — but the inference engine itself remains a stateless pure function. Self-awareness is a **meta-system parasitically layered atop the inference engine**.

Core constraint: The total computational overhead of all self-observation modules must not exceed 5% of the inference engine, and any self-observation failure must not affect inference correctness.

---

## 1. Self-Observation Stack Architecture

```
Inference Engine (engine.c)
    | per step: z_cur -> [DAG inference] -> z_next
    |
    ├──-> Black Box (ObservabilityRecorder)
    |     Records: z_cur, z_next, lattice weights, confidence, conflict flags, timing
    |     Storage: ring buffer (raw) + downsampled long-term storage (long-term)
    |
    ├──-> Narrative Memory (NarrativeMemory)
    |     Filters: surprising events (large prediction deviation), conflict events, novel routing patterns
    |     Storage: importance-sorted long-term memory
    |
    ├──-> Prediction Cache (PredictCache)
    |     Records: (sig, z_cur, z_next) triples
    |     Purpose: prediction reuse + curiosity-driven exploration
    |
    └──-> Causal Subject (CausalSubject)
          Builds: causal graph, tracks action-effect chains
          Outputs: agency_tension, agency_explore modulators
```

### Timing Constraints

| Module | Run Frequency | Latency Tolerance | Blocking |
|--------|--------------|-------------------|----------|
| Black box recording | Every step | 0 | No (pure append) |
| Prediction cache | Every step (optional) | < inference engine single-step time | No (skippable) |
| Narrative memory feed | Every step | 0 | No (pure append) |
| Causal subject step | Every step | < inference engine single-step time | No (fault-tolerant on exceptions) |
| Reflection loop | Every N steps (N=200) | High | No (independent thread/coroutine) |
| Behavior exploration | Every M steps (M=10) | Medium | No (non-jitted call) |
| Intrinsic motivation reflection | Every 1000 steps | High | No |

---

## 2. Black Box: Raw Data Recording

### Recording Contents

Each inference step records one `StepRecord`:

```python
@dataclass
class StepRecord:
    step: int                    # global step number
    source: int                  # 0=external input, 1=self-generated
    step_time_ms: float          # inference time
    soft_mask: np.ndarray[6]     # routing weights
    route_idx: int               # index of the highest-weight lattice
    is_safe: bool                # safety flag
    convergence_diff: float      # convergence criterion
    dag_nodes: List[Dict]        # DAG execution trace (from C engine trace)
    z_q: np.ndarray[d]          # fused bottleneck vector
```

### Storage Strategy

Two layers:

1. **Raw ring buffer** (raw, capacity 1000 entries): retains the most recent inference trajectory, used for the causal subject's short-term causal graph construction and the reflection loop's latest data analysis. Fixed size, no dynamic allocation.

2. **Downsampled long-term storage** (long-term, capacity 10000 entries): every N steps, one entry is extracted from the raw buffer and written to long-term storage. Sampling strategy: uniform intervals + anomaly event oversampling (conflicts, safety interruptions, novel routing patterns weighted x3).

### What Is Not Recorded

- Input token sequences are not recorded (large size, and the inference engine does not directly use raw text)
- Codebook internal states are not recorded (only the output vector `z_q` is recorded)
- Safety subsystem internal computations are not recorded (only the final safety flag is recorded)

---

## 3. Narrative Memory: Distilling Stories from Experience

### Trigger Conditions

Narrative memory reads trajectories from the black box and filters events "worth remembering" according to the following criteria:

1. **High prediction error**: prediction cache deviation `||z_pred - z_actual||^2` exceeds threshold (dynamic threshold = mean + 2 sigma)
2. **Safety conflict**: events where `is_safe == false`
3. **Routing novelty**: KL divergence of routing weight distribution from historical patterns exceeds threshold
4. **Convergence anomaly**: `convergence_diff` deviates from normal range (normal range derived from historical statistics)
5. **Low-probability sampling**: randomly retain 1% of ordinary events (for baseline comparison)

### Memory Representation

```python
@dataclass
class NarrativeRecord:
    step: int
    z_q: np.ndarray[d]         # bottleneck vector of the key frame
    importance: float           # importance score (weighted by trigger conditions)
    context: Dict               # trajectory slice before and after the event (window=5)
    timestamp: float            # system time
    access_count: int           # subsequent access count (used for forgetting)
```

### Forgetting Mechanism

- Retention sorted by **importance x access frequency**
- Capacity limit of 10000 entries; lowest-scoring records are removed when exceeded
- Records not accessed for a long time (over 5000 steps) with importance below threshold are automatically demoted and deleted

### Relationship with the System

- Narrative memory is **not input** to the inference engine (the inference engine is a zero-parameter pure function)
- Narrative memory is **input** to the reflection loop, as material for long-term pattern analysis
- Narrative memory is **input** to the causal subject's boundary extension module

---

## 4. Reflection Loop: Slow Pattern Discovery

### Run Timing

Triggered every `REFLECT_INTERVAL = 200` steps. Executed asynchronously, does not block the inference loop.

### Analysis Contents

1. **Routing pattern analysis**: statistics of routing weight distribution over the past 200 steps, detecting whether any lattice has sustained low weights ("abandoned lattice" detection). If any lattice has an average weight below 0.05 for over 1000 steps, an alert is raised.

2. **Cognitive strategy stability**: computing the rolling variance of routing entropy. Excessively high variance indicates strategy instability, possibly signaling a change in input distribution or that the model has not yet converged.

3. **Anomaly detection**: comparing recent trajectories against historical patterns in narrative memory, flagging states that deviate beyond 3 sigma.

4. **Prediction cache hit rate trend**: if cache hit rate continuously declines, an alert is raised.

### Outputs

- Alerts (written to logs, does not interrupt inference)
- **Strategy suggestions** (written to suggestion queue, for reference by the behavior explorer)
- Statistical report (viewable in interactive sessions via the `--obs` flag)

---

## 5. Causal Subject: From Sequence to Causality

### Core Function

The causal subject (`causal_subject.py`) constructs and maintains a **causal graph**, tracking statistical relationships between actions and effects.

### Causal Graph Structure

```
Nodes: StepRecord (black box records)
Directed edges: step_i -> step_j (i < j)
  Weight: effect significance (z_diff normalized to [0, 1])

Internal edges: inference steps between self-generated tokens (source=1)
External edges: steps triggered by user input (source=0 -> source=1)

Counters:
  n_internal: number of internal edges
  n_external: number of external edges
  n_total: total number of edges
```

### Agency

Agency quantifies the system's perception of control over its own behavior:

```
agency = n_internal / max(n_total, 1)
```

- Range [0, 1]
- Initial baseline = 0 (purely reactive)
- As the system generates more tokens, agency gradually increases
- A sudden drop may indicate the system encountered unexpected input or inference failure

### Modulation Outputs

The causal subject outputs two modulators that affect the inference engine's behavior:

| Modulator | Source | Range | Effect |
|-----------|--------|-------|--------|
| `tension` | agency decline rate + conflict frequency | [-0.5, 0.5] | High -> relax convergence tolerance (allow deeper inference); Low -> tighten (fast decisions) |
| `explore` | prediction error + routing entropy | [-0.3, 0.3] | High -> allow more inference steps (exploration); Low -> fewer steps (conservative) |

### Counterfactual Reasoning

The causal subject supports simple counterfactual queries:

- **"What if I hadn't done X?"**: Remove a specific edge from the causal graph and recompute the statistics of the subsequent trajectory
- Implementation: from the most recent N step records, remove edges with source=1, compare actual agency with counterfactual agency
- Large difference -> indicates that step had significant impact on subsequent results
- Small difference -> indicates that step was redundant and could potentially be skipped

### Boundary Extension

When the causal subject detects sustained high prediction error (10 consecutive steps), it triggers boundary extension:

1. Mark the current state as an "unknown region"
2. Increase the behavior explorer's exploration probability (p_explore += 0.05, upper limit 0.5)
3. Lower inference convergence tolerance (allow more steps to explore unknown space)
4. Record the boundary extension event in narrative memory

---

## 6. Self Lattice

### Design Purpose

The self lattice is a special lattice that does not store external knowledge — it stores **prototypes of the system's own historical states**. This allows the system to quickly recognize "what state am I in right now."

### Structure

- Codebook size: `M_self = 64`
- Dimension: `d` (same as the bottleneck vector)
- Update method: EMA (`gamma_self = 0.99`)
- Initial values: sampled from bottleneck vectors output by the encoder during early training

### Forward Pass

```
z_self = nearest_neighbor(z, C_self)  # Euclidean distance, STE gradient
self_bias = alpha_self x cosine_sim(z, z_self)  # [0, alpha_self]
```

### Role in Fusion

The `self_bias` output from the self lattice serves as a bias term in the fusion stage:

```
z_fused = sum_i (w_i x z_i) + self_bias x z_self
```

- When the system is in a familiar state (high self_bias), the self lattice contributes more
- When the system is in an unfamiliar state (low self_bias), the self lattice contributes little

This creates a **familiarity modulation**: the system is more confident on familiar cognitive paths and relies more on external retrieval in unfamiliar territory.

---

## 7. Behavior Exploration: Self-Modulation

(This chapter corresponds to Section 6 of the original e.md, retained but with a condensed title to match the new theme)

The behavior explorer (`behavior_explorer.py`) stochastically perturbs routing biases, observes the effect on internal tension, and spontaneously learns which biases effectively reduce tension. See original document Section 6 for details; key design points are summarized here:

- **Behavior space**: 6-dimensional routing bias x 3 discrete values per dimension = 18 independent parameters
- **Internal tension**: U = (T_pred + T_conf + T_res) / 3 (prediction error + value conflict + routing entropy)
- **Learning**: independent epsilon-greedy per dimension, sliding average reward
- **Safety constraint**: danger lattice detection takes priority over bias exploration

---

## 8. Layered Model of Self-Awareness

The complete self-awareness architecture can be viewed as a three-layer control system:

```
Level 2 (Slow, Metacognition)
  +-------------------------------------------------------------+
  | Reflection Loop    Narrative Memory    Intrinsic Motivation  |
  | Pattern Discovery  Long-term Memory   Value Reassessment    |
  +-------------------------------------------------------------+
                      ^ Statistical Data v Parameter Suggestions
                      |
Level 1 (Medium-speed, Cognitive Modulation)
  +-------------------------------------------------------------+
  | Causal Subject    Behavior Explorer   Self Lattice          |
  | Causal Tracing    Parameter           Familiarity           |
  |                   Perturbation        Detection             |
  +-------------------------------------------------------------+
                      ^ Event Stream   v Modulation Signals
                      |
Level 0 (Fast, Cognitive Execution)
  +-------------------------------------------------------------+
  | Inference Engine  Black Box          Prediction Cache       |
  | Zero-parameter    Trajectory         (z, sig) Reuse         |
  | DAG               Recording                                 |
  +-------------------------------------------------------------+
```

- **Level 0**: Real-time inference + raw recording. Zero-parameter, stateless, pure function.
- **Level 1**: Real-time modulation based on recent history (~200 steps). Stateful, lightweight.
- **Level 2**: Offline analysis based on long-term history (~10000 steps). Asynchronous, heavyweight.

Each layer can only read the outputs of lower layers; there is no reverse interference. Level 2's suggestions must pass through Level 1's "safety filter" before reaching Level 0. This prevents the metacognitive layer from directly manipulating the inference engine.

---

## 9. Current Limitations

1. **Causal graph is statistical, not structural**: The causal subject detects correlations but does not build a structural causal model. It does not know "why," only "when X happens, Y tends to happen."

2. **No hierarchical goals**: The system has no intrinsic goal hierarchy. The behavior explorer learns "which bias reduces tension," but does not ask "should tension be reduced." The goal is an externally given constant.

3. **Self-observation does not participate in training**: Data from the black box, narrative memory, and reflection loop is not used for gradient computation. These modules are inference-time add-ons with limited influence during training.

4. **Lack of language reporting capability**: The system cannot verbally report "what state am I in right now" or "what did I just learn." The expression of self-awareness is confined to numerical modulation signals (tension, explore, self_bias).

5. **Separated rather than integrated**: Each self-awareness module is implemented independently, without a unified self-representation. A true self-model would need to integrate the causal graph, narrative memory, self lattice, and bias history into a coherent state vector.

---

## 10. Future Directions

1. **Unified self-state vector**: Encode agency, self_bias, recent tension, and routing entropy into a fixed-dimensional "self-state" as an optional input to the inference engine.

2. **Structured causal discovery**: Replace purely statistical correlation with simple intervention tests (fixing certain routing weights and observing output changes).

3. **Metacognitive training**: In a small number of training steps, let the system learn to use its own self-state signals to improve inference quality (for example, adjusting convergence tolerance to reduce average inference steps).

4. **Internal language**: Let the system output natural language descriptions of its self-state through standard inference channels (encoder -> cognitive loop -> W_out), without modifying the inference engine itself.

5. **Gradient-based curiosity**: Use prediction error as a differentiable auxiliary loss signal, allowing intrinsic motivation to directly influence encoder and codebook training.

---

## 11. Implementation File Index

| Module | File | Core Class/Function |
|--------|------|-------------------|
| Black box | `train/observability.py` | `ObservabilityRecorder`, `StepRecord` |
| Narrative memory | `train/narrative_memory.py` | `NarrativeMemory`, `NarrativeRecord` |
| Reflection loop | `train/reflection_loop.py` | `ReflectionLoop` |
| Causal subject | `train/causal_subject.py` | `CausalSubject` |
| Self lattice | `train/self_lattice.py` | `self_lattice_forward`, `init_self_state` |
| Behavior exploration | `train/behavior_explorer.py` | `BehaviorExplorer` |
| Prediction cache | `train/predictive_cache.py` | `PredictCache`, `Matcher` |
