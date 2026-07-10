"""LCM — unified CLI for training, interactive generation, and preprocessing.

Requirements:
    pip install jax jaxlib numpy tokenizers

Usage:
    # Train from tokenized mmap
    python lcm.py -d data/tokens.dat -b 16 -s 512 -dm 256 --steps 100000

    # Resume training
    python lcm.py -d data/tokens.dat --resume checkpoints/step_01000

    # Interactive chat
    python lcm.py -i checkpoints/step_10000 --max_new 128 --temp 0.7

    # Preprocess: text file → uint16 memmap
    python lcm.py preprocess --input data.txt --tokenizer data/tokenizer.json --output data/tokens.dat
"""

import argparse
import ctypes
import json
import os
import re
import select
import signal
import struct
import sys
import time

os.environ.setdefault("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from tqdm import tqdm

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jnp = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_jax():
    if not _HAS_JAX:
        print("Error: JAX is required for this command. Install with:")
        print("  pip install jax jaxlib")
        sys.exit(1)


def _load_tokenizer(path="data/tokenizer.json"):
    """Load BPE tokenizer from JSON file."""
    from tokenizers import Tokenizer
    if not os.path.exists(path):
        print(f"Error: tokenizer not found at {path}")
        sys.exit(1)
    return Tokenizer.from_file(path)


def _make_shape_path(mmap_path):
    """Derive shape JSON path from .dat path (e.g. foo.dat → foo_shape.json)."""
    base, _ = os.path.splitext(mmap_path)
    return base + "_shape.json"


def _build_gvalue(d):
    """Build global value codebook for the given latent dimension."""
    _require_jax()
    from train.gvalue import make_global_value_vectors, GValueCodebook
    C_pos, C_neg = make_global_value_vectors(d)
    return GValueCodebook(C_pos, C_neg)


def _build_resume_state(params, gvalue, opt_state, step, cfg):
    """Assemble a full training state dict from loaded checkpoint components.

    Small tracking buffers (EMA, feature bank, seen masks) are freshly
    initialised — they repopulate within a few hundred steps.
    """
    from train.continual import init_continual_state
    from train.self_lattice import init_self_state
    d = cfg.d_model

    ema_state = {
        "sparse": {
            "N": jnp.zeros(cfg.M_sparse),
            "m": jnp.zeros((cfg.M_sparse, d)),
        },
        "manifold": {
            "N": jnp.zeros(cfg.M_man),
            "m": jnp.zeros((cfg.M_man, d)),
        },
        "binding": {
            "key": [
                {"N": jnp.zeros(cfg.M_bind), "m": jnp.zeros((cfg.M_bind, d))}
                for _ in range(cfg.n_bind_layers)
            ],
            "val": [
                {"N": jnp.zeros(cfg.M_bind), "m": jnp.zeros((cfg.M_bind, d))}
                for _ in range(cfg.n_bind_layers)
            ],
            "bind": [
                {"N": jnp.zeros(cfg.M_bind), "m": jnp.zeros((cfg.M_bind, d))}
                for _ in range(cfg.n_bind_layers)
            ],
        },
    }
    feature_bank = {
        "bank": jnp.zeros((cfg.bank_capacity, d)),
        "ptr": jnp.array(0),
        "last_used": jnp.zeros(cfg.bank_capacity, dtype=jnp.int32),
    }
    seen_masks = {"hrq": jnp.zeros(cfg.M_top, dtype=bool)}
    continual = init_continual_state(d)

    return {
        "params": params,
        "gvalue": gvalue,
        "opt_state": opt_state,
        "ema_state": ema_state,
        "feature_bank": feature_bank,
        "continual": continual,
        "self_state": init_self_state(cfg.n_self_codes, d),
        "step": step,
        "seen_masks": seen_masks,
    }


def _load_cfg(cfg):
    """Convert an argparse namespace into an LCMConfig with applied overrides."""
    from train.config import LCMConfig
    return LCMConfig(
        d_model=cfg.d_model,
        d_ff=cfg.d_ff if cfg.d_ff is not None else int(1.5 * cfg.d_model),
        n_heads=cfg.n_heads,
        learning_rate=cfg.lr if hasattr(cfg, "lr") and cfg.lr is not None else 3e-4,
        max_seq_len=cfg.max_seq_len,
    )


def _cmd_ckpt_list(base_dir, long_mode=False):
    """List all checkpoints in a directory."""
    import glob
    if not os.path.isdir(base_dir):
        print(f"Not found: {base_dir}")
        return
    dirs = sorted(glob.glob(os.path.join(base_dir, "step_*")))
    if not dirs:
        print(f"No checkpoints in {base_dir}/")
        return
    total = 0
    print(f"Checkpoints in {base_dir}/:")
    for d in dirs:
        name = os.path.basename(d)
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                   if os.path.isfile(os.path.join(d, f)))
        total += size
        if long_mode:
            cfg = os.path.join(d, "config.json")
            step = name.replace("step_", "")
            step_info = f"  step={step}"
            if os.path.isfile(cfg):
                import json
                with open(cfg) as f:
                    c = json.load(f)
                step_info += f"  d_model={c.get('d_model','?')}"
            print(f"  {name}  {size/1e6:.0f} MB{step_info}")
        else:
            print(f"  {name}  ({size/1e6:.0f} MB)")
    print(f"  Total: {len(dirs)} checkpoints, {total/1e6:.0f} MB")


def _cmd_data_stats(data_path, shape_path=None, num_samples=5, tokenizer_path="data/tokenizer.json"):
    """Show token data statistics and sample decoded text."""
    import json
    import numpy as np

    shape_path = shape_path or data_path.replace(".dat", "_shape.json")
    if not os.path.isfile(shape_path):
        print(f"Shape file not found: {shape_path}")
        return
    with open(shape_path) as f:
        meta = json.load(f)
    n_tokens = meta.get("n_tokens", "?")
    print(f"Data file: {data_path}")
    print(f"Shape:     {n_tokens:,} tokens" if isinstance(n_tokens, int) else f"Shape:  {n_tokens}")
    file_size = os.path.getsize(data_path)
    print(f"Size:      {file_size / 1e6:.0f} MB ({file_size:,} bytes)")
    print(f"Dtype:     uint16 ({file_size // 2:,} tokens)")

    # Try to load tokenizer and decode samples
    if os.path.isfile(tokenizer_path):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tokenizer_path)
            data = np.memmap(data_path, dtype=np.uint16, mode="r")
            print(f"\nSamples (first {num_samples}):")
            for i in range(min(num_samples, 5)):
                start = i * 512
                end = start + min(100, len(data) - start)
                ids = data[start:end].tolist()
                txt = tok.decode(ids)
                print(f"  [{i}] {txt[:120]}...")
        except Exception as e:
            print(f"  (tokenizer decode skipped: {e})")
    print()


def _cmd_ckpt_prune(base_dir, keep=5):
    """Prune old checkpoints, keep N latest."""
    import glob
    dirs = sorted(glob.glob(os.path.join(base_dir, "step_*")),
                  key=lambda d: os.path.getmtime(d))
    if len(dirs) <= keep:
        print(f"Only {len(dirs)} checkpoints, nothing to prune (keep={keep})")
        return
    remove = dirs[:-keep]
    for d in remove:
        import shutil
        shutil.rmtree(d)
        print(f"  Removed: {d}")
    print(f"Pruned {len(remove)}, kept {keep}")


def _fmt_step(step):
    """Zero-padded step string, e.g. 1234 → '01234'."""
    return f"{step:05d}"


def _find_latest_checkpoint(base_dir):
    """Find the most recent checkpoint directory under base_dir/.

    Looks for step_XXXXX/ dirs and returns the path + step number.
    Returns (None, 0) if nothing found.
    """
    import glob
    if not os.path.isdir(base_dir):
        return None, 0
    dirs = glob.glob(os.path.join(base_dir, "step_*"))
    if not dirs:
        # Maybe it's already a checkpoint dir (has config.json)
        if os.path.isfile(os.path.join(base_dir, "config.json")):
            return base_dir, 0
        return None, 0
    latest = max(dirs, key=lambda d: os.path.getmtime(d))
    try:
        step = int(os.path.basename(latest).replace("step_", ""))
    except ValueError:
        step = 0
    return latest, step


def _prompt_resume(default_dir, stage_name, yes_mode=False):
    """Check if a checkpoint exists and prompt to resume.

    Returns:
        resume_path or None
    """
    path, step = _find_latest_checkpoint(default_dir)
    if path is None:
        return None
    if yes_mode:
        print(f"  -> 使用最近检查点: {path} (step {step})")
        return path
    try:
        ans = input(f"  [{stage_name}] 检测到检查点 {path} (step {step}), 从中恢复? [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            return path
    except (EOFError, KeyboardInterrupt):
        pass
    return None


# ── mode 1: training ─────────────────────────────────────────────────────────

def train(args):
    """Training mode with clean model/memory separation.

    Stage 1 (train_model): Decoder-only LM pretraining.
        Delegates to train_lm.py. Trains gen_head standalone. No enc, no CB.

    Stage 2 (train_memory): Encoder + codebook training.
        Loads trained gen_head (frozen), trains encoder + all 6 lattices.

    Stage 3 (joint fine-tune): Combined training with lower LR.
    """
    _require_jax()
    from train.config import LCMConfig

    # ── Stage 1: train model (decoder LM) ────────────────────────────────
    if args.stage == 1:
        print("\nStage 1: decoder-only LM pretraining (gen_head)")
        from train.train_lm import train_lm
        train_lm(
            cfg=LCMConfig(),
            output_dir=args.save_dir or "checkpoints/lm_stage1",
            steps=args.steps,
            lr=args.lr or 3e-4,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            log_every=100,
            save_every=args.save,
            from_full_ckpt=args.load_lm,
            data_path=args.data,
            shape_path=args.shape or _make_shape_path(args.data) if args.data else None,
        )
        return

    # ── Stage 2: train memory (encoder + codebooks) ──────────────────────
    if args.stage == 2:
        print("\nStage 2: encoder + codebook training")
        from train.train_memory import train_memory
        resume = args.resume
        if not resume:
            out_dir = args.save_dir or "checkpoints"
            resume = _prompt_resume(out_dir, "Stage 2", args.yes)
        train_memory(
            cfg=LCMConfig(),
            data_path=args.data,
            shape_path=args.shape or _make_shape_path(args.data),
            output_dir=args.save_dir or "checkpoints",
            lm_checkpoint=args.load_lm,
            steps=args.memory_steps,
            lr=args.lr_stage2,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            log_every=100,
            save_every=args.save,
            resume_from=resume,
        )
        return

    # ── Stage 3: joint fine-tuning ──
    print("\nStage 3: joint fine-tuning (all params unfrozen)")
    from train.train import create_train_state, train_step, compute_codebook_utilization
    from train.data import WikiDataIter

    resume = args.resume
    cfg = _load_cfg(args)
    d = cfg.d_model
    shape_path = args.shape or _make_shape_path(args.data)
    save_dir = args.save_dir or "checkpoints"
    object.__setattr__(cfg, 'learning_rate', args.lr_stage3)

    print(f"Data:     {args.data}")
    print(f"Save to:  {save_dir}/step_XXXXX/  (every {args.save} steps)")
    print(f"Config:   d_model={d}  d_ff={cfg.d_ff}  n_heads={cfg.n_heads}")
    print(f"          B={args.batch_size}  N={args.seq_len}  "
          f"lr={cfg.learning_rate:.1e}")
    print(f"          steps={args.steps}  save={args.save}")

    # Init / resume
    rng = jax.random.PRNGKey(42)
    if not resume:
        resume = _prompt_resume(save_dir, "Stage 3", args.yes)
    if resume:
        from train.checkpoint import load_checkpoint as bin_load
        params, gvalue, opt_state, ckpt_step = bin_load(
            resume, cfg=None, rng=rng, load_opt=True)
        import json
        ckpt_cfg_path = os.path.join(resume, "config.json")
        if os.path.exists(ckpt_cfg_path):
            with open(ckpt_cfg_path) as f:
                for k, v in json.load(f).items():
                    if hasattr(cfg, k):
                        object.__setattr__(cfg, k, v)
        state = _build_resume_state(params, gvalue, opt_state, ckpt_step, cfg)
        start_step = ckpt_step
    else:
        state = create_train_state(cfg, rng)
        start_step = 0

    import train.train as train_mod
    original_get_cfg = train_mod._get_global_cfg
    train_mod._get_global_cfg = lambda: cfg

    data_iter = WikiDataIter(args.data, shape_path, B=args.batch_size, N=args.seq_len)

    def _handler(sig, frame):
        if state["step"] > 0:
            ckpt_dir = os.path.join(save_dir, f"step_{_fmt_step(state['step'])}")
            from train.checkpoint import save_checkpoint as bin_save
            bin_save(state, cfg, output_dir=ckpt_dir, step=state["step"])
            print(f"\nSaved interrupt checkpoint → {ckpt_dir}/")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)

    print(f"\nStep {start_step} → {args.steps} | checkpoints → {save_dir}/")

    from tqdm import tqdm
    pbar = tqdm(total=args.steps - start_step, desc="   training", unit="step",
                initial=start_step, position=0)
    for step in range(start_step, args.steps):
        batch = next(data_iter)
        rng, step_rng = jax.random.split(rng)
        state, comps = train_step(state, batch, step_rng)

        if step % 100 == 0:
            parts = [
                f"step {step:6d}",
                f"lm={float(comps['lm']):.4f}",
                f"vq={float(comps['vq']):.4f}",
            ]
            for key, label in [("contrast", "ctrst"), ("orth", "orth"),
                               ("val", "val"), ("ewc", "ewc"), ("margin", "mgn"),
                               ("self", "self")]:
                v = comps.get(key, 0.0)
                if v is not None:
                    parts.append(f"{label}={float(v):.4f}")
            tqdm.write("  ".join(parts))
            from train.model import forward as fwd_inner
            _, _, _, aux, _ = fwd_inner(
                state["params"], None, batch[0], cfg, training=True, rng=step_rng)
            util = compute_codebook_utilization(
                state["params"], aux,
                ema_state=state.get("ema_state"),
                seen_masks=state.get("seen_masks"))
            if util:
                util_str = "  ".join(
                    f"{k}={v:.0f}%/{v2:.0f}%" for k, (v, v2) in util.items())
                tqdm.write(f"  CB util: {util_str}")
            hrq_seen = int(jnp.sum(state.get("seen_masks", {}).get("hrq", jnp.zeros(1))))
            tqdm.write(f"  hrq_seen={hrq_seen}")

        if args.save > 0 and step > 0 and step % args.save == 0:
            ckpt_dir = os.path.join(save_dir, f"step_{_fmt_step(step)}")
            from train.checkpoint import save_checkpoint as bin_save
            bin_save(state, cfg, output_dir=ckpt_dir, step=step)

        pbar.update(1)

    pbar.close()

    final_ckpt_dir = os.path.join(save_dir, f"step_{_fmt_step(state['step'])}")
    from train.checkpoint import save_checkpoint as bin_save
    bin_save(state, cfg, output_dir=final_ckpt_dir, step=state["step"])
    train_mod._get_global_cfg = original_get_cfg
    print("Training complete.")


# ── mode 2: interactive chat (C inference engine) ─────────────────────────────

def interact(args):
    """Interactive REPL using the C cognitive inference engine.

    No JAX dependency — pure numpy encoder + C DAG engine + gen_head.
    """
    print(f"Loading checkpoint: {args.interact}")

    # Silence tokenizers parallelism warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    obs_params = {}
    if getattr(args, 'obs', False):
        obs_params = {
            'raw_capacity': args.raw_capacity,
            'obs_every_n': args.obs_every_n,
            'lt_enabled': args.lt_enabled,
            'lt_max_records': args.lt_max_records,
        }
        print(f"[OBS]  Black box: raw={args.raw_capacity} every_n={args.obs_every_n}")
        print(f"[NARR] Narrative memory: {'on' if args.lt_enabled else 'off'} "
              f"max={args.lt_max_records}")

    engine = LCMInferEngine(args.interact, **obs_params)

    # ── Prediction cache ──
    if getattr(args, 'pred', False):
        engine.enable_prediction(
            cache_capacity=args.pred_cache,
            reflect_interval=args.pred_reflect)

    # ── Causal subject ──
    if getattr(args, 'causal', False):
        engine.enable_causal(
            enable_counterfactual=getattr(args, 'pred', False))

    max_new = args.max_new
    temp = args.temperature

    print(f"Ready.  max_new={max_new}  temp={temp}  "
          f"decoder={engine.decoder['format']}")
    print("─" * 50)

    if args.active:
        # ── Active output mode ────────────────────────────────────────
        # The engine runs a continuous cognitive loop, decoding every
        # step's z into language tokens. No "silence" or threshold —
        # language IS the output of the cognitive process.
        print("[ACTIVE] Continuous language output loop.")
        print("[ACTIVE] Type any prompt to redirect; Ctrl+C or 'quit' to exit.")
        print("─" * 50)

        import select as _select
        bos_token = 2
        eos_token = 3

        token_ids = [bos_token]
        enc_state = None
        kv_cache = None
        active_running = True

        def _handle_stdin(prompt):
            nonlocal token_ids, enc_state, kv_cache, active_running
            prompt = prompt.strip()
            if not prompt:
                return
            if prompt.lower() in ("quit", "exit"):
                active_running = False
                return
            if prompt.startswith("/"):
                return
            print(f"> {prompt}")
            print("─── Generating ───")
            encoding = engine.tokenizer.encode(prompt)
            token_ids = encoding.ids
            if not token_ids or token_ids[0] != bos_token:
                token_ids = [bos_token] + token_ids
            enc_state = None
            kv_cache = None
            for tid in engine.generate(
                    prompt, max_new=max_new, temperature=temp,
                    use_loop=args.loop, show_trace=args.trace):
                text = engine.tokenizer.decode([tid], skip_special_tokens=True)
                print(text, end="", flush=True)
                token_ids.append(tid)
                if tid == eos_token:
                    break
            print()
            token_ids = token_ids[-engine.max_seq_len:]
            if engine.decoder['format'] != 'old':
                kv_cache = None
            enc_state = None

        while active_running:
            rlist, _, _ = _select.select([sys.stdin], [], [], args.active_interval)
            if rlist:
                line = sys.stdin.readline()
                if line:
                    _handle_stdin(line.rstrip('\n'))
                continue

            x = np.array(token_ids[-engine.max_seq_len:], dtype=np.int32)

            if enc_state is None:
                z, enc_state = _encoder_full_with_state(engine.encoder, x, engine.n_heads)
            else:
                z = encoder_recurrent_step_cy(engine.encoder, enc_state,
                                              token_ids[-1], engine.n_heads)

            z_q, _ = engine.cognitive_loop(
                z, use_safety=False,
                agency_tension_mod=engine._last_agency_mod['tension'],
                agency_explore_mod=engine._last_agency_mod['explore'])

            traces = engine.get_trace()
            if traces:
                from train.predictive_cache import pack_trace_sig
                last_t = traces[-1]
                engine._last_sig = pack_trace_sig(
                    last_t.get('weights', np.zeros(7)),
                    last_t.get('confidences', np.zeros(7)),
                    last_t.get('has_conflict', False))

            if engine.decoder['format'] == 'old':
                logits = gen_head_forward_old(engine.decoder, z_q, x)
                kv_cache = None
            else:
                logits, kv_cache = gen_head_new_single_cy(engine.decoder, z_q, x, kv_cache)

            next_id = sample_categorical(logits, temp, top_k=50)
            text = engine.tokenizer.decode([next_id], skip_special_tokens=True)
            print(text, end="", flush=True)
            token_ids.append(next_id)

            if len(token_ids) > engine.max_seq_len * 2:
                token_ids = token_ids[-engine.max_seq_len:]

        print("\n[ACTIVE] Loop ended.")
        return

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            break

        # ── Special commands ──
        if prompt.startswith("/"):
            cmd = prompt[1:].strip().split()
            if cmd[0] == "flag" and len(cmd) >= 2:
                try:
                    step = int(cmd[1])
                    record = engine.obs.ring.pop_by_step(step)
                    if engine.narr is not None:
                        nrec = engine.narr.promote(record)
                        print(f"[NARR] Step {step} promoted to long-term narrative"
                              f"{' ✓' if nrec else ' — not found'}"
                              f"{' (pinned)' if nrec and nrec.pinned else ''}")
                    else:
                        print("[NARR] Narrative memory not enabled")
                except ValueError:
                    print("[NARR] Usage: /flag <step_number>")
            elif cmd[0] == "pin" and len(cmd) >= 2:
                try:
                    step = int(cmd[1])
                    if engine.narr is not None:
                        ok = engine.narr.pin(step)
                        print(f"[NARR] Step {step} pinned"
                              f"{' ✓' if ok else ' — not found'}")
                    else:
                        print("[NARR] Narrative memory not enabled")
                except ValueError:
                    print("[NARR] Usage: /pin <step_number>")
            elif cmd[0] == "unpin" and len(cmd) >= 2:
                try:
                    step = int(cmd[1])
                    if engine.narr is not None:
                        ok = engine.narr.unpin(step)
                        print(f"[NARR] Step {step} unpinned"
                              f"{' ✓' if ok else ' — not found'}")
                    else:
                        print("[NARR] Narrative memory not enabled")
                except ValueError:
                    print("[NARR] Usage: /unpin <step_number>")
            elif cmd[0] == "narrative":
                if engine.narr is None:
                    print("[NARR] Narrative memory not enabled")
                else:
                    engine.narr.print_narrative()
            elif cmd[0] == "obs-summary":
                engine.obs.print_summary()
                if engine.narr is not None:
                    engine.narr.print_summary()
                if engine.pred is not None:
                    engine._print_pred_stats()
            elif cmd[0] == "pred-stats":
                if engine.pred is not None:
                    engine._print_pred_stats()
                else:
                    print("[PRED] Prediction cache not enabled (use --pred to enable)")
            elif cmd[0] == "causal":
                engine._print_causal_stats()
            else:
                print(f"Unknown command: /{cmd[0]}")
                print("  /flag <step>     — Flag step as important, promote to long-term narrative")
                print("  /pin <step>      — Pin step, protect from forgetting")
                print("  /unpin <step>    — Unpin step")
                print("  /narrative       — Show long-term narrative timeline")
                print("  /obs-summary     — Show observation/prediction system stats")
                print("  /pred-stats      — Show prediction cache details")
                print("  /causal          — Show causal subject stats")
            continue

        print("─── Generating ───")
        try:
            for token_id in engine.generate(
                    prompt, max_new=max_new, temperature=temp,
                    use_loop=args.loop, show_trace=args.trace):
                text = engine.tokenizer.decode([token_id], skip_special_tokens=True)
                print(text, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        print()

        # Observability post-generation summary / export
        if args.obs_summary:
            engine.obs.print_summary()
            if engine.narr is not None:
                engine.narr.print_summary()

        # Narrative consolidation (model-driven forgetting after each turn)
        if engine.narr is not None and args.narr_consolidate:
            try:
                dropped = engine.narr.consolidate(
                    keep_threshold=args.narr_keep_threshold)
                if dropped > 0:
                    print(f"[NARR] Post-generation: dropped {dropped} records "
                          f"(keep ≥ {args.narr_keep_threshold})")
            except Exception as e:
                print(f"[NARR] Consolidation error: {e}")

        if args.obs_export:
            if engine.narr is not None:
                engine.narr.export_json(args.obs_export)


# ── mode 3: preprocess ────────────────────────────────────────────────────────

def preprocess(args):
    """Tokenise a text file and write a uint16 memmap array."""
    _require_jax()
    # Auto-train tokenizer if missing
    if not os.path.exists(args.tokenizer):
        if not args.vocab_size:
            print(f"Error: tokenizer not found at {args.tokenizer}")
            print("       Provide --vocab-size to train a new tokenizer.")
            sys.exit(1)
        print(f"Training tokenizer (vocab_size={args.vocab_size}) ...")
        from train.data import train_tokenizer
        train_tokenizer(
            text_path=args.input,
            tokenizer_path=args.tokenizer,
            vocab_size=args.vocab_size,
        )

    tokenizer = _load_tokenizer(args.tokenizer)

    # If output is /dev/null, this was just a tokenizer training run
    if args.output == "/dev/null":
        print("Tokenizer training complete. Skipping mmap write.")
        return

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    print(f"Tokenising: {args.input}")
    from tqdm import tqdm

    # Auto-detect JSONL: if the first non-empty line parses as JSON with a "text" key
    import json as _json
    _is_jsonl = False
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    _is_jsonl = isinstance(_json.loads(line), dict)
                except Exception:
                    _is_jsonl = False
                break

    def _iter_lines(path):
        """Yield text lines from plain text or JSONL."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if _is_jsonl:
                    obj = _json.loads(line)
                    line = obj.get("text", obj.get("content", ""))
                    if not line:
                        continue
                yield line

    # First pass: count tokens
    total = 0
    n_lines = 0
    for _ in _iter_lines(args.input):
        n_lines += 1
    for line in tqdm(_iter_lines(args.input), desc="   counting tokens", total=n_lines, unit="line"):
        total += len(tokenizer.encode(line).ids)

    print(f"Total tokens: {total:,}")

    # Create pre-sized memmap
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fp = np.memmap(args.output, dtype=np.uint16, mode="w+", shape=(total,))

    # Second pass: tokenize and write directly
    pos = 0
    for line in tqdm(_iter_lines(args.input), desc="   tokenizing", total=n_lines, unit="line"):
        ids = np.array(tokenizer.encode(line).ids, dtype=np.uint16)
        fp[pos:pos + len(ids)] = ids
        pos += len(ids)

    fp.flush()

    # Write companion shape file
    shape_path = _make_shape_path(args.output)
    import json
    with open(shape_path, "w") as f:
        json.dump({"n_tokens": total}, f)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done: {total:,} tokens → {args.output} ({size_mb:.1f} MB)")
    print(f"Shape:  {shape_path}")


# ── mode 4: data cleaning (standalone pipeline) ───────────────────────────────

def clean(args):
    """5-stage data cleaning pipeline.

    Stages:
      ① Language filter — FastTextLangId, removes non-Chinese documents
      ② Unicode repair — UnicodeReformatter (ftfy), normalises mojibake
      ③ Heuristic filter — WordCountFilter + custom regex (copyright, ISBN, etc.)
      ④ Exact dedup — SHA-256, removes byte-identical documents
      ⑤ Fuzzy dedup — MinHash+LSH, removes near-duplicates

    Input: JSONL (one JSON document per line) from WikiExtractor or similar.
    Output: Cleaned JSONL at --output path.

    Uses NeMo Curator filter components directly (not the distributed pipeline
    framework, which was restructured in nemo-curator 1.x).
    """
    try:
        from nemo_curator.stages.text.filters import (
            DocumentFilter,
            FastTextLangId,
            WordCountFilter,
        )
        from nemo_curator.stages.text.modifiers import UnicodeReformatter
        HAS_NEMO = True
    except ImportError as e:
        print(f"Warning: limited nemo-curator install — {e}")
        HAS_NEMO = False
        DocumentFilter = object
        FastTextLangId = None
        WordCountFilter = None
        UnicodeReformatter = None

    import json, hashlib, struct
    from collections import defaultdict
    from tqdm import tqdm

    # ── helpers ─────────────────────────────────────────────────────────

    def _load_jsonl(path):
        """Yield parsed JSON objects from a JSONL file."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ── custom filter: wiki template residue ────────────────────────────
    class WikiNoiseFilter(DocumentFilter if HAS_NEMO else object):
        """Reject documents containing Wikipedia template/footnote residue."""
        def __init__(self):
            if HAS_NEMO:
                super().__init__()
            self._name = "wiki_noise"
            self._pattern = re.compile(
                r"(版权所有|修订于|最后编辑|ISBN分类：|"
                r"cite book|cite web|cite news|cite journal|eprint=|"
                r"reflist|DEFAULTSORT|Authority control|"
                r"Commons category|Wikidata|coord|"
                r"Use dmy dates|Use mdy dates|"
                r"multiple issues|orphan|stub|"
                r"dead link|deadurl|citation needed|"
                r"^\* |^\| alt=|^=== |"
                r"{{[^}]+}}|&lt;ref&gt;|</ref>)",
                re.MULTILINE | re.IGNORECASE,
            )

        def score_document(self, text: str):
            return 0 if self._pattern.search(text) else 1

        def keep_document(self, score: int):
            return score > 0

    # ── MinHash for fuzzy dedup ─────────────────────────────────────────
    _MH_SEED = 42

    def _minhash(text, n=5, num_hashes=256):
        """Compute a simple MinHash signature (list of hashes)."""
        shingles = set()
        text_lower = text.lower()
        for i in range(len(text_lower) - n + 1):
            shingle = text_lower[i:i + n]
            shingles.add(hashlib.sha256(shingle.encode()).digest()[:4])
        signatures = []
        for i in range(num_hashes):
            seed = struct.pack(">I", _MH_SEED + i)
            sig = min(hashlib.sha256(seed + s).digest()[:4] for s in shingles) if shingles else b"\x00" * 4
            signatures.append(struct.unpack(">I", sig)[0])
        return signatures

    def _jaccard_from_signatures(sig_a, sig_b):
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    # ── main pipeline ───────────────────────────────────────────────────

    print(f"[CLEAN] Input:   {args.input}")
    print(f"[CLEAN] Output:  {args.output}")
    print(f"[CLEAN] Stages:  lang_filter → unicode_repair → heuristic_filter → "
          f"exact_dedup → fuzzy_dedup")
    print()

    # Stage ①: read
    docs = list(_load_jsonl(args.input))
    n0 = len(docs)
    print(f"[CLEAN] Read {n0} documents")

    # Stage ②: language filter
    lang_model_path = args.langid_model or "lid.176.bin"
    if HAS_NEMO and FastTextLangId is not None and os.path.exists(lang_model_path):
        print("[CLEAN] Stage ① — Language filter (FastText) ...")
        lang_filter = FastTextLangId(
            model_path=lang_model_path,
            min_langid_score=args.langid_threshold,
        )
        lang_filter.load_model()
        kept = []
        n_non_zh = 0
        for doc in tqdm(docs, desc="   language filter", unit="doc"):
            text = doc.get("text", "")
            raw = lang_filter.score_document(text)  # returns "[score, LANG]"
            import ast
            score, lang_code = ast.literal_eval(raw)
            if lang_code == "ZH" and score >= args.langid_threshold:
                kept.append(doc)
            else:
                n_non_zh += 1
        docs = kept
        print(f"         Removed {n_non_zh} non-Chinese documents")
    else:
        print("[CLEAN] Stage ① — Language filter skipped (fasttext model not available)")

    # Stage ③: Unicode repair
    if HAS_NEMO and UnicodeReformatter is not None:
        print("[CLEAN] Stage ② — Unicode repair ...")
        reformatter = UnicodeReformatter(normalization="NFKC", fix_character_width=True)
        for doc in tqdm(docs, desc="   unicode repair", unit="doc"):
            text = doc.get("text", "")
            if text:
                doc["text"] = reformatter.modify_document(text)
    else:
        print("[CLEAN] Stage ② — Unicode repair skipped")

    # Stage ④: heuristic filters
    print("[CLEAN] Stage ③ — Heuristic filtering ...")
    n_word_removed = 0
    n_noise_removed = 0
    wc_filter = WordCountFilter(min_words=args.min_words, lang="zh") if HAS_NEMO and WordCountFilter is not None else None
    noise_filter = WikiNoiseFilter()
    kept = []
    for doc in tqdm(docs, desc="   heuristic filter", unit="doc"):
        text = doc.get("text", "")
        # word count check
        if wc_filter is not None:
            wc_score = wc_filter.score_document(text)
            if not wc_filter.keep_document(wc_score):
                n_word_removed += 1
                continue
        # wiki noise check
        noise_score = noise_filter.score_document(text)
        if not noise_filter.keep_document(noise_score):
            n_noise_removed += 1
            continue
        kept.append(doc)
    docs = kept
    print(f"         Removed {n_word_removed} by word count, {n_noise_removed} by noise filter")

    # Stage ⑤: exact dedup
    if not args.skip_exact_dedup:
        print("[CLEAN] Stage ④ — Exact dedup (SHA-256) ...")
        seen_hashes = set()
        kept = []
        for doc in tqdm(docs, desc="   exact dedup", unit="doc"):
            text = doc.get("text", "")
            h = hashlib.sha256(text.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                kept.append(doc)
        n_exact_dup = len(docs) - len(kept)
        docs = kept
        print(f"         Removed {n_exact_dup} exact duplicates")
    else:
        print("[CLEAN] Stage ④ — Exact dedup skipped")

    # Stage ⑥: fuzzy dedup
    if not args.skip_fuzzy_dedup:
        print("[CLEAN] Stage ⑤ — Fuzzy dedup (MinHash+LSH) ...")
        # Banding LSH: hash each doc into buckets, compare within buckets
        num_hashes = args.num_hashes
        num_bands = args.num_buckets
        rows_per_band = num_hashes // num_bands
        threshold = args.fuzzy_threshold

        sigs = {}
        for i, doc in enumerate(tqdm(docs, desc="   minhash sigs", unit="doc")):
            sigs[i] = _minhash(doc.get("text", ""), n=args.min_ngram, num_hashes=num_hashes)
        buckets = defaultdict(list)
        for idx, sig in tqdm(sigs.items(), desc="   LSH bucketing", unit="doc"):
            for band in range(num_bands):
                band_hash = hash(tuple(sig[band * rows_per_band:(band + 1) * rows_per_band]))
                buckets[(band, band_hash)].append(idx)

        removed = set()
        for band_bucket in tqdm(buckets.values(), desc="   compare candidates", unit="bucket"):
            if len(band_bucket) < 2:
                continue
            for i in range(len(band_bucket)):
                if band_bucket[i] in removed:
                    continue
                for j in range(i + 1, len(band_bucket)):
                    if band_bucket[j] in removed:
                        continue
                    sim = _jaccard_from_signatures(sigs[band_bucket[i]], sigs[band_bucket[j]])
                    if sim > threshold:
                        removed.add(band_bucket[j])

        n_fuzzy_dup = len(removed)
        docs = [d for i, d in enumerate(docs) if i not in removed]
        print(f"         Removed {n_fuzzy_dup} near-duplicates (threshold={threshold})")
    else:
        print("[CLEAN] Stage ⑤ — Fuzzy dedup skipped")

    # Write output
    n_final = len(docs)
    print(f"[CLEAN] {n0} → {n_final} documents (removed {n0 - n_final})")
    print(f"[CLEAN] Writing → {args.output}")
    with tqdm(total=n_final, desc="   writing", unit="doc") as pbar:
        with open(args.output, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                pbar.update(1)
    print("[CLEAN] Done.")


# ── mode 5: C inference engine (pure numpy + liblcm.so) ──────────────────────

import ctypes
from pathlib import Path

_INFER_DIR = Path(__file__).parent / "infer"


def _elu(x):
    return np.where(x < 0, np.exp(x) - 1, x)


def _elu_plus_one(x):
    return _elu(x) + 1.0


def _layer_norm(x, scale, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


HEADER_FMT = '<iiiifI'
HEADER_SIZE = 24


def _read_bin_header(path):
    with open(path, 'rb') as f:
        hdr = f.read(HEADER_SIZE)
    M, dim, n_layers, cb_type, c, crc = struct.unpack(HEADER_FMT, hdr)
    return int(M), int(dim), int(n_layers), int(cb_type), float(c), int(crc)


def _read_bin_data(path, has_hash=False):
    with open(path, 'rb') as f:
        f.seek(HEADER_SIZE)
        data = f.read()
    if has_hash:
        data = data[:-32]
    return np.frombuffer(data, dtype=np.float32)


def load_encoder(ckpt_dir, d, n_layers, V, d_ff, max_seq_len):
    path = Path(ckpt_dir) / "encoder.bin"
    flat = np.fromfile(path, dtype=np.float32)
    pos = 0
    embed = flat[pos:pos + V*d].reshape(V, d); pos += V*d
    rel_bias = flat[pos:pos + 2*max_seq_len - 1]; pos += 2*max_seq_len - 1
    n_attn = d * d
    layers = []
    for _ in range(n_layers):
        ln1_s = flat[pos:pos+d].copy(); pos += d
        ln1_b = flat[pos:pos+d].copy(); pos += d
        w_q = flat[pos:pos+n_attn].reshape(d, d).copy(); pos += n_attn
        w_k = flat[pos:pos+n_attn].reshape(d, d).copy(); pos += n_attn
        w_v = flat[pos:pos+n_attn].reshape(d, d).copy(); pos += n_attn
        w_o = flat[pos:pos+n_attn].reshape(d, d).copy(); pos += n_attn
        ln2_s = flat[pos:pos+d].copy(); pos += d
        ln2_b = flat[pos:pos+d].copy(); pos += d
        w_1 = flat[pos:pos+d*d_ff].reshape(d, d_ff).copy(); pos += d*d_ff
        w_2 = flat[pos:pos+d*d_ff].reshape(d, d_ff).copy(); pos += d*d_ff
        w_3 = flat[pos:pos+d_ff*d].reshape(d_ff, d).copy(); pos += d_ff*d
        layers.append({'ln1_scale': ln1_s, 'ln1_bias': ln1_b,
                       'w_q': w_q, 'w_k': w_k, 'w_v': w_v, 'w_o': w_o,
                       'ln2_scale': ln2_s, 'ln2_bias': ln2_b,
                       'w_1': w_1, 'w_2': w_2, 'w_3': w_3})
    q_pool = flat[pos:pos+d].copy(); pos += d
    w_proj = flat[pos:pos+d*d].reshape(d, d).copy()
    return {'embed': embed, 'rel_bias': rel_bias,
            'layers': layers, 'q_pool': q_pool, 'w_proj': w_proj}


def load_decoder(ckpt_dir, d, V):
    path = Path(ckpt_dir) / "decoder.bin"
    flat = np.fromfile(path, dtype=np.float32)
    new_size = V*d + 4*d*d + 2*d*4*d + 4*d*V
    old_size = d*d + d*V
    if len(flat) == new_size:
        pos = 0
        w_embed = flat[pos:pos+V*d].reshape(V, d); pos += V*d
        w_q = flat[pos:pos+d*d].reshape(d, d); pos += d*d
        w_k = flat[pos:pos+d*d].reshape(d, d); pos += d*d
        w_v = flat[pos:pos+d*d].reshape(d, d); pos += d*d
        w_o = flat[pos:pos+d*d].reshape(d, d); pos += d*d
        w_1 = flat[pos:pos+d*4*d].reshape(d, d*4); pos += d*4*d
        w_2 = flat[pos:pos+d*4*d].reshape(d, d*4); pos += d*4*d
        w_3 = flat[pos:pos+4*d*V].reshape(d*4, V)
        return {'format': 'new', 'w_embed': w_embed, 'w_q': w_q, 'w_k': w_k,
                'w_v': w_v, 'w_o': w_o, 'w_1': w_1, 'w_2': w_2, 'w_3': w_3}
    if len(flat) in (old_size, 0):
        if len(flat) == old_size:
            w_proj = flat[:d*d].reshape(d, d).copy()
            w_out = flat[d*d:].reshape(d, V).copy()
        else:
            w_proj = np.random.randn(d, d).astype(np.float32) * (d ** -0.5)
            w_out = np.random.randn(d, V).astype(np.float32) * (d ** -0.5)
        return {'format': 'old', 'w_proj': w_proj, 'w_out': w_out}
    raise ValueError(f"decoder.bin: unexpected size {len(flat)}")


def load_codebook_for_c(ckpt_dir, filename):
    path = Path(ckpt_dir) / filename
    M, d, n_layers, cb_type, c, crc = _read_bin_header(path)
    data = _read_bin_data(path,
                          has_hash=(filename.startswith('gvalue') or filename.startswith('danger')))
    return np.ascontiguousarray(data, dtype=np.float32), M, d


# ── Encoder forward ─────────────────────────────────────────────────────────

def encoder_forward(params, token_ids, n_heads):
    N = len(token_ids)
    d = params['embed'].shape[1]
    h = params['embed'][token_ids]
    for layer in params['layers']:
        h_norm = _layer_norm(h, layer['ln1_scale'], layer['ln1_bias'])
        h_attn = _linear_attn(h_norm, layer['w_q'], layer['w_k'],
                              layer['w_v'], layer['w_o'], n_heads)
        h = h + h_attn
        h_norm = _layer_norm(h, layer['ln2_scale'], layer['ln2_bias'])
        h_glu = _glu(h_norm, layer['w_1'], layer['w_2'], layer['w_3'])
        h = h + h_glu
    q = _elu_plus_one(params['q_pool'][None, :])
    k = _elu_plus_one(h)
    v = h
    z = (q @ (k.T @ v)) / (q @ k.sum(axis=0, keepdims=True).T + 1e-6)
    return z[0] @ params['w_proj']


def _linear_attn(x, w_q, w_k, w_v, w_o, n_heads):
    N, D = x.shape
    d_h = D // n_heads
    Q = _elu_plus_one(x @ w_q).reshape(N, n_heads, d_h).transpose(1, 0, 2)
    K = _elu_plus_one(x @ w_k).reshape(N, n_heads, d_h).transpose(1, 0, 2)
    V = (x @ w_v).reshape(N, n_heads, d_h).transpose(1, 0, 2)
    kv = np.einsum('hnd,hne->hde', K, V)
    Z = np.einsum('hnd,hde->hne', Q, kv)
    norm = np.einsum('hnd,hkd->hnk', Q, K.sum(axis=1, keepdims=True)).squeeze(-1)
    Z = Z / (norm[:, :, None] + 1e-6)
    return Z.transpose(1, 0, 2).reshape(N, D) @ w_o


def _glu(x, w_1, w_2, w_3):
    return (_sigmoid(x @ w_1) * (x @ w_2)) @ w_3


# ─── Incremental encoder (recurrent, O(d²) per step) ─────────────────────
#
#   The batch encoder recomputes the full sliding window each step (O(N·d²)).
#   The incremental version maintains per-layer KV cumsums and appends each
#   new token's contribution in O(d²) — same approach as the decoder's kv_cache.
#
#   Cumulative sums grow unboundedly with generation length.  A full reset
#   (re-encode from scratch) every ENC_RESET_INTERVAL steps bounds drift.
#
#   NOTE: The batch encoder is BIDIRECTIONAL; the incremental version is
#   effectively CAUSAL (new tokens attend to all past, but old tokens don't
#   see new ones).  This is more appropriate for autoregressive generation.

ENC_RESET_INTERVAL = 256  # full re-encode every N steps to reset cumsum


def _encoder_full_with_state(params, x, n_heads):
    """Full batch encode + return recurrent state for incremental updates.

    Args:
        params: Encoder param dict.
        x: (N,) int32 token IDs for the full prompt.
        n_heads: Number of attention heads.

    Returns:
        z: (d,) bottleneck vector.
        state: dict with per-layer cumsum caches.
    """
    N = len(x)
    d = params['embed'].shape[1]
    d_h = d // n_heads

    h = params['embed'][x]  # (N, d)

    layers_state = []
    for layer in params['layers']:
        h_norm = _layer_norm(h, layer['ln1_scale'], layer['ln1_bias'])

        Q = _elu_plus_one(h_norm @ layer['w_q'])  # (N, d)
        K = _elu_plus_one(h_norm @ layer['w_k'])  # (N, d)
        V = h_norm @ layer['w_v']                  # (N, d)

        # Multi-head split
        q = Q.reshape(N, n_heads, d_h).transpose(1, 0, 2)  # (H, N, d_h)
        k = K.reshape(N, n_heads, d_h).transpose(1, 0, 2)
        v = V.reshape(N, n_heads, d_h).transpose(1, 0, 2)

        # KV cumsum over all N positions (bidirectional)
        kv = np.einsum('hnd,hne->hde', k, v)  # (H, d_h, d_h)
        k_sum = k.sum(axis=1)                  # (H, d_h)

        # Full attention output
        Z = np.einsum('hnd,hde->hne', q, kv)
        norm = np.einsum('hnd,hd->hn', q, k_sum)[:, :, None]
        Z = Z / (norm + 1e-6)
        Z = Z.transpose(1, 0, 2).reshape(N, d) @ layer['w_o']

        h_new = h + Z
        h_new = h_new + _glu(
            _layer_norm(h_new, layer['ln2_scale'], layer['ln2_bias']),
            layer['w_1'], layer['w_2'], layer['w_3'])
        h = h_new

        layers_state.append({'kv': kv.copy(), 'k': k_sum.copy()})

    # Global attention pooling (bidirectional sum over N)
    k_pool = _elu_plus_one(h)
    pool_kv = np.einsum('nd,ne->de', k_pool, h)
    pool_k = k_pool.sum(axis=0)

    q_pool = _elu_plus_one(params['q_pool'][None, :])[0]
    z = (q_pool @ pool_kv) / (q_pool @ pool_k + 1e-6)
    z = z @ params['w_proj']

    state = {
        'embed': params['embed'],
        'layers': layers_state,
        'pool_kv': pool_kv,
        'pool_k': pool_k,
        'q_pool': params['q_pool'],
        'w_proj': params['w_proj'],
    }
    return z, state


def _encoder_recurrent_step(state, token_id, layer_params, n_heads):
    """Incremental encoder update for one new token.

    Updates cumsums in-place and returns the new z.  O(d²) per layer.

    Args:
        state: Mutable recurrent state from _encoder_full_with_state.
        token_id: int, the new token to append.
        layer_params: params['layers'] list.
        n_heads: Number of attention heads.

    Returns:
        z_new: (d,) updated bottleneck.
    """
    d = state['q_pool'].shape[0]
    d_h = d // n_heads

    h = state['embed'][token_id]  # (d,)

    for l, layer in enumerate(layer_params):
        ls = state['layers'][l]

        h_norm = _layer_norm(h, layer['ln1_scale'], layer['ln1_bias'])

        q = _elu_plus_one(h_norm @ layer['w_q']).reshape(n_heads, d_h)
        k = _elu_plus_one(h_norm @ layer['w_k']).reshape(n_heads, d_h)
        v = (h_norm @ layer['w_v']).reshape(n_heads, d_h)

        # Append to cumsum
        ls['kv'] += np.einsum('hd,he->hde', k, v)
        ls['k'] += k

        # Attention: φ(q) @ kv / (φ(q) @ k_sum)
        num = np.einsum('hd,hde->he', q, ls['kv'])
        den = np.einsum('hd,hd->h', q, ls['k'])
        attn_out = (num / (den[:, None] + 1e-6)).reshape(d) @ layer['w_o']

        h = h + attn_out
        h = h + _glu(
            _layer_norm(h, layer['ln2_scale'], layer['ln2_bias']),
            layer['w_1'], layer['w_2'], layer['w_3'])

    # Pooling
    k_pool = _elu_plus_one(h)
    state['pool_kv'] += np.outer(k_pool, h)
    state['pool_k'] += k_pool

    q_pool = _elu_plus_one(state['q_pool'][None, :])[0]
    z = (q_pool @ state['pool_kv']) / (q_pool @ state['pool_k'] + 1e-6)
    return z @ state['w_proj']


# ── Gen_head forward ────────────────────────────────────────────────────────

def gen_head_new_single(params, z_q, token_ids, kv_cache=None):
    d = len(z_q)
    if kv_cache is None:
        k_start = _elu_plus_one(z_q @ params['w_k'])
        v_start = z_q @ params['w_v']
        kv_sum = np.outer(k_start, v_start)
        k_sum = k_start
        token_embed = z_q
    else:
        kv_sum, k_sum = kv_cache
        token_embed = params['w_embed'][token_ids[-1]]
    q = _elu_plus_one(token_embed @ params['w_q'])
    k = _elu_plus_one(token_embed @ params['w_k'])
    v = token_embed @ params['w_v']
    kv_sum += np.outer(k, v)
    k_sum += k
    attn_out = ((q @ kv_sum) / (np.dot(q, k_sum) + 1e-8)) @ params['w_o']
    glu_out = _sigmoid(attn_out @ params['w_1']) * (attn_out @ params['w_2'])
    return glu_out @ params['w_3'], (kv_sum, k_sum)


def gen_head_forward_old(params, z_q, token_ids):
    return _elu(z_q @ params['w_proj']) @ params['w_out']


def sample_categorical(logits, temperature=0.7, top_k=50):
    """Sample token ID from logits with temperature + top-k."""
    # Cython-accelerated path (fused softmax + CDF walk, no per-call Generator)
    try:
        from train._lcm_cy import sample_categorical_cy as _sc
        return _sc(np.ascontiguousarray(logits, dtype=np.float32), temperature, top_k)
    except ImportError:
        pass
    # Pure-Python fallback
    logits = logits / max(temperature, 1e-8)
    if top_k > 0 and top_k < len(logits):
        logits[logits < -np.sort(-logits)[top_k]] = -1e9
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(np.random.default_rng().choice(len(probs), p=probs))


def gen_head_new_single_cy(path, z_q, token_ids, kv_cache=None):
    """Cython-accelerated gen_head forward with pure-Python fallback.

    Wraps genhead_step_cy (fused manual loops for d×d ops) with BLAS for the
    final d_ff×V logit projection.  Pure-numpy fallback matches original
    gen_head_new_single signature.
    """
    try:
        from train._lcm_cy import genhead_step_cy as _gh
        from train._lcm_cy import init_rng_cy as _ir
        _ir(0)  # seed once (no-op on subsequent calls)
    except ImportError:
        return gen_head_new_single(path, z_q, token_ids, kv_cache)

    d = len(z_q)
    is_first = kv_cache is None
    if is_first:
        kv_sum = np.zeros((d, d), dtype=np.float32)
        k_sum = np.zeros(d, dtype=np.float32)
    else:
        kv_sum, k_sum = kv_cache

    gate_act = np.empty(path['w_1'].shape[1], dtype=np.float32)
    last_id = token_ids[-1] if not is_first else 0
    _gh(path['w_embed'], path['w_q'], path['w_k'], path['w_v'], path['w_o'],
        path['w_1'], path['w_2'], z_q, last_id,
        kv_sum, k_sum, int(is_first), gate_act)

    logits = gate_act @ path['w_3']
    return logits, (kv_sum, k_sum)


def encoder_recurrent_step_cy(path, state, token_id, n_heads):
    """Cython-accelerated incremental encoder step with pure-Python fallback."""
    try:
        from train._lcm_cy import encoder_recurrent_step_cy as _er
    except ImportError:
        return _encoder_recurrent_step(state, token_id, path['layers'], n_heads)

    d = path['q_pool'].shape[0]
    z_out = np.empty(d, dtype=np.float32)
    _er(
        path['embed'], path['q_pool'], path['w_proj'],
        [ls['kv'] for ls in state['layers']],
        [ls['k'] for ls in state['layers']],
        state['pool_kv'], state['pool_k'],
        [l['ln1_scale'] for l in path['layers']],
        [l['ln1_bias'] for l in path['layers']],
        [l['ln2_scale'] for l in path['layers']],
        [l['ln2_bias'] for l in path['layers']],
        [l['w_q'] for l in path['layers']],
        [l['w_k'] for l in path['layers']],
        [l['w_v'] for l in path['layers']],
        [l['w_o'] for l in path['layers']],
        [l['w_1'] for l in path['layers']],
        [l['w_2'] for l in path['layers']],
        [l['w_3'] for l in path['layers']],
        token_id, n_heads, z_out)
    return z_out


def trace_step_conflict_source(trace_step):
    """Extract conflict source from a C engine trace step dict."""
    if trace_step.get('has_conflict'):
        return 'cengine'
    return None


# ── LCMInferEngine ──────────────────────────────────────────────────────────

class LCMInferEngine:
    """Inference engine: Python (encoder/gen_head) + C (cognitive DAG)."""

    def __init__(self, checkpoint_dir, liblcm_path=None,
                 raw_capacity=1000, obs_every_n=1,
                 lt_enabled=True, lt_max_records=10000):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lib_path = liblcm_path or (_INFER_DIR / "liblcm.so")

        with open(self.checkpoint_dir / "config.json") as f:
            self.cfg = json.load(f)
        self.d = self.cfg['d_model']
        self.V = self.cfg['vocab_size']
        self.max_seq_len = self.cfg['max_seq_len']
        self.n_heads = self.cfg['n_heads']
        self.n_lattices = self.cfg['n_lattices']

        self.lib = ctypes.CDLL(str(self.lib_path))
        self.lib.lcm_infer_step.restype = ctypes.c_int
        self.lib.lcm_infer_loop.restype = ctypes.c_int
        self.lib.lcm_get_trace.restype = ctypes.c_int
        self.lib.lcm_get_trace.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int]

        print(f"[INFER] Loading checkpoint from {checkpoint_dir}...")
        self.encoder = load_encoder(
            checkpoint_dir, self.d, self.cfg['n_encoder_layers'],
            self.V, self.cfg['d_ff'], self.max_seq_len)
        self.decoder = load_decoder(checkpoint_dir, self.d, self.V)
        self._load_codebooks()

        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(
            self.checkpoint_dir / "tokenizer.json"))
        self.tokenizer.enable_padding()

        # ── Observability (black box: raw recording) ──
        from train.observability import ObservabilityRecorder
        self.obs = ObservabilityRecorder(
            raw_capacity=raw_capacity, record_every_n=obs_every_n)

        # ── Narrative memory (importance filter + long-term storage) ──
        from train.narrative_memory import NarrativeMemory
        self.narr = NarrativeMemory(max_records=lt_max_records) if lt_enabled else None

        lt_status = f"on (max={lt_max_records})" if lt_enabled else "off"
        print(f"[OBS]  Black box: raw={raw_capacity} every_n={obs_every_n}")
        print(f"[NARR] Narrative memory: {lt_status}")

        # ── Predictive cache (implicit inference reuse) ──
        self._pred_cache_capacity = getattr(raw_capacity, 'pred_cache_capacity', 2048)
        self._pred_enabled = False
        self._last_sig = 0
        self.pred = None  # PredictiveSystem — lazy init via enable_prediction()

        # ── Causal subject (self-awareness and agency) ──
        self._causal_enabled = False
        self.causal = None  # CausalSubject — lazy init via enable_causal()
        self._last_agency_mod = {'tension': 0.0, 'explore': 0.0}

    def _load_codebooks(self):
        ckpt = self.checkpoint_dir
        hrq_data, _, _ = load_codebook_for_c(ckpt, "hrq_codebook.bin")
        n_hrq = self.cfg.get('n_hrq_layers', 3)
        M_top = self.cfg.get('M_top', 512)
        M_fine = self.cfg.get('M_fine', 256)
        self.hrq_M = M_top + n_hrq * M_fine
        self.hrq_C = hrq_data

        sparse_data, _, _ = load_codebook_for_c(ckpt, "sparse_codebook.bin")
        self.sparse_M = self.cfg.get('M_sparse', 512)
        self.sparse_C = np.ascontiguousarray(
            sparse_data[:self.sparse_M * self.d], dtype=np.float32)

        lr_flat = np.fromfile(ckpt / "lowrank_codebook.bin", dtype=np.float32)
        M_lr = self.cfg.get('M_lr', 256)
        ranks = self.cfg.get('ranks', [2, 4, 8])
        self.lr_M = M_lr
        pos, U_list = 0, []
        for r_k in ranks:
            U_list.append(lr_flat[pos:pos + M_lr * r_k].reshape(M_lr, r_k))
            pos += M_lr * r_k
        self.lr_C = np.ascontiguousarray(
            U_list[0] @ lr_flat[pos:].reshape(self.d, -1)[:, :ranks[0]].T,
            dtype=np.float32)

        man_data, self.man_M, _ = load_codebook_for_c(ckpt, "manifold_codebook.bin")
        self.man_C = np.ascontiguousarray(man_data[:self.man_M * self.d], dtype=np.float32)
        t_dim = self.cfg.get('t_dim', 4)
        self.man_T = np.ascontiguousarray(man_data[self.man_M * self.d:], dtype=np.float32)
        self.man_t_dim = t_dim

        bind_flat = np.fromfile(ckpt / "bind_codebook.bin", dtype=np.float32)
        M_bind = self.cfg.get('M_bind', 512)
        self.bind_M = M_bind
        offset = 2 * M_bind * self.d
        self.bind_C = np.ascontiguousarray(
            bind_flat[offset:offset + M_bind * self.d], dtype=np.float32)

        contrast_flat = np.fromfile(ckpt / "contrast_codebook.bin", dtype=np.float32)
        M_contrast = self.cfg.get('M_contrast', 512)
        self.contrast_M = M_contrast
        self.contrast_C = np.ascontiguousarray(
            contrast_flat[:M_contrast * self.d], dtype=np.float32)

        gv_data, gv_n, _ = load_codebook_for_c(ckpt, "gvalue_codebook.bin")
        self.gv_n = gv_n
        self.gv_pos = np.ascontiguousarray(gv_data[:gv_n * self.d], dtype=np.float32)
        self.gv_neg = np.ascontiguousarray(gv_data[gv_n * self.d:], dtype=np.float32)

        danger_data, danger_M, _ = load_codebook_for_c(ckpt, "danger_codebook.bin")
        expected_half = danger_M * self.d
        if len(danger_data) >= 2 * expected_half:
            self.danger_t = np.ascontiguousarray(danger_data[:expected_half], dtype=np.float32)
            self.danger_n = np.ascontiguousarray(danger_data[expected_half:2*expected_half], dtype=np.float32)
            self.danger_M = danger_M
        else:
            self.danger_t = np.zeros(1, dtype=np.float32)
            self.danger_n = np.zeros(1, dtype=np.float32)
            self.danger_M = 0

    def _ptr(self, arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def _normalise_codebook_scale(self, z):
        """Scale z to match codebook entry norms so distance ≈ semantics."""
        hrq = self.hrq_C.reshape(-1, self.d)
        cb_mean_norm = float(np.sqrt(np.mean(np.sum(hrq ** 2, axis=-1))))
        z_norm = float(np.linalg.norm(z))
        if z_norm > 1e-8:
            z = z * (cb_mean_norm / z_norm)
        return z

    def cognitive_step(self, z):
        z = self._normalise_codebook_scale(np.ascontiguousarray(z, dtype=np.float32))
        z_out = np.zeros(self.d, dtype=np.float32)
        d = ctypes.c_int(self.d)
        ret = self.lib.lcm_infer_step(
            self._ptr(z), d,
            self._ptr(self.hrq_C),    ctypes.c_int(self.hrq_M),
            self._ptr(self.sparse_C), ctypes.c_int(self.sparse_M),
            self._ptr(self.lr_C),     ctypes.c_int(self.lr_M),
            self._ptr(self.man_C),    ctypes.c_int(self.man_M),
            self._ptr(self.man_T),    ctypes.c_int(self.man_t_dim),
            self._ptr(self.bind_C),   ctypes.c_int(self.bind_M),
            self._ptr(self.contrast_C), ctypes.c_int(self.contrast_M),
            self._ptr(self.gv_pos),   ctypes.c_int(self.gv_n),
            self._ptr(self.gv_neg),
            ctypes.c_int(self.n_lattices),
            self._ptr(z_out),
        )
        if ret != 0:
            print(f"[INFER] WARNING: lcm_infer_step returned {ret}", file=sys.stderr)
        return z_out

    def cognitive_loop(self, z, conv_tol=1e-3, entropy_thresh=0.5,
                       max_steps=32, use_safety=True,
                       agency_tension_mod=0.0, agency_explore_mod=0.0):
        """Cognitive inference loop with agency-modulated convergence.

        Args:
            z: Input vector (d,).
            conv_tol: Base convergence tolerance.
            entropy_thresh: Base entropy threshold.
            max_steps: Base max inference steps.
            use_safety: Whether to use gvalue/danger checking.
            agency_tension_mod: From agency.get_tension_modulator() [-0.5, 0.5].
                High → relax conv_tol (confident, deeper reasoning).
                Low  → tighten conv_tol (cautious, shorter reasoning).
            agency_explore_mod: From agency.get_explore_modulator() [-0.3, 0.3].
                High → allow more steps (exploratory).
                Low  → fewer steps (conservative).
        """
        mod = agency_tension_mod * 0.5
        modulated_tol = conv_tol * (1.0 - mod)
        modulated_tol = float(np.clip(modulated_tol, conv_tol * 0.5, conv_tol * 2.0))

        explore_m = agency_explore_mod * 0.3
        modulated_max = max_steps + int(round(explore_m * max_steps))
        modulated_max = int(np.clip(modulated_max, 4, 64))

        z = self._normalise_codebook_scale(np.ascontiguousarray(z, dtype=np.float32))
        z_out = np.zeros(self.d, dtype=np.float32)
        d = ctypes.c_int(self.d)
        gv_n = self.gv_n if use_safety else 0
        dm = self.danger_M if use_safety else 0
        ret = self.lib.lcm_infer_loop(
            self._ptr(z), d,
            self._ptr(self.hrq_C),    ctypes.c_int(self.hrq_M),
            self._ptr(self.sparse_C), ctypes.c_int(self.sparse_M),
            self._ptr(self.lr_C),     ctypes.c_int(self.lr_M),
            self._ptr(self.man_C),    ctypes.c_int(self.man_M),
            self._ptr(self.man_T),    ctypes.c_int(self.man_t_dim),
            self._ptr(self.bind_C),   ctypes.c_int(self.bind_M),
            self._ptr(self.contrast_C), ctypes.c_int(self.contrast_M),
            self._ptr(self.gv_pos),   ctypes.c_int(gv_n),
            self._ptr(self.gv_neg),
            self._ptr(self.danger_t), ctypes.c_int(dm),
            self._ptr(self.danger_n),
            ctypes.c_int(self.n_lattices),
            ctypes.c_float(modulated_tol), ctypes.c_float(entropy_thresh),
            ctypes.c_int(modulated_max),
            self._ptr(z_out),
        )
        converged = (ret == 0)
        if not converged:
            print(f"[INFER] NOTE: cognitive loop aborted (ret={ret})", file=sys.stderr)
        return z_out, converged

    LATTICE_NAMES = ['HRQ', 'Sparse', 'LowRank', 'Manifold', 'Binding', 'Contrast', 'Self']

    def get_trace(self):
        d = self.d
        per_step = 7 + 7 + d + 2
        n = self.lib.lcm_get_trace(None, 0)
        if n <= 0:
            return None
        buf = (ctypes.c_float * (n * per_step))()
        self.lib.lcm_get_trace(buf, n * per_step)
        arr = np.frombuffer(buf, dtype=np.float32).reshape(n, per_step)
        traces = []
        for s in range(n):
            traces.append({
                'step': int(arr[s, 14 + d]),
                'weights': arr[s, :7].copy(),
                'confidences': arr[s, 7:14].copy(),
                'z_next': arr[s, 14:14 + d].copy(),
                'has_conflict': bool(arr[s, 14 + d + 1]),
            })
        return traces

    def print_trace(self, traces=None):
        if traces is None:
            traces = self.get_trace()
        if not traces:
            print("[TRACE] No trace data available.")
            return
        lat_names = [n[:4] for n in self.LATTICE_NAMES]
        hdr = f"{'Step':>4} | {'|'.join(f'{n:>6}' for n in lat_names)} | Z_norm | Conflict"
        print("[TRACE] Per-step trace:")
        print("-" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for t in traces:
            wstr = '|'.join(f'{w:>6.3f}' for w in t['weights'])
            print(f"{t['step']:>4} | {wstr} | {np.linalg.norm(t['z_next']):>7.4f} | "
                  f"{'YES' if t['has_conflict'] else 'no':>8}")
        print("-" * len(hdr))

    def enable_prediction(self, cache_capacity=2048, reflect_interval=1000):
        """Enable the prediction cache system.

        Args:
            cache_capacity: Cache capacity (entries).
            reflect_interval: Intrinsic motivation reflection interval (steps).
        """
        from train.predictive_cache import PredictiveSystem
        self.pred = PredictiveSystem(
            d_model=self.d,
            cache_capacity=cache_capacity,
            reflect_interval=reflect_interval,
        )
        self._pred_enabled = True
        self._last_sig = 0
        print(f"[PRED] Prediction cache: cache={cache_capacity}  "
              f"reflect_every={reflect_interval}")

    def _print_pred_stats(self):
        if self.pred is None:
            return
        stats = self.pred.get_stats()
        m = stats['matcher']
        print(f"\n{'=' * 46}")
        print(f"  Prediction Cache")
        print(f"{'=' * 46}")
        print(f"  Cache:   {stats['n_cache']}/{self.pred.cache.capacity}")
        print(f"  ε:       {m['epsilon']:.3f}")
        print(f"  MAB Q:   {'  '.join(f'{k}={v:.3f}' for k, v in m['op_q'].items())}")
        print(f"  MAB best: {'  '.join(f'{k}={v:.3f}' for k, v in m['op_best'].items())}")
        print(f"  Proxy:   {'on' if stats['motivation']['proxy_enabled'] else 'off'}")
        print(f"  Logged:  {stats['motivation']['n_logged']}")
        print(f"{'=' * 46}\n")

    def enable_causal(self, enable_counterfactual=True):
        """Enable causal subject."""
        from train.causal_subject import CausalSubject
        self.causal = CausalSubject(
            d_model=self.d,
            enable_counterfactual=enable_counterfactual and self._pred_enabled,
            enable_boundary=True,
        )
        self._causal_enabled = True
        print(f"[CAUSAL] Causal subject: counterfactual={enable_counterfactual and self._pred_enabled}")

    def _print_causal_stats(self):
        if self.causal is None:
            print("[CAUSAL] Causal subject not enabled")
            return
        stats = self.causal.get_full_stats()
        print(f"\n{'=' * 46}")
        print(f"  Causal Subject")
        print(f"{'=' * 46}")
        print(f"  Agency:     {stats['agency']['current']:.3f} "
              f"(baseline={stats['agency']['baseline']:.3f})")
        print(f"  Causal Edges: {stats['graph']['n_edges']} "
              f"({stats['graph']['n_internal']} int / {stats['graph']['n_external']} ext)")
        print(f"  Boundary:   {stats['boundary']['n_extended']} targets "
              f"{stats['boundary']['extended']}")
        cf = stats.get('counterfactual', {})
        if cf.get('enabled'):
            print(f"  Counterfact Δ: {cf['avg_delta']:.4f}")
        print(f"  Action Table: {stats['action_table']['n_entries']} entries")
        print(f"{'=' * 46}\n")

    def print_architecture():
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            LCM Cognitive Inference Engine                   ║
    ║        Zero-Parameter Dynamic Dataflow Graph                ║
    ╚══════════════════════════════════════════════════════════════╝
                            Input z (from encoder)
                                  |
                    ┌─────────────┼─────────────┐
                    │             │             │
              Retrieval      Manifold      Contrast
              (HRQ/Sparse/   (Poincaré     (Poincaré
               LowRank/      slide+tanh)    dist wgt)
               Binding)
                    │             │             │
                    └──────┬──────┴──────┬──────┘
                           │             │
                     HRR Bind       Self Check
                     (circular      (identity
                      conv)          passthrough)
                     HRR Unbind
                           │             │
                           └──────┬──────┘
                                  │
                              Fusion
                         (distance-weighted
                          + optional value bias)
                              │
                            Safety
                      (Danger+GValue+Consist.)
                              │
                         Convergence
                      (z diff + entropy)
                              │
                          z_q output → gen_head → sample
        """)

    def generate(self, prompt, max_new=128, temperature=0.7, top_k=50,
                 bos_token=2, eos_token=3, use_loop=True, show_trace=False):
        encoding = self.tokenizer.encode(prompt)
        token_ids = encoding.ids
        if not token_ids or token_ids[0] != bos_token:
            token_ids = [bos_token] + token_ids
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[-self.max_seq_len:]

        # Pre-encode past sequence once for efficiency
        token_ids = list(token_ids)
        kv_cache = None

        # Incremental encoder state (recurrent cumsum cache)
        enc_state = None

        for gen_step in range(max_new):
            # Source tagging: first step is external (user prompt), subsequent are internal
            source = 0 if gen_step == 0 else 1

            x = np.array(token_ids[-self.max_seq_len:], dtype=np.int32)

            # Incremental encoder: full encode on first step, recurrent O(d²) update thereafter.
            # Full re-encode every ENC_RESET_INTERVAL steps to bound cumsum drift in float32.
            if enc_state is None:
                z, enc_state = _encoder_full_with_state(self.encoder, x, self.n_heads)
            elif gen_step % ENC_RESET_INTERVAL == 0:
                x_full = np.array(token_ids[-self.max_seq_len:], dtype=np.int32)
                z, enc_state = _encoder_full_with_state(self.encoder, x_full, self.n_heads)
            else:
                z = encoder_recurrent_step_cy(self.encoder, enc_state, token_ids[-1], self.n_heads)
            z_cur = z.copy()

            # ── Prediction cache query (before inference) ──
            z_pred = None
            pred_op = ''
            pred_param = None
            if self._pred_enabled and self.pred is not None:
                z_pred, pred_op, pred_param = self.pred.matcher.predict(
                    z_cur, self._last_sig)

            t0 = time.time()
            if use_loop:
                z_q, converged = self.cognitive_loop(
                    z, use_safety=False,
                    agency_tension_mod=self._last_agency_mod['tension'],
                    agency_explore_mod=self._last_agency_mod['explore'])
                if show_trace:
                    tqdm.write(f"\n[TOKEN {gen_step}] loop converged={converged}:")
                    self.print_trace()
            else:
                z_q = self.cognitive_step(z)
                converged = True
            step_ms = (time.time() - t0) * 1000

            # ── Pack signature from trace ──
            traces = self.get_trace()
            if traces:
                last_t = traces[-1]
                from train.predictive_cache import pack_trace_sig
                sig = pack_trace_sig(
                    last_t.get('weights', np.zeros(7)),
                    last_t.get('confidences', np.zeros(7)),
                    last_t.get('has_conflict', False))
            else:
                sig = 0

            # ── Update prediction cache & motivation ──
            if self._pred_enabled and self.pred is not None:
                self.pred.cache.append(sig, z_cur, z_q, gen_step)
                if z_pred is not None:
                    self.pred.matcher.update(z_pred, z_q)
                    self.pred.motivation.log(
                        op=pred_op, param=pred_param,
                        z_pred=z_pred, z_actual=z_q, step=gen_step)
                else:
                    self.pred.motivation.log(
                        op='', param=None,
                        z_pred=z_cur, z_actual=z_q, step=gen_step)
                self.pred.motivation.reflect(gen_step, self.pred.matcher)

            self._last_sig = sig

            if self.decoder['format'] == 'old':
                logits = gen_head_forward_old(self.decoder, z_q, x)
                kv_cache = None
            else:
                logits, kv_cache = gen_head_new_single_cy(self.decoder, z_q, x, kv_cache)

            next_id = sample_categorical(logits, temperature, top_k)

            # ── Record to black box (self-observation) ──
            record = None
            if traces:
                last_t = traces[-1]
                record = self.obs.record_step(
                    step=gen_step,
                    source=source,
                    step_time_ms=step_ms,
                    soft_mask=last_t.get('weights'),
                    route_idx=np.argmax(last_t.get('weights', [0])).item()
                        if last_t.get('weights') is not None else None,
                    is_safe=not last_t.get('has_conflict', False),
                    convergence_diff=0.0 if converged else None,
                    dag_nodes=traces,
                    conflict_source=None if not last_t.get('has_conflict')
                        else trace_step_conflict_source(last_t),
                    z_q=z_q,
                )

            # ── Feed to narrative memory ──
            if record is not None and self.narr is not None:
                self.narr.feed(record)

            # ── Feed to causal subject ──
            if record is not None and self._causal_enabled and self.causal is not None:
                try:
                    action_desc = f"gen_step_{gen_step}"
                    causal_info = self.causal.step(
                        record, pred_cache=self.pred.cache if self.pred else None,
                        action_desc=action_desc)
                    # Store agency modulators for next step
                    self._last_agency_mod['tension'] = causal_info.get(
                        'agency_modulator_tension', 0.0)
                    self._last_agency_mod['explore'] = causal_info.get(
                        'agency_modulator_explore', 0.0)
                except Exception as e:
                    print(f"[CAUSAL] Feed error: {e}")

            yield next_id
            token_ids.append(next_id)
            if next_id == eos_token:
                break

    def generate_text(self, prompt, **kwargs):
        return self.tokenizer.decode(
            list(self.generate(prompt, **kwargs)), skip_special_tokens=True)


def _build_cython(d_model=256):
    """Compile Cython .pyx + C inference engine.

    Equivalent to:
        python setup.py build_ext --inplace
        cd infer && make LCM_D=<d_model>

    Args:
        d_model: Latent dimension, passed as LCM_D to C engine Makefile.
    """
    try:
        from Cython.Build import cythonize
        from setuptools import setup
        import numpy as np
    except ImportError:
        print("Error: Cython or setuptools not installed.")
        print("  pip install cython setuptools")
        sys.exit(1)

    print("[1/2] Compiling Cython extensions...")
    setup(
        name="lcm_cy",
        ext_modules=cythonize(
            ["train/_lcm_cy.pyx", "train/_metrics_cy.pyx"],
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            },
        ),
        include_dirs=[np.get_include()],
        script_args=["build_ext", "--inplace"],
    )

    import os as _os
    # LCM_D override via env var; default to d_model argument
    lcm_d = _os.environ.get("LCM_D", str(d_model))
    infer_dir = _os.path.join(_os.path.dirname(__file__) or ".", "infer")
    print(f"[2/2] Building C inference engine (LCM_D={lcm_d})...")
    import subprocess as _sp
    ret = _sp.run(["make", f"LCM_D={lcm_d}", "-C", infer_dir])
    if ret.returncode != 0:
        print(f"Warning: C engine build returned {ret.returncode}")
    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description="LCM — Latent Codebook Model CLI")

    # ── top-level flags ─────────────────────────────────────────────────
    p.add_argument("-d", "--data", default=None,
                   help="Training: path to tokenised .dat mmap file")
    p.add_argument("-i", "--interact", default=None,
                   help="Interactive: path to checkpoint directory")
    p.add_argument("-M", "--max-seq-len", type=int, default=512,
                   help="Maximum sequence length")

    # ── training options ────────────────────────────────────────────────
    p.add_argument("-b", "--batch-size", type=int, default=16,
                   help="Batch size (default 16)")
    p.add_argument("-s", "--seq-len", type=int, default=512,
                   help="Sequence length per sample (default 512)")
    p.add_argument("-dm", "--d-model", type=int, default=256,
                   help="Latent dimension (default 256)")
    p.add_argument("-f", "--d-ff", type=int, default=None,
                   help="FFN hidden dimension (default: 1.5 × d_model)")
    p.add_argument("-n", "--n-heads", type=int, default=4,
                   help="Attention heads (default 4)")
    p.add_argument("-l", "--lr", type=float, default=None,
                   help="Learning rate (default 3e-4)")
    p.add_argument("-S", "--steps", type=int, default=100000,
                   help="Total training steps (default 100000)")
    p.add_argument("-c", "--save", type=int, default=1000,
                   help="Save checkpoint every N steps (default 1000)")
    p.add_argument("-o", "--save-dir", default=None,
                   help="Checkpoint save directory (default: ./checkpoints)")
    p.add_argument("-r", "--resume", default=None,
                   help="Resume from checkpoint directory, e.g. checkpoints/step_01000")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Auto-confirm prompts (use latest checkpoint)")
    p.add_argument("--auto", action="store_true",
                   help="Auto mode: monitor loss, auto-recover NaN/crashes, save best checkpoint")
    p.add_argument("-p", "--shape", default=None,
                   help="Path to shape JSON (auto-derived from -d if omitted)")

    # ── stage-based training (1/2/3) ────────────────────────────────────
    p.add_argument("-g", "--stage", type=int, default=None, choices=[1, 2, 3],
                   help="Training stage. 1=train_model (decoder LM via train_lm.py), 2=train_memory (encoder+codebooks via train_memory.py), 3=joint finetune")
    p.add_argument("-L", "--load-lm", default=None,
                   help="Load trained gen_head from LM checkpoint (.pkl), used by --stage 2")
    p.add_argument("-2", "--lr-stage2", type=float, default=1e-4,
                   help="Stage 2 lr (default 1e-4)")
    p.add_argument("-3", "--lr-stage3", type=float, default=1e-4,
                   help="Stage 3 lr (default 1e-4)")
    p.add_argument("-m", "--memory-steps", type=int, default=20000,
                   help="Stage 2 steps (default 20000)")

    # ── interactive options (C inference engine) ───────────────────────────
    p.add_argument("-N", "--max-new", type=int, default=128,
                   help="Max tokens to generate per turn (default 128)")
    p.add_argument("-T", "--temp", type=float, dest="temperature",
                   default=0.7, help="Sampling temperature (default 0.7)")
    p.add_argument("-t", "--trace", action="store_true",
                   help="Show cognitive trace per token")
    p.add_argument("--loop", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    p.add_argument("-A", "--show-arch", action="store_true",
                   help="Print architecture diagram and exit")

    # ── observability ─────────────────────────────────────────────────────
    p.add_argument("--obs", action="store_true",
                   help="Enable observability recorder (dual-layer narrative memory)")
    p.add_argument("--raw-capacity", type=int, default=1000,
                   help="Raw trace ring buffer capacity (default 1000)")
    p.add_argument("--obs-every-n", type=int, default=1,
                   help="Record every N steps (default 1 = all)")
    p.add_argument("--lt", dest="lt_enabled", action="store_true",
                   default=True, help="Enable long-term narrative")
    p.add_argument("--no-lt", dest="lt_enabled", action="store_false",
                   help="Disable long-term narrative")
    p.add_argument("--lt-max", type=int, default=10000, dest='lt_max_records',
                   help="Max long-term records (default 10000)")
    p.add_argument("--obs-summary", action="store_true",
                   help="Print observability summary after generation")
    p.add_argument("--obs-export", default=None,
                   help="Export observability data to JSON file")
    p.add_argument("--narr-consolidate", action="store_true",
                   help="Run model-driven narrative consolidation after each turn")
    p.add_argument("--narr-keep-threshold", type=float, default=0.05,
                   help="Narrative keep threshold (default 0.05, drop entries below this)")

    # ── prediction cache ──
    p.add_argument("--pred", action="store_true",
                   help="Enable prediction cache (implicit inference reuse)")
    p.add_argument("--pred-cache", type=int, default=2048,
                   help="Prediction cache capacity (default 2048)")
    p.add_argument("--pred-reflect", type=int, default=1000,
                   help="Intrinsic motivation reflect interval (default 1000)")

    # ── causal subject ──
    p.add_argument("--causal", action="store_true",
                   help="Enable causal subject (self-awareness and agency)")

    # ── active output ──
    p.add_argument("--active", action="store_true",
                   help="Active output mode: model can initiate conversation via internal drive")
    p.add_argument("--active-interval", type=float, default=0.3,
                   help="Seconds between internal cognitive ticks (default 0.3)")
    p.add_argument("--active-tension", type=float, default=0.15,
                   help="Tension threshold for active output (default 0.15)")
    p.add_argument("--active-surprise", type=float, default=0.3,
                   help="Prediction error threshold for active output (default 0.3)")
    p.add_argument("--active-max-burst", type=int, default=32,
                   help="Max tokens per active output burst (default 32)")

    # ── cognitive training ──
    p.add_argument("-C", "--cog-train", action="store_true", help="Cognitive training")
    p.add_argument("-G", "--cog-steps", type=int, default=50000, help="Cog steps (default 50000)")
    p.add_argument("-R", "--cog-lr", type=float, default=3e-4, help="Cog lr (default 3e-4)")
    p.add_argument("-J", "--cog-batch", type=int, default=4, help="Cog batch (default 4)")
    p.add_argument("-Q", "--cog-seq", type=int, default=256, help="Cog seq len (default 256)")
    p.add_argument("-V", "--cog-save", type=int, default=1000, help="Cog save interval (default 1000)")
    p.add_argument("-F", "--from-lm-ckpt", default=None, help="Load Stage 1 gen_head")

    # ── language LCM training ──
    p.add_argument("--lang-train", action="store_true", help="Language LCM training (Stage 1)")
    p.add_argument("--lang-infer", default=None, metavar="CKPT",
                   help="Language LCM inference checkpoint path")
    p.add_argument("--lang-steps", type=int, default=100000, help="Lang LCM steps (default 100000)")
    p.add_argument("--lang-lr", type=float, default=3e-4, help="Lang LCM lr (default 3e-4)")
    p.add_argument("--lang-batch", type=int, default=16, help="Lang LCM batch (default 16)")
    p.add_argument("--lang-seq", type=int, default=512, help="Lang LCM seq len (default 512)")
    p.add_argument("--lang-save", type=int, default=10000, help="Lang LCM save interval (default 10000)")
    p.add_argument("--from-lang-ckpt", default=None, help="Load Stage 1 Language LCM for cog training")
    p.add_argument("--use-qwen", action="store_true",
                   help="Use frozen Qwen2.5-0.5B as active channel (auto-loads weights)")
    p.add_argument("--compile", action="store_true",
                   help="Pre-compile training graph (run once before --cog-train)")
    p.add_argument("--cache-dir", default="/root/autodl-tmp/jax_cache",
                   help="JAX persistent compilation cache dir (default /root/autodl-tmp/jax_cache)")
    p.add_argument("--auto-batch", action="store_true",
                   help="Auto-calc optimal batch/seq based on GPU memory")
    p.add_argument("--prompt", default=None, help="Prompt text for --lang-infer")
    # ── subcommands ─────────────────────────────────────────────────────
    sub = p.add_subparsers(dest="mode")

    pp = sub.add_parser("preprocess", help="Tokenise text → uint16 mmap")
    pp.add_argument("-i", "--input", required=True, help="Input .txt file")
    pp.add_argument("-t", "--tokenizer", default="data/tokenizer.json",
                    help="Tokenizer JSON path (auto-trained if missing)")
    pp.add_argument("-o", "--output", required=True, help="Output .dat mmap path")
    pp.add_argument("-v", "--vocab-size", type=int, default=0,
                    help="Train new tokenizer with this vocab size")

    pc = sub.add_parser("clean", help="5-stage data cleaning with NeMo Curator")
    pc.add_argument("-i", "--input", required=True, help="Input JSONL (from WikiExtractor)")
    pc.add_argument("-o", "--output", required=True, help="Output JSONL path")
    pc.add_argument("-d", "--output-dir", default=None, help="Cache/output directory")
    pc.add_argument("-m", "--langid-model", default="lid.176.bin",
                    help="FastText language ID model path")
    pc.add_argument("-c", "--langid-threshold", type=float, default=0.3,
                    help="Min language ID confidence (default 0.3)")
    pc.add_argument("-w", "--min-words", type=int, default=50,
                    help="Min word count to keep (default 50)")
    pc.add_argument("-e", "--skip-exact-dedup", action="store_true",
                    help="Skip exact deduplication stage")
    pc.add_argument("-f", "--skip-fuzzy-dedup", action="store_true",
                    help="Skip fuzzy deduplication stage")
    pc.add_argument("--num-hashes", type=int, default=256,
                    help="MinHash num_hashes (default 256)")
    pc.add_argument("-u", "--num-buckets", type=int, default=16,
                    help="MinHash num_buckets (default 16)")
    pc.add_argument("--min-ngram", type=int, default=5,
                    help="Min ngram size for fuzzy dedup (default 5)")
    pc.add_argument("--fuzzy-threshold", type=float, default=0.7,
                    help="Jaccard similarity threshold for fuzzy dedup (default 0.7)")

    # ── unified training subcommand ──
    ptr = sub.add_parser("train", help="Unified training entry")
    ptr.add_argument("--stage", required=True, choices=["1","2","3","cog"], help="Training stage")
    ptr.add_argument("-d", "--data", help="Path to tokenized .dat file")
    ptr.add_argument("-b", "--batch-size", type=int, default=16)
    ptr.add_argument("-s", "--seq-len", type=int, default=512)
    ptr.add_argument("-S", "--steps", type=int, default=100000)
    ptr.add_argument("-l", "--lr", type=float, default=None)
    ptr.add_argument("-o", "--save-dir", default=None)
    ptr.add_argument("-c", "--save", type=int, default=1000)
    ptr.add_argument("-r", "--resume", default=None)
    ptr.add_argument("-L", "--load-lm", default=None)
    ptr.add_argument("-2", "--lr-stage2", type=float, default=1e-4)
    ptr.add_argument("-3", "--lr-stage3", type=float, default=1e-4)
    ptr.add_argument("-m", "--memory-steps", type=int, default=20000)
    ptr.add_argument("-y", "--yes", action="store_true")
    ptr.add_argument("--auto", action="store_true", help="Auto mode: monitor + auto-recover")
    ptr.add_argument("-F", "--from-lm-ckpt", default=None)
    ptr.add_argument("-C", "--cog-steps", type=int, default=50000)
    ptr.add_argument("-R", "--cog-lr", type=float, default=3e-4)
    ptr.add_argument("-J", "--cog-batch", type=int, default=4)
    ptr.add_argument("-Q", "--cog-seq", type=int, default=256)
    ptr.add_argument("-V", "--cog-save", type=int, default=1000)
    ptr.add_argument("--d-model", type=int, default=256)
    ptr.add_argument("-f", "--d-ff", type=int, default=None)
    ptr.add_argument("-n", "--n-heads", type=int, default=4)

    # ── checkpoint subcommands ──
    pck = sub.add_parser("ckpt", help="Manage checkpoints")
    pck_sub = pck.add_subparsers(dest="ckpt_action")
    pck_ls = pck_sub.add_parser("list", help="List checkpoints")
    pck_ls.add_argument("dir", nargs="?", default="checkpoints",
                        help="Checkpoint directory (default: checkpoints/)")
    pck_ls.add_argument("-l", dest="long", action="store_true",
                        help="Detailed listing (config, sizes)")
    pck_pr = pck_sub.add_parser("prune", help="Prune old checkpoints, keep N latest")
    pck_pr.add_argument("dir", nargs="?", default="checkpoints",
                        help="Checkpoint directory (default: checkpoints/)")
    pck_pr.add_argument("-k", type=int, default=5,
                        help="Keep latest N checkpoints (default 5)")
    pck_df = pck_sub.add_parser("diff", help="Compare two checkpoints")
    pck_df.add_argument("dir1", help="First checkpoint directory")
    pck_df.add_argument("dir2", help="Second checkpoint directory")

    # ── eval subcommands ──
    pev = sub.add_parser("eval", help="Evaluate model")
    pev_sub = pev.add_subparsers(dest="eval_action")
    pev_ppl = pev_sub.add_parser("ppl", help="Compute perplexity on data")
    pev_ppl.add_argument("checkpoint", help="Checkpoint directory")
    pev_ppl.add_argument("data", help="Path to .dat file")
    pev_ppl.add_argument("--batches", type=int, default=10, help="Number of batches (default 10)")
    pev_ppl.add_argument("--shape", default=None, help="Path to shape JSON")
    pev_tr = pev_sub.add_parser("trace", help="Run cognitive loop trace")
    pev_tr.add_argument("checkpoint", help="Checkpoint directory")
    pev_tr.add_argument("--prompt", default="Hello", help="Prompt text (default: Hello)")
    pev_tr.add_argument("--tokens", type=int, default=5, help="Tokens to trace (default 5)")

    # ── report subcommand ──
    prp = sub.add_parser("report", help="Model architecture report")
    prp.add_argument("checkpoint", help="Checkpoint directory")

    # ── serve subcommand ──
    ps = sub.add_parser("serve", help="Start HTTP inference server (OpenAI-compatible)")
    ps.add_argument("checkpoint", help="Checkpoint directory")
    ps.add_argument("--port", type=int, default=8080, help="Port (default 8080)")
    ps.add_argument("--host", default="0.0.0.0", help="Host (default 0.0.0.0)")

    # ── batch subcommand ──
    pb = sub.add_parser("batch", help="Batch generate from JSONL")
    pb.add_argument("checkpoint", help="Checkpoint directory")
    pb.add_argument("-i", "--input", required=True, help="Input JSONL file")
    pb.add_argument("-o", "--output", required=True, help="Output JSONL file")

    # ── chat subcommand ──
    pch = sub.add_parser("chat", help="Interactive chat REPL")
    pch.add_argument("checkpoint", help="Checkpoint directory")
    pch.add_argument("--max-new", type=int, default=128)
    pch.add_argument("--temp", type=float, default=0.7)

    # ── data subcommands ──
    pds = sub.add_parser("data", help="Data inspection tools")
    pds_sub = pds.add_subparsers(dest="data_action")
    pds_st = pds_sub.add_parser("stats", help="Token data statistics")
    pds_st.add_argument("data", help="Path to .dat file")
    pds_st.add_argument("--shape", default=None, help="Path to shape JSON")
    pds_st.add_argument("--top-k", type=int, default=20, help="Top-K tokens (default 20)")
    pds_st.add_argument("-t", "--tokenizer", default="data/tokenizer.json")
    pds_sm = pds_sub.add_parser("sample", help="Sample decoded text from data")
    pds_sm.add_argument("data", help="Path to .dat file")
    pds_sm.add_argument("--shape", default=None, help="Path to shape JSON")
    pds_sm.add_argument("-t", "--tokenizer", default="data/tokenizer.json")
    pds_sm.add_argument("-n", "--count", type=int, default=5, help="Number of samples (default 5)")
    pds_sm.add_argument("-l", "--length", type=int, default=100, help="Tokens per sample (default 100)")
    pds_sm.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")

    # ── chart subcommand ──
    pch = sub.add_parser("chart", help="Generate training metrics HTML chart from saved JSON")
    pch.add_argument("-i", "--input", default="checkpoints/metrics.json",
                     help="Path to metrics.json (default: checkpoints/metrics.json)")
    pch.add_argument("-o", "--output", default="metrics.html",
                     help="Output HTML path (default: metrics.html)")

    # ── export subcommand ──
    pe = sub.add_parser("export", help="Export cog_train checkpoint → C inference engine format")
    pe.add_argument("ckpt_dir", help="Path to cog_train checkpoint directory (e.g. checkpoints/cog/step_010000)")
    pe.add_argument("-o", "--out", default=None, help="Output directory (default: ckpt_dir + _infer)")
    pe.add_argument("--data-dir", default="data", help="Data directory (for tokenizer.json)")

    # ── build subcommand ──
    pb = sub.add_parser("build", help="Compile Cython acceleration extensions (.pyx → .so)")
    pb.add_argument("--inplace", action="store_true", default=True,
                    help="Build in-place (default: true)")

    args = p.parse_args()

    # ── dispatch ────────────────────────────────────────────────────────
    if args.mode == "preprocess":
        preprocess(args)
    elif args.mode == "clean":
        clean(args)
    elif args.mode == "train":
        _cmd_train(args)
        return
    elif args.mode == "ckpt":
        if args.ckpt_action == "list":
            _cmd_ckpt_list(args.dir, args.long)
        elif args.ckpt_action == "prune":
            _cmd_ckpt_prune(args.dir, args.k)
        elif args.ckpt_action == "diff":
            from train.cli_extras import cmd_ckpt_diff
            cmd_ckpt_diff(args)
        return
    elif args.mode == "eval":
        if args.eval_action == "ppl":
            from train.cli_extras import cmd_eval_ppl
            cmd_eval_ppl(args)
        elif args.eval_action == "trace":
            from train.cli_extras import cmd_eval_trace
            cmd_eval_trace(args)
        return
    elif args.mode == "report":
        from train.cli_extras import cmd_report
        cmd_report(args)
        return
    elif args.mode == "serve":
        from train.cli_extras import cmd_serve
        cmd_serve(args)
        return
    elif args.mode == "batch":
        from train.cli_extras import cmd_batch
        cmd_batch(args)
        return
    elif args.mode == "chat":
        from train.cli_extras import cmd_chat
        cmd_chat(args)
        return
    elif args.mode == "data":
        from train.cli_extras import cmd_data_stats as cd_stats
        from train.cli_extras import cmd_data_sample
        if args.data_action == "stats":
            cd_stats(args)
        elif args.data_action == "sample":
            cmd_data_sample(args)
        return
    elif args.mode == "stats":
        _cmd_data_stats(args.data, args.shape, args.num_samples, args.tokenizer)
        return
    elif args.mode == "chart":
        from train.monitor import make_chart_html
        make_chart_html(args.input, args.output)
        return
    elif args.mode == "build":
        _build_cython(d_model=getattr(args, 'd_model', None) or getattr(args, 'dm', None) or 256)
        return
    elif args.mode == "export":
        from train.export_cog_ckpt import export
        out = args.out or args.ckpt_dir.rstrip("/") + "_infer"
        export(args.ckpt_dir, out, args.data_dir)
        return
    elif args.show_arch:
        LCMInferEngine.print_architecture()

        return
    elif args.compile:
        _require_jax()
        os.environ.setdefault("JAX_PERSISTENT_CACHE_DIR", args.cache_dir)
        os.makedirs(args.cache_dir, exist_ok=True)
        from train.config import LCMConfig
        from train.cog_train import make_train_step, init_cog_params
        import optax
        if (args.auto_batch or args.use_qwen) and args.cog_batch == 4:
            try:
                gpu = jax.devices()[0]
                free_mb = gpu.memory_stats()["bytes_limit"] // (1024*1024)
                print(f"[AUTO] GPU: {free_mb}MB free")
                if args.use_qwen:
                    args.cog_batch = max(2, min(8, (free_mb - 8000) // 2000))
                    args.cog_seq = 128
                print(f"[AUTO] -> B={args.cog_batch}, N={args.cog_seq}")
            except: pass
        B, N = args.cog_batch or 2, args.cog_seq or 128
        print(f"[COMPILE] B={B}, N={N}, cache={args.cache_dir}")
        cfg = LCMConfig()
        rng = jax.random.PRNGKey(42)
        lck = "checkpoints/qwen_model/qwen_params.npz" if args.use_qwen else None
        p, ss = init_cog_params(cfg, jax.random.split(rng)[1], lang_ckpt=lck)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=0.01))
        ts = make_train_step(cfg, opt)
        os = opt.init(p)
        d = (jnp.zeros((B, N), dtype=jnp.int32), jnp.ones((B, N), dtype=jnp.int32))
        import time as tm
        t0 = tm.time()
        r = ts(p, os, d, 3e-4, jax.random.split(rng)[1], ss)
        print(f"[COMPILE] Done in {tm.time()-t0:.0f}s, loss={float(r[2]):.4f}")
        return
    elif args.cog_train:
        _require_jax()
        if args.cache_dir:
            os.environ.setdefault("JAX_PERSISTENT_CACHE_DIR", args.cache_dir)
            os.makedirs(args.cache_dir, exist_ok=True)
        from train.config import LCMConfig
        from train.cog_train import train_cog
        cfg = LCMConfig()
        shape = args.shape or (args.data.replace(".dat", "_shape.json") if args.data else None)
        out_dir = args.save_dir or "checkpoints/cog"
        resume = args.resume
        if not resume:
            resume = _prompt_resume(out_dir, "CogTrain", args.yes)

        # ── Auto-batch (auto if --use-qwen) ──────────────────────────
        if (args.auto_batch or args.use_qwen) and args.cog_batch == 4:
            try:
                gpu = jax.devices()[0]
                free_mb = gpu.memory_stats()["bytes_limit"] // (1024*1024)
                if args.use_qwen:
                    args.cog_batch = max(4, min(16, (free_mb - 6000) // 1500))
                    args.cog_seq = 256
                else:
                    # No Qwen: encoder + codebooks + 8L decoder
                    overhead = 1500
                    per = cfg.d_model * 512 * 8 * 4 // (1024*1024)
                    max_b = max(1, min(128, (free_mb - overhead) // per))
                    args.cog_batch = max_b
                    args.cog_seq = min(1024, 256 * max(1, (free_mb - overhead) // (per * 4)))
                print(f"[AUTO] GPU: {free_mb}MB free → B={args.cog_batch}, N={args.cog_seq}")
            except Exception as e:
                print(f"[AUTO] Probe failed ({e}), using defaults B={args.cog_batch}, N={args.cog_seq}")

        qwen_ckpt = None
        if args.use_qwen:
            qwen_ckpt = "checkpoints/qwen_model/qwen_params.npz"
            if not os.path.exists(qwen_ckpt):
                print(f"[QWEN] Weights not found at {qwen_ckpt}, downloading...")
                from huggingface_hub import hf_hub_download
                qwen_ckpt = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
            print(f"[QWEN] Using frozen Qwen2.5-0.5B as active channel")
        train_cog(
            cfg=cfg,
            output_dir=out_dir,
            steps=args.cog_steps,
            lr=args.cog_lr,
            batch_size=args.cog_batch,
            seq_len=args.cog_seq,
            log_every=100,
            save_every=args.cog_save,
            data_path=args.data,
            shape_path=shape,
            lang_ckpt=qwen_ckpt or args.from_lang_ckpt or args.from_lm_ckpt,
            resume=resume,
            joint=(args.stage == 3),
            auto_mode=args.auto,
        )
    elif args.lang_train:
        _require_jax()
        from train.config import LCMConfig
        from train.train_lang_lcm import train_lang_lcm
        cfg = LCMConfig()
        shape = args.shape or (args.data.replace(".dat", "_shape.json") if args.data else None)
        out_dir = args.save_dir or "checkpoints/lang_lm"
        train_lang_lcm(
            cfg=cfg,
            output_dir=out_dir,
            steps=args.lang_steps,
            lr=args.lang_lr,
            batch_size=args.lang_batch,
            seq_len=args.lang_seq,
            log_every=100,
            save_every=args.lang_save,
            from_ckpt=args.resume,
            data_path=args.data,
            shape_path=shape,
        )
    elif args.lang_infer:
        _require_jax()
        from train.config import LCMConfig
        from train.lang_lcm import init_lang_lcm_params, lang_lcm_generate
        from train.train_lang_lcm import load_checkpoint
        cfg = LCMConfig()
        params, _ = load_checkpoint(args.lang_infer)
        tokenizer = _load_tokenizer()
        prompt_text = args.prompt or "今天天气"
        prompt_ids = tokenizer.encode(prompt_text).ids
        rng = jax.random.PRNGKey(42)
        tokens = lang_lcm_generate(
            params, prompt_ids, max_len=args.max_new or 128,
            bos_id=1, eos_id=2, rng=rng, cfg=cfg)
        output = tokenizer.decode(tokens)
        print(f"\n[LANG] Prompt: {prompt_text}")
        print(f"[LANG] Output: {output}")
    elif args.data:
        train(args)
    elif args.interact:
        interact(args)
    else:
        p.print_help()
        sys.exit(1)


def _cmd_train(subargs):
    """Dispatch for 'python lcm.py train --stage X'."""
    _require_jax()
    stage = subargs.stage
    if stage == "cog":
        from train.config import LCMConfig
        from train.cog_train import train_cog
        cfg = LCMConfig()
        shape = subargs.shape or (subargs.data.replace(".dat", "_shape.json") if subargs.data else None)
        out_dir = subargs.save_dir or "checkpoints/cog"
        resume = subargs.resume
        if not resume:
            resume = _prompt_resume(out_dir, "CogTrain", subargs.yes)
        train_cog(cfg=cfg, output_dir=out_dir, steps=subargs.cog_steps,
                  lr=subargs.cog_lr, batch_size=subargs.cog_batch, seq_len=subargs.cog_seq,
                  log_every=100, save_every=subargs.cog_save, data_path=subargs.data,
                  shape_path=shape, lm_ckpt=subargs.from_lm_ckpt, resume=resume,
                  joint=False, auto_mode=subargs.auto)
    else:
        subargs.stage = int(stage)
        train(subargs)


if __name__ == "__main__":
    main()
