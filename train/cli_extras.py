"""CLI extras for lcm.py — modular subcommand dispatch.

All functions take a single 'args' namespace argument and print to stdout.
Lazy imports throughout — nothing is imported at module level.
"""

import sys
import os
import json

__all__ = [
    "cmd_train",
    "cmd_ckpt_diff",
    "cmd_eval_ppl",
    "cmd_eval_trace",
    "cmd_report",
    "cmd_serve",
    "cmd_batch",
    "cmd_chat",
    "cmd_data_sample",
    "cmd_data_stats",
]


def cmd_train(args):
    """Unified training entry.  args.stage is '1', '2', '3', or 'cog'."""
    stage = args.stage
    if stage == "cog":
        from train.cog_train import train_cog
        from train.config import LCMConfig
        shape = args.shape or (args.data.replace(".dat", "_shape.json")
                               if args.data else None)
        out_dir = args.save_dir or "checkpoints/cog"
        joint = (args.stage_num == 3) if hasattr(args, 'stage_num') else False
        train_cog(
            cfg=LCMConfig(),
            output_dir=out_dir,
            steps=args.cog_steps,
            lr=args.cog_lr,
            batch_size=args.cog_batch,
            seq_len=args.cog_seq,
            log_every=100,
            save_every=args.cog_save,
            data_path=args.data,
            shape_path=shape,
            lang_ckpt=args.from_lm_ckpt,
            resume=args.resume,
            joint=joint,
        )
        return
    try:
        import lcm as _lcm_mod
        _lcm_mod.train(args)
    except Exception:
        print("Error: cmd_train pass-through requires lcm.py in sys.path",
              file=sys.stderr)
        raise


def cmd_ckpt_diff(args):
    """Compare two checkpoint directories.  numpy-only, no JAX."""
    import numpy as np
    d1, d2 = args.dir1, args.dir2

    def _norm(vec):
        l2 = float(np.sqrt(np.sum(vec ** 2)))
        linf = float(np.max(np.abs(vec)))
        frob = float(np.sqrt(np.sum(vec ** 2)))
        return l2, linf, frob

    def _list_bins(path):
        if not os.path.isdir(path):
            return set()
        return {f for f in os.listdir(path)
                if f.endswith(".bin") and f != "opt_state.bin"}

    bins1 = _list_bins(d1)
    bins2 = _list_bins(d2)
    common = sorted(bins1 & bins2)
    only1 = sorted(bins1 - bins2)
    only2 = sorted(bins2 - bins1)

    if not common and not only1 and not only2:
        print("ckpt-diff: no .bin files found in either directory.")
        return

    header = (f"{'File':>28}  {'Shape':>14}  {'L2':>12}  {'Linf':>12}  "
              f"{'Frob':>12}  {'MaxDiff':>12}")
    print(header)
    print("-" * len(header))

    for fname in common:
        p1 = os.path.join(d1, fname)
        p2 = os.path.join(d2, fname)
        a1 = np.fromfile(p1, dtype=np.float32)
        a2 = np.fromfile(p2, dtype=np.float32)
        if len(a1) != len(a2):
            print(f"  {fname:>28}  size mismatch "
                  f"({len(a1)} vs {len(a2)} floats)")
            continue
        diff = a1 - a2
        l2_1, linf_1, frob_1 = _norm(a1)
        maxdiff = float(np.max(np.abs(diff)))
        shape_str = f"({len(a1)},)"
        print(f"  {fname:>28}  {shape_str:>14}  {l2_1:>12.4f}  "
              f"{linf_1:>12.4f}  {frob_1:>12.4f}  {maxdiff:>12.6f}")

    if only1:
        for f in only1:
            p = os.path.join(d1, f)
            print(f"  {f:>28}  only in dir1  "
                  f"({os.path.getsize(p) / 1024:.0f} KB)")
    if only2:
        for f in only2:
            p = os.path.join(d2, f)
            print(f"  {f:>28}  only in dir2  "
                  f"({os.path.getsize(p) / 1024:.0f} KB)")


def cmd_eval_ppl(args):
    """Load checkpoint + data, compute perplexity with JAX."""
    import numpy as np
    import jax
    import jax.numpy as jnp

    ckpt = args.checkpoint
    config_path = os.path.join(ckpt, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: no config.json in {ckpt}", file=sys.stderr)
        return

    with open(config_path) as f:
        cfg = json.load(f)

    from train.checkpoint import _load_encoder, _load_decoder
    from train.config import LCMConfig

    lcfg = LCMConfig()
    for k, v in cfg.items():
        if hasattr(lcfg, k):
            object.__setattr__(lcfg, k, v)

    enc_path = os.path.join(ckpt, "encoder.bin")
    dec_path = os.path.join(ckpt, "decoder.bin")
    if not os.path.exists(enc_path) or not os.path.exists(dec_path):
        print("Error: encoder.bin or decoder.bin not found in checkpoint",
              file=sys.stderr)
        return

    data_path = getattr(args, 'data', None)
    if not data_path or not os.path.exists(data_path):
        for c in [cfg.get("data_path", ""), "data/zhwiki_tokens.dat",
                  "data/tokens.dat"]:
            if c and os.path.exists(c):
                data_path = c
                break
    if not data_path or not os.path.exists(data_path):
        print("Error: no data file found; use --data to specify",
              file=sys.stderr)
        return

    shape_path = (getattr(args, 'shape', None) or
                  data_path.replace(".dat", "_shape.json"))
    from train.data import WikiDataIter
    data_iter = WikiDataIter(data_path, shape_path, B=1, N=lcfg.max_seq_len)

    from train.model import forward as model_forward
    # Full model params: encoder+decoder alone is NOT enough — train/model.py
    # forward touches route/hrq/sparse/lowrank/manifold/binding/contrast/
    # fusion and would KeyError. load_checkpoint assembles the complete tree
    # (missing codebooks get re-initialised; the config comes from the
    # checkpoint's own config.json).
    from train.checkpoint import load_checkpoint
    params, _, _, _ = load_checkpoint(ckpt, load_opt=False)

    gh = params.get('gen_head')
    if isinstance(gh, dict) and gh.get('format') == 'cog':
        # cog checkpoint: passive channel is a linear readout
        # logits = z @ W_out (z from the encoder), predicting the next token
        # after the window (targets[:, -1] == x[N]) — matches cog training.
        from train.encoder import encoder_forward

        def _eval_step(params, batch):
            inputs, targets = batch
            z = encoder_forward(params['encoder'], inputs, lcfg.n_heads)  # (B, N, d)
            logits = z[:, -1, :] @ params['gen_head']['w_out']  # (B, V)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(
                jnp.take_along_axis(
                    log_probs, targets[:, -1][..., None], axis=-1).squeeze(-1))
            return loss
    else:
        def _eval_step(params, batch):
            inputs, targets = batch
            logits, _, _, extra, _ = model_forward(
                params, None, inputs, lcfg, training=False)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(
                jnp.take_along_axis(
                    log_probs, targets[..., None], axis=-1).squeeze(-1))
            return loss

    batches = int(getattr(args, 'batches', 10))
    losses = []
    for i in range(batches):
        try:
            batch = next(data_iter)
            loss_val = float(_eval_step(params, batch))
            losses.append(loss_val)
            print(f"  batch {i+1:>3d}/{batches}  loss={loss_val:.4f}")
        except StopIteration:
            print(f"  data exhausted at batch {i+1}")
            break

    if not losses:
        print("Error: no batches evaluated", file=sys.stderr)
        return

    avg_loss = float(np.mean(losses))
    ppl = float(np.exp(avg_loss))
    print(f"\n  Average cross-entropy loss: {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.4f}")


def cmd_eval_trace(args):
    """Run inference engine on a short prompt, show per-step trace."""
    import numpy as np

    ckpt = args.checkpoint
    prompt = getattr(args, 'prompt', "Hello")

    import importlib
    lcm_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "lcm.py")
    spec = importlib.util.spec_from_file_location("lcm", lcm_path)
    lcm_mod = importlib.util.module_from_spec(spec)
    sys.modules["lcm"] = lcm_mod
    spec.loader.exec_module(lcm_mod)

    engine = lcm_mod.LCMInferEngine(ckpt)
    engine.print_trace = lambda tr=None: None

    print(f"[EVAL] Prompt: {prompt!r}")
    print(f"[EVAL] Running cognitive loop ...")

    encoding = engine.tokenizer.encode(prompt)
    token_ids = encoding.ids
    if not token_ids or token_ids[0] != 2:
        token_ids = [2] + token_ids
    x = np.array(token_ids[-engine.max_seq_len:], dtype=np.int32)

    z, _ = lcm_mod._encoder_full_with_state(
        engine.encoder, x, engine.n_heads)
    z_q, converged = engine.cognitive_loop(z)

    traces = engine.get_trace()
    if traces is None:
        print("[TRACE] No trace data available from C engine.")
        return

    lat_names = [n[:4] for n in engine.LATTICE_NAMES]
    hdr = (f"{'Step':>4} | "
           f"{'|'.join(f'{n:>6}' for n in lat_names)} | "
           f"{'Entropy':>8} | Conflict")
    print("[TRACE] Per-step trace:")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    n_conflict = 0
    for t in traces:
        wstr = '|'.join(f'{w:>6.3f}' for w in t['weights'])
        w = t['weights']
        entropy = float(-np.sum(w * np.log(np.clip(w, 1e-10, 1.0))))
        print(f"{t['step']:>4} | {wstr} | {entropy:>8.4f} | "
              f"{'YES' if t['has_conflict'] else 'no':>8}")
        if t['has_conflict']:
            n_conflict += 1
    print("-" * len(hdr))
    total = len(traces)
    conv_rate = (total - n_conflict) / max(total, 1) * 100
    print(f"\n[SUMMARY] Total cognitive steps: {total}")
    print(f"[SUMMARY] Converged: {converged}")
    print(f"[SUMMARY] Steps with conflict: {n_conflict}/{total}")
    print(f"[SUMMARY] Convergence rate: {conv_rate:.1f}%")


def cmd_report(args):
    """Print model architecture report from a checkpoint directory."""
    import numpy as np

    ckpt = args.checkpoint
    config_path = os.path.join(ckpt, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: no config.json in {ckpt}", file=sys.stderr)
        return

    with open(config_path) as f:
        cfg = json.load(f)

    d = cfg.get("d_model", 256)
    print(f"Checkpoint: {ckpt}")
    print(f"d_model:    {d}")
    print(f"vocab_size: {cfg.get('vocab_size', '?')}")
    print(f"n_heads:    {cfg.get('n_heads', '?')}")
    print(f"max_seq_len:{cfg.get('max_seq_len', '?')}")
    print(f"n_lattices: {cfg.get('n_lattices', '?')}")
    print()

    codebook_defs = [
        ("hrq_codebook.bin",      "HRQ",       cfg.get("M_top", 512),
         cfg.get("n_hrq_layers", 3)),
        ("sparse_codebook.bin",   "Sparse",    cfg.get("M_sparse", 512), 1),
        ("lowrank_codebook.bin",  "LowRank",   cfg.get("M_lr", 256),
         len(cfg.get("ranks", [2, 4, 8]))),
        ("manifold_codebook.bin", "Manifold",  cfg.get("M_man", 512), 1),
        ("bind_codebook.bin",     "Binding",   cfg.get("M_bind", 512), 3),
        ("contrast_codebook.bin", "Contrast",  cfg.get("M_contrast", 512), 3),
        ("gvalue_codebook.bin",   "GValue",    cfg.get("n_value_pairs", 4), 1),
        ("danger_codebook.bin",   "Danger",    cfg.get("M_danger", 256), 1),
    ]

    total_params = 0
    total_size = 0
    print(f"{'Module':>12}  {'Type':>10}  {'M':>6}  {'d':>4}  "
          f"{'Layers':>6}  {'Size':>10}")
    print("-" * 65)

    for fname, cb_type, M, layers in codebook_defs:
        fpath = os.path.join(ckpt, fname)
        if not os.path.exists(fpath):
            continue
        fsize = os.path.getsize(fpath)
        n_params = fsize // 4
        total_params += n_params
        total_size += fsize
        sz_str = (f"{fsize / 1024:.0f} KB" if fsize < 1e6
                  else f"{fsize / 1e6:.1f} MB")
        print(f"  {cb_type:>10}  {'codebook':>10}  {M:>6}  {d:>4}  "
              f"{layers:>6}  {sz_str:>10}")

    enc_path = os.path.join(ckpt, "encoder.bin")
    if os.path.exists(enc_path):
        esize = os.path.getsize(enc_path)
        ne = cfg.get("n_encoder_layers", 2)
        total_params += esize // 4
        total_size += esize
        sz_str = (f"{esize / 1024:.0f} KB" if esize < 1e6
                  else f"{esize / 1e6:.1f} MB")
        print(f"  {'Encoder':>10}  {'layers':>10}  {ne:>6}  {d:>4}  "
              f"{'':>6}  {sz_str:>10}")

    dec_path = os.path.join(ckpt, "decoder.bin")
    if os.path.exists(dec_path):
        dsize = os.path.getsize(dec_path)
        total_params += dsize // 4
        total_size += dsize
        sz_str = (f"{dsize / 1024:.0f} KB" if dsize < 1e6
                  else f"{dsize / 1e6:.1f} MB")
        print(f"  {'Decoder':>10}  {'gen_head':>10}  {'':>6}  {d:>4}  "
              f"{'':>6}  {sz_str:>10}")

    tok_path = os.path.join(ckpt, "tokenizer.json")
    if os.path.exists(tok_path):
        import tokenizers as _tk
        try:
            tok = _tk.Tokenizer.from_file(str(tok_path))
            print(f"\n  Tokenizer: vocab_size={tok.get_vocab_size()}")
        except Exception:
            pass

    print(f"\n{'':>12}  {'':>10}  {'':>6}  {'':>4}  {'':>6}  "
          f"{total_size / 1e6:.1f} MB total")
    print(f"  Total parameters (float32): {total_params:,}")


def cmd_serve(args):
    """Simple HTTP server using ONLY Python stdlib.

    POST /v1/completions, GET /v1/models.
    """
    import http.server
    import json as _json
    import urllib.parse
    import threading

    port = int(getattr(args, 'port', 8080))

    import importlib
    lcm_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "lcm.py")
    spec = importlib.util.spec_from_file_location("lcm", lcm_path)
    lcm_mod = importlib.util.module_from_spec(spec)
    sys.modules["lcm"] = lcm_mod
    spec.loader.exec_module(lcm_mod)

    _engine = lcm_mod.LCMInferEngine(args.checkpoint)
    _engine_max_new = int(getattr(args, 'max_new', 128))
    _engine_temp = float(getattr(args, 'temperature', 0.7))

    request_id = [0]
    lock = threading.Lock()

    class LCMHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("[HTTP] %s\n" % (fmt % args))

        def _respond(self, code, body):
            body_bytes = _json.dumps(
                body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/v1/models":
                self._respond(
                    200, {"data": [{"id": "lcm", "object": "model"}]})
            else:
                self._respond(404, {"error": "not_found"})

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/v1/completions":
                self._respond(404, {"error": "not_found"})
                return

            content_len = int(self.headers.get("Content-Length", 0))
            if content_len == 0:
                self._respond(400, {"error": "empty body"})
                return
            body = self.rfile.read(content_len)
            try:
                req = _json.loads(body)
            except Exception as e:
                self._respond(
                    400, {"error": f"invalid json: {e}"})
                return

            prompt = req.get("prompt", "")
            if not prompt:
                self._respond(400, {"error": "empty prompt"})
                return

            max_new = req.get("max_tokens", _engine_max_new)
            temp = req.get("temperature", _engine_temp)

            try:
                with lock:
                    rid = request_id[0]
                    request_id[0] += 1
                    tokens = list(_engine.generate(
                        prompt, max_new=max_new, temperature=temp))
                    text = _engine.tokenizer.decode(
                        tokens, skip_special_tokens=True)
                resp = {
                    "id": f"cmpl-{rid}",
                    "object": "text_completion",
                    "choices": [{"text": text, "index": 0}],
                    "usage": {
                        "prompt_tokens":
                            len(_engine.tokenizer.encode(prompt).ids),
                        "completion_tokens": len(tokens),
                        "total_tokens":
                            len(tokens) +
                            len(_engine.tokenizer.encode(prompt).ids),
                    },
                }
                self._respond(200, resp)
            except Exception as e:
                self._respond(500, {"error": str(e)})

    server = http.server.HTTPServer(("0.0.0.0", port), LCMHandler)
    print(f"[SERVE] Listening on port {port}")
    print(f"[SERVE] POST /v1/completions")
    print(f"[SERVE] GET  /v1/models")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVE] Shutting down.")
        server.server_close()


def cmd_batch(args):
    """Batch generation from JSONL file."""
    import json as _json

    ckpt = args.checkpoint

    import importlib
    lcm_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "lcm.py")
    spec = importlib.util.spec_from_file_location("lcm", lcm_path)
    lcm_mod = importlib.util.module_from_spec(spec)
    sys.modules["lcm"] = lcm_mod
    spec.loader.exec_module(lcm_mod)

    engine = lcm_mod.LCMInferEngine(ckpt)
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return

    lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(_json.loads(line))

    from tqdm import tqdm
    results = []
    for item in tqdm(lines, desc="   generating", unit="item"):
        prompt = item.get("prompt", "")
        max_new = int(item.get("max_new", 128))
        temp = float(item.get("temp", 0.7))
        try:
            tokens = list(engine.generate(
                prompt, max_new=max_new, temperature=temp))
            text = engine.tokenizer.decode(
                tokens, skip_special_tokens=True)
        except Exception as e:
            text = f"[ERROR: {e}]"
        item["generation"] = text
        results.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(_json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(results)} results to {output_path}")


def cmd_chat(args):
    """Interactive REPL with multi-turn context."""
    import json as _json

    ckpt = args.checkpoint

    import importlib
    lcm_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "lcm.py")
    spec = importlib.util.spec_from_file_location("lcm", lcm_path)
    lcm_mod = importlib.util.module_from_spec(spec)
    sys.modules["lcm"] = lcm_mod
    spec.loader.exec_module(lcm_mod)

    engine = lcm_mod.LCMInferEngine(ckpt)

    history = []
    max_new = int(getattr(args, 'max_new', 128))
    temp = float(getattr(args, 'temp', 0.7))

    def _format_prompt(history):
        lines = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    print(f"[CHAT] Interactive REPL. Type /help for commands.")
    print(f"       max_new={max_new}, temp={temp}")
    print("-" * 50)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Bye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        if user_input.startswith("/"):
            parts = user_input[1:].strip().split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if cmd == "reset":
                history.clear()
                print("[CHAT] Conversation history reset.")
                continue
            elif cmd == "save":
                if not arg:
                    print("[CHAT] Usage: /save <path>")
                    continue
                try:
                    with open(arg, "w", encoding="utf-8") as f:
                        f.write(_json.dumps(
                            history, ensure_ascii=False, indent=2))
                    print(f"[CHAT] Log saved to {arg}")
                except Exception as e:
                    print(f"[CHAT] Error saving: {e}")
                continue
            elif cmd == "help":
                print("  /reset       -- Clear conversation history")
                print("  /save <path> -- Save conversation log to JSON")
                print("  /help        -- Show this message")
                print("  quit/exit    -- Exit")
                continue
            else:
                print(f"[CHAT] Unknown command: /{cmd}")
                continue

        history.append({"role": "user", "content": user_input})
        prompt = _format_prompt(history)
        print("--- Generating ---")

        gen_text = ""
        try:
            for token_id in engine.generate(
                    prompt, max_new=max_new, temperature=temp):
                text = engine.tokenizer.decode(
                    [token_id], skip_special_tokens=True)
                print(text, end="", flush=True)
                gen_text += text
            print()
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            print(f"\n[Error: {e}]")

        history.append({"role": "assistant", "content": gen_text})


def cmd_data_sample(args):
    """Sample and decode tokenized data."""
    import json as _json
    import numpy as np

    dat_path = args.data
    shape_json = (getattr(args, 'shape', None) or
                  dat_path.replace(".dat", "_shape.json"))
    tokenizer_path = getattr(args, 'tokenizer', "data/tokenizer.json")

    if not os.path.exists(dat_path):
        print(f"Error: .dat file not found: {dat_path}", file=sys.stderr)
        return
    if not os.path.exists(tokenizer_path):
        print(f"Error: tokenizer not found: {tokenizer_path}",
              file=sys.stderr)
        return

    n_tokens = None
    if os.path.exists(shape_json):
        with open(shape_json) as f:
            meta = _json.load(f)
        n_tokens = meta.get("n_tokens")

    data = np.memmap(dat_path, dtype=np.uint16, mode="r")
    if n_tokens is None:
        n_tokens = len(data)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    count = int(getattr(args, 'count', 5))
    length = int(getattr(args, 'length', 100))
    seed = int(getattr(args, 'seed', 42))

    rng = np.random.default_rng(seed)
    max_start = n_tokens - length - 1

    print(f"Data file:   {dat_path}")
    print(f"Total tokens:{n_tokens:,}")
    print(f"Tokenizer:   {tokenizer_path}")
    print(f"Samples:     {count} x {length} tokens (seed={seed})")
    print()

    for i in range(count):
        start = int(rng.integers(0, max_start)) if max_start > 0 else 0
        ids = data[start:start + length].tolist()
        text = tok.decode(ids)
        print(f"[{i}] (offset {start})")
        print(f"    {text[:200]}")
        print()


def cmd_data_stats(args):
    """Token data statistics."""
    import json as _json
    import numpy as np

    dat_path = args.data
    shape_json = (getattr(args, 'shape', None) or
                  dat_path.replace(".dat", "_shape.json"))
    tokenizer_path = getattr(args, 'tokenizer', "data/tokenizer.json")
    top_k = int(getattr(args, 'top_k', 20))

    if not os.path.exists(dat_path):
        print(f"Error: .dat not found: {dat_path}", file=sys.stderr)
        return

    n_tokens = None
    if os.path.exists(shape_json):
        with open(shape_json) as f:
            meta = _json.load(f)
        n_tokens = meta.get("n_tokens")

    data = np.memmap(dat_path, dtype=np.uint16, mode="r")
    if n_tokens is None:
        n_tokens = len(data)

    file_size = os.path.getsize(dat_path)
    print(f"Data:  {dat_path}")
    print(f"Shape: {n_tokens:,} tokens  ({file_size / 1e6:.0f} MB)")
    print()

    if os.path.exists(tokenizer_path):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        vs = tok.get_vocab_size()
        unique = len(np.unique(data))
        coverage = unique / vs * 100 if vs > 0 else 0
        print(f"Tokenizer:   vocabsize={vs}")
        print(f"Unique:     {unique:,} / {vs}  ({coverage:.1f}% coverage)")
        print()

        counts = np.bincount(data.ravel(), minlength=vs)
        top_indices = np.argsort(counts)[::-1][:top_k]
        print(f"Top-{top_k} most frequent tokens:")
        print(f"{'Rank':>4}  {'ID':>6}  {'Freq':>12}  "
              f"{'Pct':>8}  {'Text'}")
        total = int(np.sum(counts))
        for rank, idx in enumerate(top_indices):
            try:
                text = tok.decode([int(idx)])
            except Exception:
                text = f"[{idx}]"
            freq = int(counts[idx])
            pct = freq / total * 100 if total > 0 else 0
            print(f"{rank + 1:>4}  {idx:>6}  {freq:>12,}  "
                  f"{pct:>7.3f}%  {text[:40]}")
        print()

        print("Token frequency histogram (ASCII):")
        max_freq = (int(counts[top_indices[0]])
                    if len(top_indices) > 0 else 1)
        bar_width = 50
        for rank, idx in enumerate(top_indices):
            freq = int(counts[idx])
            bar_len = (int(freq / max_freq * bar_width)
                       if max_freq > 0 else 0)
            bar = "#" * bar_len
            try:
                label = (tok.decode([int(idx)])[:12]
                         .replace("\n", "\\n"))
            except Exception:
                label = f"[{idx}]"
            freq_str = f"{freq:,}"
            print(f"  {label:>12} |{bar:<{bar_width}}| {freq_str:>10}")
    else:
        print(f"Tokenizer not found at {tokenizer_path}")
        print(f"Total unique token IDs: {len(np.unique(data)):,}")
        unique_ids, counts = np.unique(data, return_counts=True)
        order = np.argsort(counts)[::-1][:top_k]
        print(f"Top-{top_k} token IDs by frequency:")
        for rank, pos in enumerate(order):
            print(f"  {rank + 1:>4}: id={unique_ids[pos]:>6}  "
                  f"freq={counts[pos]:>10,}")
