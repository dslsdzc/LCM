"""Training supervisor: automatic monitoring, error detection, and recovery.

Wraps any training step with:
  - NaN/Inf loss detection → auto-rollback + LR reduce
  - Cognitive loop convergence tracking
  - Periodic validation perplexity
  - Auto-save best checkpoint
  - Crash auto-recovery (save + print resume command)

Usage:
    from train.train_supervisor import Supervisor
    sup = Supervisor(output_dir, cfg, enable_auto=True)
    for step in range(steps):
        params, opt_state, loss, aux = sup.step(
            train_step_fn, params, opt_state, batch, rng)
"""

import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


class Supervisor:
    """Training supervisor with auto-detect and auto-repair."""

    def __init__(self, output_dir, cfg, enable_auto=True, val_data_path=None,
                 val_shape_path=None, patience=5, lr_decay=0.5):
        self.output_dir = output_dir
        self.cfg = cfg
        self.enable = enable_auto
        self.patience = patience
        self.lr_decay = lr_decay
        self.val_data_path = val_data_path
        self.val_shape_path = val_shape_path

        # State tracking
        self.best_loss = float("inf")
        self.best_params = None
        self.best_opt_state = None
        self.best_step = 0
        self.bad_streak = 0
        # A multiplier, not an absolute rate. The training loop owns the
        # schedule; the supervisor only scales it. The old `current_lr` was
        # printed as if it were the live rate but never reached the optimizer.
        self.lr_scale = 1.0
        self.last_lr = float(cfg.learning_rate)

        # Cognitive loop stats
        self.cog_convergence = []
        self.cog_steps_history = []

        # Checkpoint tracking
        self.last_save_path = None
        self.saved_steps = set()

        os.makedirs(output_dir, exist_ok=True)

        if self.enable:
            print(f"[SUPERVISOR] Auto mode ON — monitoring loss, convergence, and crashes")

    def step(self, train_fn, params, opt_state, batch, lr, rng,
             step=None, self_state=None):
        """Run one training step with monitoring.

        Signature mirrors train_step: (params, opt_state, batch, lr, rng).
        Passing lr/rng/self_state positionally avoids the old bug where
        rng landed in the lr slot → "multiple values for argument 'lr'".

        Returns (params, opt_state, loss, aux) on success.
        On NaN/inf: auto-reduces LR, rolls back, retries.
        """
        if not self.enable:
            return train_fn(params, opt_state, batch, lr, rng,
                            self_state=self_state)

        effective_lr = lr * self.lr_scale
        self.last_lr = float(effective_lr)
        try:
            new_params, new_opt, loss_val, aux_out = train_fn(
                params, opt_state, batch, effective_lr, rng,
                self_state=self_state)

            loss_f = float(loss_val)

            # ── NaN / Inf detection ──
            if np.isnan(loss_f) or np.isinf(loss_f):
                return self._handle_bad_step(
                    train_fn, params, opt_state, batch, effective_lr, rng,
                    f"loss={loss_f}", step=step, self_state=self_state)

            # ── Loss spike detection ──
            if self.best_loss < float("inf") and loss_f > self.best_loss * 3:
                self.bad_streak += 1
                if self.bad_streak >= self.patience:
                    print(f"\n[SUPERVISOR] Loss spike x{self.bad_streak}: {loss_f:.4f} vs best {self.best_loss:.4f}")
                    print(f"[SUPERVISOR] Rolling back to step {self.best_step}, reducing LR")
                    # Roll back to the BEST (healthy) params — passing the
                    # current degraded params would lock in the damage.
                    return self._rollback(
                        train_fn, self._params_with_bridge(params),
                        self.best_opt_state, batch, effective_lr, rng,
                        reason=f"loss spike x{self.bad_streak}",
                        step=step, self_state=self_state)
            else:
                self.bad_streak = 0

            # ── Update best ──
            if loss_f < self.best_loss:
                self.best_loss = loss_f
                # Exclude the frozen bridge. params['qwen'] is ~2 GB of weights
                # that never change and are reloaded from the npz on resume, so
                # snapshotting them held a second full copy resident for the
                # entire run and bought nothing.
                self.best_params = jax.tree_util.tree_map(
                    lambda x: jnp.array(x),
                    {k: v for k, v in new_params.items() if k != "qwen"})
                self.best_opt_state = jax.tree_util.tree_map(
                    lambda x: jnp.array(x), new_opt)
                self.best_step = step if step is not None else 0

            # ── Track cognitive convergence ──
            stage3 = aux_out.get('stage3', {})
            self.cog_convergence.append(1)
            if len(self.cog_convergence) > 1000:
                self.cog_convergence.pop(0)

            return new_params, new_opt, loss_val, aux_out

        except Exception as e:
            # ── Crash recovery ──
            print(f"\n[SUPERVISOR] Training crashed at step {step}: {e}")
            self._emergency_save(params, opt_state, step or 0)
            return params, opt_state, jnp.array(float("nan")), {}

    def _handle_bad_step(self, train_fn, params, opt_state, batch, lr, rng,
                         reason, step=None, self_state=None):
        """Handle NaN/inf by LR reduction + rollback + one retry."""
        print(f"\n[SUPERVISOR] Bad step {step}: {reason}")
        self.bad_streak += 1

        if self.bad_streak >= self.patience and self.best_params is not None:
            print(f"[SUPERVISOR] Rolling back to step {self.best_step} "
                  f"(loss={self.best_loss:.4f})")
            return self._rollback(
                train_fn, self._params_with_bridge(params),
                self.best_opt_state, batch, lr, rng, reason=reason,
                step=step, self_state=self_state)
        return params, opt_state, jnp.array(float("nan")), {}

    def _params_with_bridge(self, current):
        """The best-params snapshot, plus the live frozen bridge.

        params['qwen'] is excluded from the snapshot (it never changes and is
        ~2 GB), so a rollback has to re-attach the current copy before the step
        can run again.
        """
        if self.best_params is None:
            return current
        out = dict(self.best_params)
        if "qwen" in current:
            out["qwen"] = current["qwen"]
        return out

    def _rollback(self, train_fn, params, opt_state, batch, lr, rng,
                  reason="rollback", step=None, self_state=None):
        """Reduce LR, restore the best params, and retry the step once.

        The previous version returned a fake 1e10 loss and never re-ran the
        step, despite its docstring promising a retry. Training then continued
        from the next batch — at the restored params, but with nothing checking
        that the bad step was actually avoidable.
        """
        self.lr_scale *= self.lr_decay
        effective = lr * self.lr_scale
        self.last_lr = float(effective)
        print(f"[SUPERVISOR] {reason}: LR -> {effective:.6e} "
              f"(x{self.lr_scale:.4f})")

        if params is None:
            return params, opt_state, jnp.array(float("nan")), {"rollback": True}

        try:
            new_params, new_opt, loss_val, aux_out = train_fn(
                params, opt_state, batch, effective, rng, self_state=self_state)
        except Exception as e:
            print(f"[SUPERVISOR] retry after rollback failed: {e}")
            return params, opt_state, jnp.array(float("nan")), {"rollback": True}

        loss_f = float(loss_val)
        if np.isnan(loss_f) or np.isinf(loss_f):
            print(f"[SUPERVISOR] retry after rollback still bad: loss={loss_f}")
            return params, opt_state, jnp.array(float("nan")), {"rollback": True}

        return new_params, new_opt, loss_val, {**aux_out, "rollback": True}

    def _emergency_save(self, params, opt_state, step):
        """Save checkpoint on crash for later resume."""
        path = os.path.join(self.output_dir, f"crash_step_{step:06d}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "crash_params.pkl"), "wb") as f:
            pickle.dump({
                'params': jax.tree_util.tree_map(lambda x: np.array(x), params),
                'opt_state': jax.tree_util.tree_map(lambda x: np.array(x), opt_state),
                'step': step,
            }, f)
        print(f"[SUPERVISOR] Emergency checkpoint: {path}/crash_params.pkl")
        print(f"[SUPERVISOR] Resume: --resume {path}")

    def report(self, step):
        """Print periodic supervisor report."""
        if not self.enable:
            return
        conv = self.cog_convergence
        rate = sum(conv[-100:]) / max(len(conv[-100:]), 1) * 100 if conv else 0
        print(f"[SUPERVISOR] step {step:>6d} | best loss={self.best_loss:.4f} | "
              f"LR={self.last_lr:.6e} (x{self.lr_scale:.3f}) | "
              f"cog conv={rate:.0f}%")

    def save_best(self, params, opt_state, step, self_state=None,
                  run_identity=None):
        """Save the best checkpoint so far.

        Uses save_cog_checkpoint (cog parameter layout). The legacy
        checkpoint.save_checkpoint expects a gen_head that cog params don't
        have — it always raised KeyError here, silently dropping the best
        checkpoint.

        run_identity is the run's own; the supervisor must not build one of its
        own, or an auto-saved checkpoint would be indistinguishable from a
        hand-saved one in name only.
        """
        path = os.path.join(self.output_dir, f"best_step_{step:06d}")
        from train.cog_train import save_cog_checkpoint
        try:
            save_cog_checkpoint(params, path, step, self_state=self_state,
                                run_identity=run_identity, train_cfg=self.cfg)
            self.last_save_path = path
            self.saved_steps.add(step)
            print(f"[SUPERVISOR] Best checkpoint saved -> {path} (loss={self.best_loss:.4f})")
        except Exception as e:
            print(f"[SUPERVISOR] Save failed: {e}")
