# LCM Training Experiment Report — V1

> From GTX 1650 (4GB) to RTX 5090 (32GB): from training a language model from scratch to integrating a frozen Qwen2.5-0.5B

---

## Environment

| Item | Spec |
|------|------|
| Hardware | GTX 1650 (4GB) → RTX 5090 (32GB) |
| Framework | JAX + Optax |
| Data | Chinese Wikipedia (zhwiki, 540M tokens) |
| Tokenizer | BPE, V=30,000 |

---

## Experiment A: Language Model Training (Stage 1)

### A1: 4-layer + codebook, 110K steps (no log)

Training log lost. Generation output was Chinese vocabulary fragments with no grammar.

### A2: Resume 5K steps (step 115000→119600)

**Params:** lr=1e-4, B=4, N=256, pos_embed added at resume

| Step | Loss | PPL |
|------|------|-----|
| 115,000 | 0.11 | 1.1 |
| 115,100 | 7.89 | 2,674 (reset due to pos_embed init) |
| 119,600 | 0.04 | 1.0 |

**Conclusion:** PPL reset to 2674 then recovered to 1.0. Still overfits.

### A3: Resume 6K steps (step 120000→125900)

**Params:** lr=1e-4, B=4, N=256

| Step | PPL |
|------|-----|
| 125,900 | 10.4 |

**Conclusion:** Stable at PPL ~10.4. Model at capacity limit.

### A4: Scratch + pos_embed, 10K steps

**Params:** lr=3e-4, B=4, N=256

| Step | PPL |
|------|-----|
| 10,300 | 1.0 |

**Conclusion:** pos_embed helps but model still overfits. 4L d=256 insufficient for syntax.

### A5: 8-layer + mHC + MTP + codebook, 50K steps

**Params:** lr=3e-4, B=4, N=256, n_hc=2, MTP depth=2, codebook 6×512

| Step | PPL |
|------|-----|
| 49,900 | 1.5 |

**Generation test (50K checkpoint):**
```
Prompt: 今天天气 (today's weather)
Output: 今天天气er���豪强机能�二次大战使徒�的形成�...
```
Recognizes Chinese word fragments but no syntax. PPL 1.5 = local n-gram statistics, not language understanding.

---

## Experiment B: Qwen2.5-0.5B Integration + Stage 2 Cognitive Training

| Component | Detail |
|-----------|--------|
| Language ability | Qwen2.5-0.5B (frozen, excluded from optimizer) |
| Cognitive module | Encoder + 6 codebooks + cognitive loop (32 steps) |
| Bridge | z_proj (896×256) |
| Hardware | RTX 5090 (32GB) |
| Reached step | 18,000 (disk full) |

**NaN bug:** Qwen weights were passed to `optimizer.init()`. AdamW momentum slots accumulated NaN gradients, corrupting all parameters. Fix: exclude Qwen from optimizer entirely.

**Engineering issues:**
- `training_log.txt` not implemented in `cog_train.py`
- Checkpoint 2GB each, 50GB disk filled after 19 saves
- Progress bar blocked by NaN steps

---

## Key Lessons

| Finding | Conclusion |
|---------|-----------|
| 4L d=256 always overfits | Small transformers can't learn syntax |
| 8L + mHC + MTP helps but not enough | 25M params is the ceiling |
| Qwen freeze works but needs optimizer isolation | stop_gradient not enough |
| training_log must be implemented | Otherwise curves are lost |

---

*Generated: 2026-07-17*
*Experiment span: 2026-06-11 → 2026-07-10*
