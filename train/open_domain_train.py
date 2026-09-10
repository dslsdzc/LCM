"""Expand the causal student to the full Qwen tokenizer vocabulary."""
import argparse
import dataclasses
import json
import os
import pickle
import random

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer

from train.config import LCMConfig
from train.causal_student_train import init_student, z_state


def compact_forward(h, z, x):
    """Generation head with a tied full-vocabulary embedding matrix."""
    e = h['w_embed'][x] + 0.5 * z[:, None, :]
    u = jnp.concatenate([z[:, None, :], e], axis=1)
    q = jax.nn.elu(u @ h['w_q']) + 1.
    k = jax.nn.elu(u @ h['w_k']) + 1.
    v = u @ h['w_v']
    kv = jnp.cumsum(k[:, :, :, None] * v[:, :, None, :], axis=1)
    ks = jnp.cumsum(k, axis=1)
    a = jnp.einsum('bnd,bnde->bne', q, kv) / (
        jnp.einsum('bnd,bnd->bn', q, ks)[..., None] + 1e-8)
    a = a @ h['w_o']
    g = jax.nn.sigmoid(a @ h['w_1']) * (a @ h['w_2'])
    hidden = g @ h['w_3']
    return (hidden @ h['w_embed'].T)[:, 1:]


def build_rows(alpaca_path, causal_path, limit, seed=17):
    with open(alpaca_path, encoding='utf-8') as f:
        raw = json.load(f)
    rows = []
    for row in raw:
        prompt = row['instruction']
        if row.get('input'):
            prompt += '\n' + row['input']
        answer = row.get('output', '')
        if prompt.strip() and answer.strip():
            rows.append((prompt, answer))
    random.Random(seed).shuffle(rows)
    rows = rows[:limit]
    with open(causal_path, encoding='utf-8') as f:
        causal = json.load(f)
    rows += [(row['user'], row['answer']) for _ in range(5) for row in causal]
    return rows


def encode(rows, tok, q_len=48, a_len=48):
    pad = tok.token_to_id('<|endoftext|>')
    bos = tok.token_to_id('<|im_start|>')
    eos = tok.token_to_id('<|im_end|>')
    out = []
    for question, answer in rows:
        qi = tok.encode(question).ids[:q_len]
        ai = tok.encode(answer).ids[:a_len - 1] + [eos]
        qi += [pad] * (q_len - len(qi))
        x = [bos] + ai[:-1]
        x += [pad] * (a_len - len(x))
        y = ai + [pad] * (a_len - len(ai))
        mask = [1.] * len(ai) + [0.] * (a_len - len(ai))
        out.append((qi, x, y, mask))
    return tuple(np.asarray(v) for v in zip(*out)), pad, bos, eos


def transfer_small(params, small_path, gids_path):
    with open(small_path, 'rb') as f:
        old = pickle.load(f)['params']
    gids = np.load(gids_path)

    def copy_matching(dst, src):
        if isinstance(dst, dict):
            return {k: copy_matching(v, src[k]) if k in src else v for k, v in dst.items()}
        if isinstance(dst, list):
            return [copy_matching(v, src[i]) for i, v in enumerate(dst)]
        return jnp.asarray(src) if np.shape(dst) == np.shape(src) else dst

    params = copy_matching(params, old)
    params['encoder']['embed'] = params['encoder']['embed'].at[gids].set(
        jnp.asarray(old['encoder']['embed']))
    params['gen_head']['w_embed'] = params['gen_head']['w_embed'].at[gids].set(
        jnp.asarray(old['gen_head']['w_embed']))
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpaca', default='data/alpaca_zh/alpaca_gpt4_data_zh.json')
    ap.add_argument('--causal', default='data/causal_dialogue.json')
    ap.add_argument('--tokenizer', default='checkpoints/Qwen2.5-0.5B-Instruct/tokenizer.json')
    ap.add_argument('--init', default='checkpoints/causal_student_v3_aug_s800')
    ap.add_argument('--output', default='checkpoints/open_domain_v4')
    ap.add_argument('--semantic-embed', default=None)
    ap.add_argument('--examples', type=int, default=4000)
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--batch-size', type=int, default=2)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    rows = build_rows(args.alpaca, args.causal, args.examples)
    (questions, inputs, targets, masks), _, _, _ = encode(rows, tok)
    vocab_size = tok.get_vocab_size()
    cfg = dataclasses.replace(
        LCMConfig(), d_model=64, d_ff=96, n_heads=4, d_head=16,
        vocab_size=vocab_size, max_seq_len=48, M_top=64, M_fine=32,
        M_sparse=64, M_lr=32, M_man=64, M_bind=64, M_contrast=64,
        n_self_codes=16, max_inference_steps=4, use_bf16=False)
    params = init_student(cfg, jax.random.PRNGKey(11), vocab_size)
    del params['gen_head']['w_3']
    params['gen_head']['w_3'] = jax.random.normal(
        jax.random.PRNGKey(12), (cfg.d_model * 4, cfg.d_model)) * (
            (cfg.d_model * 4) ** -0.5)
    if args.semantic_embed:
        semantic = jnp.asarray(np.load(args.semantic_embed))
        params['encoder']['embed'] = semantic
        params['gen_head']['w_embed'] = semantic
    params = transfer_small(
        params, args.init + '/student.pkl', args.init + '/global_token_ids.npy')
    optimizer = optax.adamw(learning_rate=3e-4, weight_decay=0.005)
    state = optimizer.init(params)

    @jax.jit
    def step(current, opt_state, question, source, target, mask):
        def loss_fn(candidate):
            latent = z_state(candidate, question, cfg)
            logits = compact_forward(candidate['gen_head'], latent, source)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, target)
            return (ce * mask).sum() / mask.sum()
        loss, grads = jax.value_and_grad(loss_fn)(current)
        grads = optax.clip_by_global_norm(1.0).update(grads, None)[0]
        updates, opt_state = optimizer.update(grads, opt_state, current)
        return optax.apply_updates(current, updates), opt_state, loss

    rng = np.random.default_rng(23)
    for index in range(args.steps):
        batch = rng.integers(0, len(rows), size=args.batch_size)
        params, state, loss = step(
            params, state, jnp.asarray(questions[batch]), jnp.asarray(inputs[batch]),
            jnp.asarray(targets[batch]), jnp.asarray(masks[batch]))
        if index % 50 == 0 or index + 1 == args.steps:
            print(f'step {index + 1} ce={float(loss):.4f}', flush=True)
    os.makedirs(args.output, exist_ok=True)
    with open(args.output + '/student_full_vocab.pkl', 'wb') as f:
        pickle.dump({
            'params': jax.tree_util.tree_map(np.asarray, params),
            'config': dataclasses.asdict(cfg),
            'step': args.steps,
        }, f)
    print('saved student-only full-vocabulary checkpoint', args.output)


if __name__ == '__main__':
    main()
