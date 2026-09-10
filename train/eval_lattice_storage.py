"""Causal interventions testing whether the active channel uses lattice memory.

This evaluator never loads Qwen.  It compares the intact student with matched
interventions on the lattice contribution while keeping encoder and generation
head weights identical.
"""
import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tokenizers import Tokenizer

from train.cog_loop import cog_loop_scan
from train.cog_train import pack_codebooks_for_c
from train.config import LCMConfig
from train.encoder import encoder_forward
from train.fusion import gen_head_forward


def encode(rows, tok, gids, q_len=32, a_len=24):
    g2l = {int(token): index for index, token in enumerate(gids)}
    pad = g2l[tok.token_to_id('<|endoftext|>')]
    bos = g2l[tok.token_to_id('<|im_start|>')]
    eos = g2l[tok.token_to_id('<|im_end|>')]
    questions, inputs, targets, masks = [], [], [], []
    for row in rows:
        question = [g2l.get(token, pad) for token in tok.encode(row['user']).ids[:q_len]]
        answer = [g2l.get(token, pad) for token in tok.encode(row['answer']).ids[:a_len - 1]] + [eos]
        questions.append(question + [pad] * (q_len - len(question)))
        source = [bos] + answer[:-1]
        inputs.append(source + [pad] * (a_len - len(source)))
        targets.append(answer + [pad] * (a_len - len(answer)))
        masks.append([1.] * len(answer) + [0.] * (a_len - len(answer)))
    return tuple(jnp.asarray(x) for x in (questions, inputs, targets, masks))


def normalize(x):
    return x / (jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True)) + 1e-6)


def states(params, questions, cfg):
    encoded = encoder_forward(params['encoder'], questions, cfg.n_heads)
    codebooks = pack_codebooks_for_c(params)
    code_norm = jnp.sqrt(sum(jnp.mean(jnp.sum(c * c, axis=-1)) for c in codebooks) / len(codebooks))
    encoded = encoded * (code_norm / (jnp.sqrt(jnp.mean(jnp.sum(encoded * encoded, axis=-1))) + 1e-8))
    run = lambda value: cog_loop_scan(
        value, codebooks, max_steps=cfg.max_inference_steps,
        thresholds=None, tau=.5)[0][-1]
    lattice = jax.vmap(run)(encoded)
    return encoded, lattice


def evaluate(params, cfg, questions, inputs, targets, masks):
    encoded, lattice = states(params, questions, cfg)
    variants = {
        'normal': normalize(encoded + lattice),
        'encoder_only': normalize(encoded),
        'lattice_only': normalize(lattice),
        'shuffled_lattice': normalize(encoded + jnp.roll(lattice, 1, axis=0)),
        'fixed_lattice': normalize(encoded + jnp.broadcast_to(lattice.mean(0), lattice.shape)),
        'zero_state': jnp.zeros_like(encoded),
    }
    result = {}
    normal_pred = None
    for name, state in variants.items():
        logits = gen_head_forward(params['gen_head'], state, inputs)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        ce = (losses * masks).sum() / masks.sum()
        pred = jnp.argmax(logits, axis=-1)
        if normal_pred is None:
            normal_pred = pred
        result[name] = {
            'cross_entropy': float(ce),
            'perplexity': float(jnp.exp(jnp.minimum(ce, 20.))),
            'token_accuracy': float(((pred == targets) * masks).sum() / masks.sum()),
            'prediction_change_vs_normal': float(((pred != normal_pred) * masks).sum() / masks.sum()),
        }
    normal_ce = result['normal']['cross_entropy']
    for value in result.values():
        value['ce_delta_vs_normal'] = value['cross_entropy'] - normal_ce
    result['diagnostics'] = {
        'encoder_lattice_cosine': float(jnp.mean(jnp.sum(normalize(encoded) * normalize(lattice), axis=-1) / encoded.shape[-1])),
        'lattice_between_sample_variance': float(jnp.mean(jnp.var(lattice, axis=0))),
        'criterion': (
            'Support requires normal CE < encoder_only CE and shuffled_lattice CE, '
            'with non-zero lattice between-sample variance.'),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', default='checkpoints/Qwen2.5-0.5B-Instruct/tokenizer.json')
    parser.add_argument('--data', default='data/causal_dialogue.json')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    with open(os.path.join(args.checkpoint, 'student.pkl'), 'rb') as handle:
        checkpoint = pickle.load(handle)
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint['params'])
    cfg = LCMConfig(**checkpoint['config'])
    gids = np.load(os.path.join(args.checkpoint, 'global_token_ids.npy'))
    with open(args.data, encoding='utf-8') as handle:
        rows = json.load(handle)
    batch = encode(rows, tok, gids, q_len=cfg.max_seq_len)
    result = evaluate(params, cfg, *batch)
    result['checkpoint_state_mode'] = checkpoint.get('state_mode', 'residual')
    output = args.output or os.path.join(args.checkpoint, 'lattice_storage_eval.json')
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
