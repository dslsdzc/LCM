"""Student-only evaluation."""
import argparse
import json
import pickle

import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from train.config import LCMConfig
from train.causal_student_train import z_state
from train.open_domain_train import compact_forward

PROMPTS = [
    "你好，请介绍一下你自己。",
    "为什么天空看起来是蓝色的？",
    "请写一个关于春天的短句。",
    "2加3等于多少？请解释。",
    "如果明天下雨，出门应该准备什么？",
    "猫和狗有什么区别？",
    "如何把一个复杂任务分成小步骤？",
    "请用一句话解释什么是因果关系。",
    "我今天有点疲惫，可以怎样安排休息？",
    "水烧开以后为什么会冒出蒸汽？",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tokenizer', default='checkpoints/Qwen2.5-0.5B-Instruct/tokenizer.json')
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    pad = tok.token_to_id('<|endoftext|>')
    bos = tok.token_to_id('<|im_start|>')
    eos = tok.token_to_id('<|im_end|>')
    with open(args.checkpoint + '/student_full_vocab.pkl', 'rb') as f:
        ck = pickle.load(f)
    p = jax.tree_util.tree_map(jnp.asarray, ck['params'])
    cfg = LCMConfig(**ck['config'])
    questions = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt).ids[:48]
        questions.append(ids + [pad] * (48 - len(ids)))
    z = z_state(p, jnp.asarray(questions, dtype=jnp.int32), cfg)
    cur = jnp.full((len(PROMPTS), 1), bos, dtype=jnp.int32)
    generated = []
    for _ in range(48):
        x = jnp.pad(cur, ((0, 0), (0, 48 - cur.shape[1])), constant_values=pad)
        logits = compact_forward(p['gen_head'], z, x)
        token = jnp.argmax(logits[:, cur.shape[1] - 1], axis=-1)
        generated.append(token)
        cur = jnp.concatenate([cur, token[:, None]], axis=1)
    generated = jnp.stack(generated, axis=1).tolist()
    rows = []
    for prompt, seq in zip(PROMPTS, generated):
        seq = seq[:seq.index(eos)] if eos in seq else seq
        answer = tok.decode(seq)
        rows.append({'prompt': prompt, 'answer': answer})
        print(prompt, '=>', answer)
    with open(args.checkpoint + '/open_eval.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
