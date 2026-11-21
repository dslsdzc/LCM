#!/usr/bin/env python3
"""Build a tiny Chinese chat-format corpus for the CPU dialogue pilot.

The goal is *conversation behaviour*, not knowledge: greetings, self-intro,
short follow-ups, "I don't know", thanks and goodbye. Each line is one
multi-turn document (the training iterator samples windows inside a document),
so role markers must appear in the text itself.

Usage:
    python scripts/build_dialogue_corpus.py \
        --output-dir /home/DslsDZC/LCM/data_chat --n 1200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.data import build_corpus  # noqa: E402
from train.tokenizer_spec import resolve_tokenizer_spec  # noqa: E402

NAMES = ["LCM", "小格", "格子", "小晶"]
GREETINGS = ["你好", "你好呀", "嗨", "早上好", "晚上好"]
GREET_REPLIES = [
    "你好，我是{name}。",
    "你好，我是{name}，很高兴见到你。",
    "嗨，我是{name}。",
]
HOW_ARE_YOU = ["你今天怎么样？", "你最近好吗？", "你还好吗？"]
HOW_REPLIES = ["我很好，谢谢。你呢？", "我很好，谢谢你问。", "还不错，谢谢。"]
NAMES_Q = ["你叫什么名字？", "你的名字是什么？", "你是谁？"]
NAMES_A = ["我叫{name}。", "我是{name}。", "你可以叫我{name}。"]
CAP_Q = ["你会做什么？", "你能做什么？", "你会聊天吗？"]
CAP_A = [
    "我现在只能进行简单对话，还在学习。",
    "我还在学习对话，只能聊一些简单的话题。",
    "我会一点点对话，但知道得还不多。",
]
LIKE_Q = ["你喜欢什么？", "你有什么爱好？"]
LIKE_A = ["我还在学习，暂时没有特别的爱好。", "我喜欢听你说话。"]
UNKNOWN_Q = ["你知道宇宙的尽头是什么吗？", "你知道明天会发生什么吗？", "你会写代码吗？"]
UNKNOWN_A = ["我不知道，我还在学习。", "这个我还不懂，抱歉。", "我暂时回答不了这个问题。"]
THANKS = ["谢谢你", "谢谢", "多谢"]
THANKS_A = ["不客气。", "不用谢。", "很高兴能帮到你。"]
SORRY = ["对不起", "抱歉"]
SORRY_A = ["没关系。", "不要紧。"]
BYE = ["再见", "拜拜", "下次见"]
BYE_A = ["再见，祝你愉快。", "拜拜，下次聊。", "再见。"]


def _pair(rng: random.Random, user: str, replies: list[str], name: str) -> tuple[str, str]:
    return user, rng.choice(replies).format(name=name)


def make_dialogue(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    turns: list[tuple[str, str]] = []

    turns.append(_pair(rng, rng.choice(GREETINGS), GREET_REPLIES, name))
    # 1–3 middle exchanges.
    middle = rng.randint(1, 3)
    pools = [
        (HOW_ARE_YOU, HOW_REPLIES),
        (NAMES_Q, NAMES_A),
        (CAP_Q, CAP_A),
        (LIKE_Q, LIKE_A),
        (UNKNOWN_Q, UNKNOWN_A),
        (THANKS, THANKS_A),
        (SORRY, SORRY_A),
    ]
    rng.shuffle(pools)
    for user_pool, reply_pool in pools[:middle]:
        turns.append(_pair(rng, rng.choice(user_pool), reply_pool, name))
    # A light follow-up makes the dialogue feel less like a Q&A form.
    if rng.random() < 0.5:
        turns.append((rng.choice(["我很好。", "我明白了。", "原来如此。"]),
                      rng.choice(["很高兴听到。", "好的。", "嗯，我记住了。"])))
    turns.append(_pair(rng, rng.choice(BYE), BYE_A, name))

    return " ".join(f"user: {u} assistant: {a}" for u, a in turns)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="/home/DslsDZC/LCM/data_chat")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--tokenizer", default="/home/DslsDZC/LCM/data/tokenizer.json")
    p.add_argument("--data-root", default="/home/DslsDZC/data/lcm")
    p.add_argument("--name", default="dialogue")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    lines = [make_dialogue(rng) for _ in range(max(1, args.n))]
    text_path = out / "dialogue.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[chat] wrote {len(lines)} dialogues -> {text_path}")

    spec = resolve_tokenizer_spec(
        path=args.tokenizer,
        document_separator="[EOS]",
        model_vocab_size=30000,
        tokenizer_id="bpe30k",
    )
    data_path = out / f"{args.name}.dat"
    shape_path = out / f"{args.name}_shape.json"
    spans_path = out / f"{args.name}_docs.npy"
    meta = build_corpus(
        str(text_path), spec, str(data_path), str(shape_path),
        str(spans_path), max_tokens=0,
    )
    print(f"[chat] tokens={meta['n_tokens']} docs={meta['n_docs']} "
          f"dtype={meta['dtype']} separator={meta['separator_id']}")
    print(f"[chat] data  -> {data_path}")
    print(f"[chat] shape -> {shape_path}")
    print(f"[chat] spans -> {spans_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
