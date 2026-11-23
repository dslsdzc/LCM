"""Small structured rule prior for few-shot integer patterns.

Humans do not fit a generic encoder to a few ``(x, y)`` pairs; they search a
small hypothesis space and prefer the simplest consistent rule. This module is
the explicit control for that prior: it parses example pairs from a prompt,
enumerates a deliberately tiny DSL, and picks the minimum-description-length
consistent rule.

It is not a language model and not part of the conformance path. Its purpose is
to separate "the task is hard" from "the current learner has no suitable
hypothesis space".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

_PAIR = re.compile(r"(\d+)\s*(?:→|->|=>)\s*(\d+)")
_QUERY = re.compile(r"问题\s*[:：]\s*(\d+)\s*(?:→|->|=>)")


@dataclass(frozen=True)
class Rule:
    """One candidate rule from the tiny DSL."""

    kind: str
    params: tuple[int, ...]

    @property
    def complexity(self) -> float:
        if self.kind == "constant":
            # A constant is a one-parameter rule, but it should not beat the
            # simplest affine explanation of a single non-constant pair.
            return 2.0
        if self.kind == "affine":
            a, b = self.params
            return float(abs(a - 1) + abs(b))
        if self.kind == "square":
            (b,) = self.params
            return float(abs(b) + 1)
        if self.kind == "reverse2":
            return 1.0
        raise ValueError(f"unknown rule kind {self.kind!r}")

    def apply(self, x: int) -> int:
        if self.kind == "constant":
            return self.params[0]
        if self.kind == "affine":
            a, b = self.params
            return a * x + b
        if self.kind == "square":
            (b,) = self.params
            return x * x + b
        if self.kind == "reverse2":
            if 10 <= x <= 99:
                return int(str(x)[::-1])
            return x
        raise ValueError(f"unknown rule kind {self.kind!r}")

    def describe(self) -> str:
        if self.kind == "constant":
            return f"y={self.params[0]}"
        if self.kind == "affine":
            a, b = self.params
            return f"y={a}x{b:+d}" if b else f"y={a}x"
        if self.kind == "square":
            b = self.params[0]
            return f"y=x^2{b:+d}" if b else "y=x^2"
        if self.kind == "reverse2":
            return "y=reverse_digits(x)"
        return self.kind


def parse_prompt(text: str) -> tuple[list[tuple[int, int]], int | None]:
    """Extract example pairs and the query value from a prompt.

    The pair regex only matches ``x→y`` with a target, so a trailing query
    ``问题：5→`` is not mistaken for an example.
    """
    pairs = [(int(a), int(b)) for a, b in _PAIR.findall(text)]
    match = _QUERY.search(text)
    query = int(match.group(1)) if match else None
    return pairs, query


def _candidates() -> Iterable[Rule]:
    for c in range(0, 10):
        yield Rule("constant", (c,))
    for a in range(1, 6):
        for b in range(-20, 21):
            yield Rule("affine", (a, b))
    for b in range(-20, 21):
        yield Rule("square", (b,))
    yield Rule("reverse2", ())


def induce(examples: Sequence[tuple[int, int]]) -> Rule | None:
    """Return the simplest rule with zero training error, or None."""
    if not examples:
        return None
    best: tuple[tuple, Rule] | None = None
    for rule in _candidates():
        mismatches = sum(rule.apply(x) != y for x, y in examples)
        if mismatches:
            continue
        key = (rule.complexity, rule.kind, rule.params)
        if best is None or key < best[0]:
            best = (key, rule)
    return None if best is None else best[1]


def predict(text: str) -> tuple[Rule | None, int | None]:
    """Parse a prompt, induce a rule, and apply it to the query."""
    examples, query = parse_prompt(text)
    rule = induce(examples)
    if rule is None or query is None:
        return rule, None
    return rule, rule.apply(query)
