"""Tests for the tiny structured rule prior (few-shot control)."""
from train.rule_prior import induce, parse_prompt, predict


def test_parse_examples_and_query():
    pairs, query = parse_prompt("示例：7→8，8→9。问题：1→")
    assert pairs == [(7, 8), (8, 9)]
    assert query == 1


def test_one_example_induces_the_simplest_affine_rule():
    rule = induce([(7, 8)])
    assert rule is not None
    assert rule.apply(3) == 4
    assert rule.apply(20) == 21
    assert rule.describe() == "y=1x+1"


def test_two_examples_disambiguate_doubling():
    rule = induce([(3, 6), (5, 10)])
    assert rule is not None
    assert rule.apply(7) == 14


def test_square_and_reverse_are_in_the_space():
    square = induce([(3, 9), (4, 16), (5, 25)])
    assert square is not None
    assert square.apply(6) == 36

    reverse = induce([(12, 21), (34, 43)])
    assert reverse is not None
    assert reverse.apply(56) == 65


def test_predict_end_to_end():
    rule, value = predict("示例：2→4，3→6，4→8。问题：9→")
    assert rule is not None
    assert value == 18


def test_no_examples_returns_none():
    assert induce([]) is None
    assert predict("问题：5→")[1] is None
