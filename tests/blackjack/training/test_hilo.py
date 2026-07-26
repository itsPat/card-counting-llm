from __future__ import annotations

from blackjack.analysis import BetAction
from blackjack.engine import PlayerAction
from blackjack.oracle import CardValue
from blackjack.training.hilo import (
    HiLoBetRamp,
    floored_true_count,
    hi_lo_play_action,
    hi_lo_running_count,
)


def test_running_count_is_order_invariant_and_balanced() -> None:
    cards = (
        CardValue.TWO,
        CardValue.SEVEN,
        CardValue.ACE,
        CardValue.SIX,
        CardValue.NINE,
        CardValue.TEN,
    )
    assert hi_lo_running_count(cards) == 0
    assert hi_lo_running_count(tuple(reversed(cards))) == 0


def test_true_count_is_floored_for_positive_and_negative_counts() -> None:
    assert floored_true_count((CardValue.TWO,) * 6) == 1
    assert floored_true_count((CardValue.TEN,)) == -1


def test_bet_ramp_uses_predeclared_token_thresholds() -> None:
    ramp = HiLoBetRamp()
    assert ramp.action(1) is BetAction.MINIMUM
    assert ramp.action(2) is BetAction.LOW
    assert ramp.action(4) is BetAction.MEDIUM
    assert ramp.action(6) is BetAction.HIGH


def test_h17_indices_override_basic_strategy_on_both_sides() -> None:
    legal = (
        PlayerAction.HIT,
        PlayerAction.STAND,
        PlayerAction.DOUBLE,
    )
    eleven = (CardValue.SIX, CardValue.FIVE)
    assert (
        hi_lo_play_action(eleven, CardValue.ACE, legal, -2)
        is PlayerAction.HIT
    )
    assert (
        hi_lo_play_action(eleven, CardValue.ACE, legal, -1)
        is PlayerAction.DOUBLE
    )

    twelve = (CardValue.TEN, CardValue.TWO)
    assert (
        hi_lo_play_action(twelve, CardValue.SIX, legal, -4)
        is PlayerAction.HIT
    )
    assert (
        hi_lo_play_action(twelve, CardValue.SIX, legal, -3)
        is PlayerAction.STAND
    )


def test_fab_four_and_ten_splitting_respect_legality() -> None:
    surrender_legal = (
        PlayerAction.HIT,
        PlayerAction.STAND,
        PlayerAction.SURRENDER,
    )
    fifteen = (CardValue.TEN, CardValue.FIVE)
    assert (
        hi_lo_play_action(
            fifteen,
            CardValue.ACE,
            surrender_legal,
            -2,
        )
        is PlayerAction.HIT
    )
    assert (
        hi_lo_play_action(
            fifteen,
            CardValue.ACE,
            surrender_legal,
            -1,
        )
        is PlayerAction.SURRENDER
    )

    pair_legal = (
        PlayerAction.HIT,
        PlayerAction.STAND,
        PlayerAction.SPLIT,
    )
    tens = (CardValue.TEN, CardValue.TEN)
    assert (
        hi_lo_play_action(tens, CardValue.FIVE, pair_legal, 4)
        is PlayerAction.STAND
    )
    assert (
        hi_lo_play_action(tens, CardValue.FIVE, pair_legal, 5)
        is PlayerAction.SPLIT
    )
