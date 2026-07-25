from __future__ import annotations

from fractions import Fraction

import pytest

from blackjack import Hand, cards
from blackjack.settlement import (
    HandOutcome,
    InsuranceOutcome,
    settle_hand,
    settle_insurance,
)


@pytest.mark.parametrize(
    ("player_labels", "dealer_labels", "from_split", "outcome", "profit"),
    [
        (("A", "K"), ("10", "9"), False, HandOutcome.BLACKJACK, Fraction(15)),
        (("A", "K"), ("A", "Q"), False, HandOutcome.PUSH, Fraction(0)),
        (("A", "K"), ("10", "9"), True, HandOutcome.WIN, Fraction(10)),
        (("10", "9"), ("10", "8"), False, HandOutcome.WIN, Fraction(10)),
        (("10", "8"), ("10", "9"), False, HandOutcome.LOSS, Fraction(-10)),
        (("10", "8"), ("9", "9"), False, HandOutcome.PUSH, Fraction(0)),
        (("10", "9"), ("10", "8", "6"), False, HandOutcome.WIN, Fraction(10)),
        (("10", "8", "7"), ("10", "9"), False, HandOutcome.BUST, Fraction(-10)),
    ],
)
def test_hand_outcomes_and_exact_profit(
    player_labels: tuple[str, ...],
    dealer_labels: tuple[str, ...],
    from_split: bool,
    outcome: HandOutcome,
    profit: Fraction,
) -> None:
    result = settle_hand(
        hand_index=0,
        player=Hand(cards(*player_labels), from_split=from_split),
        wager=Fraction(10),
        dealer=Hand(cards(*dealer_labels)),
        surrendered=False,
    )
    assert result.outcome is outcome
    assert result.profit == profit


def test_surrender_returns_half_the_wager() -> None:
    result = settle_hand(
        hand_index=0,
        player=Hand(cards("10", "6")),
        wager=Fraction(15),
        dealer=None,
        surrendered=True,
    )
    assert result.outcome is HandOutcome.SURRENDER
    assert result.profit == Fraction(-15, 2)
    assert result.payout == Fraction(15, 2)


def test_insurance_cost_and_two_to_one_profit() -> None:
    won = settle_insurance(original_wager=Fraction(15), dealer_has_blackjack=True)
    lost = settle_insurance(original_wager=Fraction(15), dealer_has_blackjack=False)
    assert won.outcome is InsuranceOutcome.WON
    assert won.stake == Fraction(15, 2)
    assert won.profit == Fraction(15)
    assert won.payout == Fraction(45, 2)
    assert lost.outcome is InsuranceOutcome.LOST
    assert lost.profit == Fraction(-15, 2)
