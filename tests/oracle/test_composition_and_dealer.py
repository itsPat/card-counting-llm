from __future__ import annotations

from fractions import Fraction

import pytest

from blackjack import cards
from blackjack.oracle import (
    CARD_VALUES,
    CardValue,
    Composition,
    DealerOutcome,
    PeekCondition,
    cards_to_values,
    dealer_blackjack_probability,
    dealer_distribution,
    hidden_hole_draws,
)


def test_full_six_deck_composition_aggregates_ten_valued_ranks() -> None:
    composition = Composition.full_shoe()
    assert composition.total == 312
    assert all(composition.count(value) == 24 for value in CARD_VALUES[:-1])
    assert composition.count(CardValue.TEN) == 96


def test_cards_collapse_face_cards_to_ten() -> None:
    assert cards_to_values(cards("A", "10", "J", "Q", "K")) == (
        CardValue.ACE,
        CardValue.TEN,
        CardValue.TEN,
        CardValue.TEN,
        CardValue.TEN,
    )


def test_composition_removal_and_direct_draw_probabilities() -> None:
    composition = Composition.from_values((CardValue.ACE, CardValue.TWO, CardValue.TWO))
    draws = {draw.value: draw for draw in composition.draws()}
    assert draws[CardValue.ACE].probability == Fraction(1, 3)
    assert draws[CardValue.TWO].probability == Fraction(2, 3)
    assert draws[CardValue.TWO].composition.count(CardValue.TWO) == 1
    with pytest.raises(ValueError):
        composition.remove(CardValue.TEN)


def test_negative_peek_changes_hidden_hole_draw_probabilities() -> None:
    composition = Composition.from_values(
        (CardValue.ACE, CardValue.TWO, CardValue.THREE)
    )
    draws = {
        draw.value: draw.probability
        for draw in hidden_hole_draws(
            composition,
            CardValue.TEN,
            PeekCondition.NO_BLACKJACK,
        )
    }
    assert draws == {
        CardValue.ACE: Fraction(1, 2),
        CardValue.TWO: Fraction(1, 4),
        CardValue.THREE: Fraction(1, 4),
    }


def test_ten_upcard_distribution_conditions_on_negative_peek() -> None:
    composition = Composition.from_values((CardValue.ACE, CardValue.SEVEN))
    unchecked = dealer_distribution(composition, CardValue.TEN)
    checked = dealer_distribution(
        composition,
        CardValue.TEN,
        PeekCondition.NO_BLACKJACK,
    )
    assert unchecked.probability(DealerOutcome.BLACKJACK) == Fraction(1, 2)
    assert unchecked.probability(DealerOutcome.SEVENTEEN) == Fraction(1, 2)
    assert checked.probability(DealerOutcome.BLACKJACK) == 0
    assert checked.probability(DealerOutcome.SEVENTEEN) == 1


def test_dealer_hits_soft_seventeen_after_ace_peek() -> None:
    composition = Composition.from_values((CardValue.SIX, CardValue.TEN))
    checked = dealer_distribution(
        composition,
        CardValue.ACE,
        PeekCondition.NO_BLACKJACK,
    )
    assert checked.probability(DealerOutcome.SEVENTEEN) == 1


def test_dealer_draws_to_bust() -> None:
    composition = Composition.from_values((CardValue.TEN, CardValue.TEN, CardValue.TEN))
    distribution = dealer_distribution(composition, CardValue.SIX)
    assert distribution.probability(DealerOutcome.BUST) == 1


def test_dealer_blackjack_probability_is_exact() -> None:
    composition = Composition.from_values(
        (
            CardValue.TEN,
            CardValue.TEN,
            CardValue.SIX,
            CardValue.SEVEN,
        )
    )
    assert dealer_blackjack_probability(composition, CardValue.ACE) == Fraction(1, 2)
    assert dealer_blackjack_probability(composition, CardValue.SIX) == 0


def test_unseen_burn_card_is_marginalized_not_exposed() -> None:
    composition = Composition.from_values(
        (
            CardValue.ACE,
            CardValue.TWO,
            CardValue.THREE,
            CardValue.SEVEN,
            CardValue.TEN,
        )
    )
    draws = hidden_hole_draws(
        composition,
        CardValue.TEN,
        PeekCondition.NO_BLACKJACK,
        unseen_unavailable=1,
    )
    assert (
        sum(
            (draw.probability for draw in draws),
            start=Fraction(0),
        )
        == 1
    )
    distribution = dealer_distribution(
        composition,
        CardValue.TEN,
        PeekCondition.NO_BLACKJACK,
        unseen_unavailable=1,
    )
    assert distribution.probability(DealerOutcome.BLACKJACK) == 0
