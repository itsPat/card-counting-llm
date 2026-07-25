from __future__ import annotations

from fractions import Fraction

import pytest

from blackjack import CasinoRules, PlayerAction
from blackjack.oracle import (
    CardValue,
    Composition,
    OracleHand,
    PeekCondition,
    PlayerSituation,
    ReturnDistribution,
    evaluate_actions,
    expected_log_growth,
    kelly_recommendation,
    round_return_distribution,
)


def test_complete_round_distribution_is_normalized() -> None:
    all_tens = Composition.from_values((CardValue.TEN,) * 30)
    analysis = round_return_distribution(
        all_tens,
        unseen_unavailable=1,
    )
    assert analysis.distribution.probability(0) == 1
    assert analysis.expected_profit == 0


def test_round_distribution_combines_naturals_and_insurance_branches() -> None:
    ace_ten_shoe = Composition.from_values(
        (*((CardValue.ACE,) * 6), *((CardValue.TEN,) * 20))
    )
    analysis = round_return_distribution(
        ace_ten_shoe,
        CasinoRules(maximum_player_hands=1),
        unseen_unavailable=1,
    )
    assert (
        sum(
            (outcome.probability for outcome in analysis.distribution.outcomes),
            start=Fraction(0),
        )
        == 1
    )
    assert len(analysis.distribution.outcomes) > 1


def test_kelly_is_zero_without_an_edge() -> None:
    fair = ReturnDistribution.from_pairs(
        ((Fraction(-1), Fraction(1, 2)), (Fraction(1), Fraction(1, 2)))
    )
    recommendation = kelly_recommendation(fair)
    assert recommendation.full_kelly == 0
    assert recommendation.half_kelly == 0


def test_kelly_matches_the_closed_form_even_money_case() -> None:
    biased = ReturnDistribution.from_pairs(
        ((Fraction(-1), Fraction(2, 5)), (Fraction(1), Fraction(3, 5)))
    )
    recommendation = kelly_recommendation(biased)
    assert recommendation.full_kelly == pytest.approx(0.2)
    assert recommendation.half_kelly == pytest.approx(0.1)
    assert recommendation.expected_log_growth > 0
    assert expected_log_growth(biased, 0.2) == pytest.approx(
        recommendation.expected_log_growth
    )


def test_kelly_respects_multi_unit_split_and_double_exposure() -> None:
    distribution = ReturnDistribution.from_pairs(
        ((Fraction(-4), Fraction(1, 10)), (Fraction(1), Fraction(9, 10)))
    )
    recommendation = kelly_recommendation(distribution)
    assert 0 < recommendation.full_kelly < 0.25


def test_published_six_deck_values_match_to_six_decimal_places() -> None:
    # Independent reference:
    # https://wizardofodds.com/games/blackjack/appendix/9/6dh17r4/
    composition = Composition.full_shoe().remove(CardValue.TEN, 2).remove(CardValue.SIX)
    state = PlayerSituation(
        composition=composition,
        hand=OracleHand((CardValue.TEN, CardValue.SIX)),
        dealer_upcard=CardValue.TEN,
        peek_condition=PeekCondition.NO_BLACKJACK,
    )
    values = {
        evaluation.action: float(evaluation.expected_profit)
        for evaluation in evaluate_actions(state)
    }
    assert values[PlayerAction.STAND] == pytest.approx(-0.540954, abs=5e-7)
    assert values[PlayerAction.HIT] == pytest.approx(-0.534676, abs=5e-7)
    assert values[PlayerAction.DOUBLE] == pytest.approx(-1.069351, abs=5e-7)
