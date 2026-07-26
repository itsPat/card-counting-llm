from __future__ import annotations

from fractions import Fraction

import pytest

from blackjack import CasinoRules
from blackjack.oracle import (
    CardValue,
    Composition,
    DealerOutcome,
    PeekCondition,
    dealer_distribution,
    exhaustive_cdz_round_return_distribution,
    round_return_distribution,
)
from blackjack.oracle.native_dealer import native_dealer_distribution


def _native_probabilities(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    *,
    hit_soft_17: bool,
) -> dict[int, float]:
    return dict(
        native_dealer_distribution(
            composition.counts,
            tuple(CardValue).index(upcard),
            condition,
            hit_soft_17=hit_soft_17,
        )
    )


@pytest.mark.parametrize(
    ("upcard", "condition"),
    (
        (CardValue.ACE, PeekCondition.NONE),
        (CardValue.ACE, PeekCondition.NO_BLACKJACK),
        (CardValue.SIX, PeekCondition.NONE),
        (CardValue.TEN, PeekCondition.NO_BLACKJACK),
    ),
)
def test_native_dealer_matches_rational_reference(
    upcard: CardValue,
    condition: PeekCondition,
) -> None:
    composition = Composition.full_shoe().remove(upcard)
    native = _native_probabilities(
        composition,
        upcard,
        condition,
        hit_soft_17=True,
    )
    exact = dealer_distribution(composition, upcard, condition)
    outcome_codes = {
        DealerOutcome.SEVENTEEN: 17,
        DealerOutcome.EIGHTEEN: 18,
        DealerOutcome.NINETEEN: 19,
        DealerOutcome.TWENTY: 20,
        DealerOutcome.TWENTY_ONE: 21,
        DealerOutcome.BUST: 22,
        DealerOutcome.BLACKJACK: 23,
    }
    assert sum(native.values()) == pytest.approx(1.0, abs=1e-12)
    for item in exact.outcomes:
        assert native.get(outcome_codes[item.outcome], 0.0) == pytest.approx(
            float(item.probability),
            abs=1e-12,
        )


def test_native_dealer_respects_soft_seventeen_rule() -> None:
    composition = Composition.from_values(
        (
            CardValue.ACE,
            CardValue.SIX,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
        )
    )
    stand = _native_probabilities(
        composition,
        CardValue.ACE,
        PeekCondition.NONE,
        hit_soft_17=False,
    )
    hit = _native_probabilities(
        composition,
        CardValue.ACE,
        PeekCondition.NONE,
        hit_soft_17=True,
    )
    assert stand != hit


def test_float64_no_split_enumeration_matches_rational_reference() -> None:
    composition = Composition.from_values(
        (
            *((CardValue.ACE,) * 4),
            *((CardValue.SIX,) * 4),
            *((CardValue.TEN,) * 8),
        )
    )
    rules = CasinoRules(maximum_player_hands=1)
    exact = round_return_distribution(
        composition,
        rules,
        unseen_unavailable=0,
    ).distribution
    numeric = exhaustive_cdz_round_return_distribution(
        composition,
        rules,
        unseen_unavailable=0,
    ).distribution

    assert float(numeric.expected_profit) == pytest.approx(
        float(exact.expected_profit),
        abs=1e-12,
    )
    profits = {item.profit for item in exact.outcomes} | {
        item.profit for item in numeric.outcomes
    }
    for profit in profits:
        assert float(numeric.probability(profit)) == pytest.approx(
            float(exact.probability(profit)),
            abs=1e-11,
        )


def test_float64_pair_rich_round_distribution_is_normalized() -> None:
    composition = Composition.from_values(
        (
            CardValue.EIGHT,
            CardValue.EIGHT,
            CardValue.SIX,
            *((CardValue.TEN,) * 12),
        )
    )
    analysis = exhaustive_cdz_round_return_distribution(
        composition,
        unseen_unavailable=0,
    )
    assert (
        sum(
            (item.probability for item in analysis.distribution.outcomes),
            start=Fraction(0),
        )
        == 1
    )
    assert analysis.distribution.probability(2) > 0


def test_dealer_upcard_workers_preserve_the_numeric_distribution() -> None:
    composition = Composition.from_values(
        (
            *((CardValue.ACE,) * 4),
            *((CardValue.SIX,) * 4),
            *((CardValue.TEN,) * 8),
        )
    )
    rules = CasinoRules(maximum_player_hands=1)
    serial = exhaustive_cdz_round_return_distribution(
        composition,
        rules,
        unseen_unavailable=0,
        worker_count=1,
    )
    parallel = exhaustive_cdz_round_return_distribution(
        composition,
        rules,
        unseen_unavailable=0,
        worker_count=2,
    )
    assert parallel.distribution == serial.distribution


def test_numeric_worker_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="worker count"):
        exhaustive_cdz_round_return_distribution(
            Composition.full_shoe(),
            worker_count=0,
        )
