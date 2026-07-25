"""Exact insurance probability, return distribution, and recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from blackjack.actions import InsuranceAction
from blackjack.oracle.composition import CardValue, Composition
from blackjack.oracle.dealer import dealer_blackjack_probability
from blackjack.oracle.distributions import ReturnDistribution


@dataclass(frozen=True, slots=True)
class InsuranceEvaluation:
    action: InsuranceAction
    dealer_blackjack_probability: Fraction
    distribution: ReturnDistribution

    @property
    def expected_profit(self) -> Fraction:
        return self.distribution.expected_profit


def evaluate_insurance(
    composition: Composition,
    dealer_upcard: CardValue,
    unseen_unavailable: int = 0,
) -> tuple[InsuranceEvaluation, InsuranceEvaluation]:
    if dealer_upcard is not CardValue.ACE:
        raise ValueError("insurance is offered only against a dealer Ace")
    blackjack_probability = dealer_blackjack_probability(
        composition,
        dealer_upcard,
        unseen_unavailable,
    )
    take = InsuranceEvaluation(
        action=InsuranceAction.TAKE,
        dealer_blackjack_probability=blackjack_probability,
        distribution=ReturnDistribution.from_pairs(
            (
                (Fraction(1), blackjack_probability),
                (Fraction(-1, 2), 1 - blackjack_probability),
            )
        ),
    )
    decline = InsuranceEvaluation(
        action=InsuranceAction.DECLINE,
        dealer_blackjack_probability=blackjack_probability,
        distribution=ReturnDistribution.constant(0),
    )
    return take, decline


def optimal_insurance(
    composition: Composition,
    dealer_upcard: CardValue,
    unseen_unavailable: int = 0,
) -> InsuranceEvaluation:
    take, decline = evaluate_insurance(
        composition,
        dealer_upcard,
        unseen_unavailable,
    )
    return take if take.expected_profit > 0 else decline
