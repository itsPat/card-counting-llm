"""Complete optimal round-return distribution from a pre-deal composition."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache

from blackjack.engine.actions import InsuranceAction
from blackjack.engine.rules import FIXED_RULES, CasinoRules
from blackjack.oracle.composition import (
    CardValue,
    Composition,
    canonical_values,
)
from blackjack.oracle.dealer import (
    PeekCondition,
    dealer_blackjack_probability,
)
from blackjack.oracle.distributions import ReturnDistribution
from blackjack.oracle.insurance import optimal_insurance
from blackjack.oracle.player import (
    OracleHand,
    PlayerSituation,
    optimal_return_distribution,
)


@dataclass(frozen=True, slots=True)
class RoundReturnAnalysis:
    distribution: ReturnDistribution

    @property
    def expected_profit(self) -> Fraction:
        return self.distribution.expected_profit


@cache
def _post_visible_distribution(
    composition: Composition,
    player_cards: tuple[CardValue, ...],
    dealer_upcard: CardValue,
    rules: CasinoRules,
    unseen_unavailable: int,
) -> ReturnDistribution:
    hand = OracleHand(player_cards)
    if dealer_upcard in (CardValue.ACE, CardValue.TEN):
        blackjack_probability = dealer_blackjack_probability(
            composition,
            dealer_upcard,
            unseen_unavailable,
        )
        take_insurance = (
            dealer_upcard is CardValue.ACE
            and optimal_insurance(
                composition,
                dealer_upcard,
                unseen_unavailable,
            ).action
            is InsuranceAction.TAKE
        )
        dealer_blackjack_profit = (
            Fraction(0) if hand.is_natural_blackjack else Fraction(-1)
        )
        if take_insurance:
            dealer_blackjack_profit += 1

        if hand.is_natural_blackjack:
            no_blackjack = ReturnDistribution.constant(rules.blackjack_profit)
        else:
            no_blackjack = optimal_return_distribution(
                PlayerSituation(
                    composition=composition,
                    hand=hand,
                    dealer_upcard=dealer_upcard,
                    peek_condition=PeekCondition.NO_BLACKJACK,
                    rules=rules,
                    unseen_unavailable=unseen_unavailable,
                )
            )
        if take_insurance:
            no_blackjack = no_blackjack.shifted(Fraction(-1, 2))
        return ReturnDistribution.mixture(
            (
                (
                    blackjack_probability,
                    ReturnDistribution.constant(dealer_blackjack_profit),
                ),
                (1 - blackjack_probability, no_blackjack),
            )
        )

    if hand.is_natural_blackjack:
        return ReturnDistribution.constant(rules.blackjack_profit)
    return optimal_return_distribution(
        PlayerSituation(
            composition=composition,
            hand=hand,
            dealer_upcard=dealer_upcard,
            peek_condition=PeekCondition.NONE,
            rules=rules,
            unseen_unavailable=unseen_unavailable,
        )
    )


def round_return_distribution(
    composition: Composition,
    rules: CasinoRules = FIXED_RULES,
    *,
    unseen_unavailable: int = 1,
) -> RoundReturnAnalysis:
    """Return optimal net-profit probabilities before the initial visible deal.

    The default unavailable count represents the one unseen burn card. The
    composition therefore contains every card not exposed to the player,
    including that burn card.
    """

    if composition.total <= unseen_unavailable + 3:
        raise ValueError("composition is too small for an initial round")
    branches: list[tuple[Fraction, ReturnDistribution]] = []
    for first_player in composition.draws():
        for dealer_upcard in first_player.composition.draws():
            for second_player in dealer_upcard.composition.draws():
                probability = (
                    first_player.probability
                    * dealer_upcard.probability
                    * second_player.probability
                )
                distribution = _post_visible_distribution(
                    second_player.composition,
                    canonical_values((first_player.value, second_player.value)),
                    dealer_upcard.value,
                    rules,
                    unseen_unavailable,
                )
                branches.append((probability, distribution))
    return RoundReturnAnalysis(ReturnDistribution.mixture(branches))
