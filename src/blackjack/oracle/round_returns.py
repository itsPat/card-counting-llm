"""Complete optimal round-return distribution from a pre-deal composition."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from multiprocessing import get_context

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


@dataclass(frozen=True, slots=True)
class _InitialBranchTask:
    first_player: CardValue
    first_probability: Fraction
    composition: Composition
    rules: CasinoRules
    unseen_unavailable: int


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
    worker_count: int = 1,
) -> RoundReturnAnalysis:
    """Return optimal net-profit probabilities before the initial visible deal.

    The default unavailable count represents the one unseen burn card. The
    composition therefore contains every card not exposed to the player,
    including that burn card.
    """

    if composition.total <= unseen_unavailable + 3:
        raise ValueError("composition is too small for an initial round")
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    tasks = tuple(
        _InitialBranchTask(
            first_player=draw.value,
            first_probability=draw.probability,
            composition=draw.composition,
            rules=rules,
            unseen_unavailable=unseen_unavailable,
        )
        for draw in composition.draws()
    )
    effective_workers = min(worker_count, len(tasks))
    if effective_workers <= 1:
        grouped = tuple(_initial_branches(task) for task in tasks)
    else:
        pool = get_context("spawn").Pool(processes=effective_workers)
        try:
            grouped = tuple(pool.map(_initial_branches, tasks))
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
    branches = tuple(branch for group in grouped for branch in group)
    return RoundReturnAnalysis(ReturnDistribution.mixture(branches))


def _initial_branches(
    task: _InitialBranchTask,
) -> tuple[tuple[Fraction, ReturnDistribution], ...]:
    branches: list[tuple[Fraction, ReturnDistribution]] = []
    for dealer_upcard in task.composition.draws():
        for second_player in dealer_upcard.composition.draws():
            probability = (
                task.first_probability
                * dealer_upcard.probability
                * second_player.probability
            )
            distribution = _post_visible_distribution(
                second_player.composition,
                canonical_values((task.first_player, second_player.value)),
                dealer_upcard.value,
                task.rules,
                task.unseen_unavailable,
            )
            branches.append((probability, distribution))
    return tuple(branches)


def round_return_cache_counts() -> tuple[tuple[str, int, int, int], ...]:
    info = _post_visible_distribution.cache_info()
    return (("round_post_visible", info.hits, info.misses, info.currsize),)


def clear_round_return_caches() -> None:
    _post_visible_distribution.cache_clear()
