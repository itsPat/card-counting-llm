"""Exact dealer outcome probabilities with hidden-hole conditioning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import cache

from blackjack.engine.rules import FIXED_RULES, CasinoRules
from blackjack.oracle.composition import (
    CARD_VALUES,
    CardValue,
    Composition,
    Draw,
)


class PeekCondition(StrEnum):
    NONE = "none"
    NO_BLACKJACK = "no_blackjack"


class DealerOutcome(StrEnum):
    BLACKJACK = "blackjack"
    SEVENTEEN = "17"
    EIGHTEEN = "18"
    NINETEEN = "19"
    TWENTY = "20"
    TWENTY_ONE = "21"
    BUST = "bust"

    @property
    def total(self) -> int | None:
        return {
            DealerOutcome.SEVENTEEN: 17,
            DealerOutcome.EIGHTEEN: 18,
            DealerOutcome.NINETEEN: 19,
            DealerOutcome.TWENTY: 20,
            DealerOutcome.TWENTY_ONE: 21,
        }.get(self)


@dataclass(frozen=True, slots=True)
class DealerOutcomeProbability:
    outcome: DealerOutcome
    probability: Fraction


@dataclass(frozen=True, slots=True)
class DealerDistribution:
    outcomes: tuple[DealerOutcomeProbability, ...]

    def __post_init__(self) -> None:
        if sum(
            (item.probability for item in self.outcomes),
            start=Fraction(0),
        ) != Fraction(1):
            raise ValueError("dealer probabilities must sum to one")

    def probability(self, outcome: DealerOutcome) -> Fraction:
        return next(
            (item.probability for item in self.outcomes if item.outcome is outcome),
            Fraction(0),
        )


def _hole_is_allowed(
    hole: CardValue,
    upcard: CardValue,
    condition: PeekCondition,
) -> bool:
    if condition is PeekCondition.NONE:
        return True
    if upcard is CardValue.ACE:
        return hole is not CardValue.TEN
    if upcard is CardValue.TEN:
        return hole is not CardValue.ACE
    return True


def _eligible_hole_total(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
) -> int:
    return sum(
        count
        for value, count in composition
        if _hole_is_allowed(value, upcard, condition)
    )


def _hidden_hole_draws_without_unavailable(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
) -> tuple[Draw, ...]:
    if composition.total < 2:
        return ()
    eligible_holes = _eligible_hole_total(composition, upcard, condition)
    if eligible_holes == 0:
        raise ValueError("peek condition leaves no possible dealer hole card")
    denominator = eligible_holes * (composition.total - 1)
    draws: list[Draw] = []
    for value, count in composition:
        if count == 0:
            continue
        unavailable_as_hole = count if _hole_is_allowed(value, upcard, condition) else 0
        numerator = count * eligible_holes - unavailable_as_hole
        if numerator > 0:
            draws.append(
                Draw(
                    value=value,
                    probability=Fraction(numerator, denominator),
                    composition=composition.remove(value),
                )
            )
    return tuple(draws)


def hidden_hole_draws(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    unseen_unavailable: int = 0,
) -> tuple[Draw, ...]:
    """Next visible card while the hole and other unseen cards stay unavailable."""

    if unseen_unavailable < 0:
        raise ValueError("unseen unavailable card count cannot be negative")
    if composition.total <= unseen_unavailable + 1:
        return ()
    # Unconstrained hidden burn cards cancel after marginalization: assigning a
    # random card to an unavailable position and then dealing visible positions
    # produces the same ordered distribution as dealing those positions directly.
    return _hidden_hole_draws_without_unavailable(
        composition,
        upcard,
        condition,
    )


def dealer_blackjack_probability(
    composition: Composition,
    upcard: CardValue,
    unseen_unavailable: int = 0,
) -> Fraction:
    if composition.total <= unseen_unavailable:
        raise ValueError("the dealer needs a possible hole card")
    target = (
        CardValue.TEN
        if upcard is CardValue.ACE
        else CardValue.ACE
        if upcard is CardValue.TEN
        else None
    )
    if target is None:
        return Fraction(0)
    return Fraction(composition.count(target), composition.total)


def _hand_value(hard_total: int, aces: int) -> tuple[int, bool]:
    soft = aces > 0 and hard_total + 10 <= 21
    return (hard_total + 10 if soft else hard_total, soft)


def _outcome_for_total(total: int) -> DealerOutcome:
    return {
        17: DealerOutcome.SEVENTEEN,
        18: DealerOutcome.EIGHTEEN,
        19: DealerOutcome.NINETEEN,
        20: DealerOutcome.TWENTY,
        21: DealerOutcome.TWENTY_ONE,
    }[total]


@cache
def _play_dealer(
    composition: Composition,
    hard_total: int,
    aces: int,
    rules: CasinoRules,
) -> tuple[DealerOutcomeProbability, ...]:
    total, soft = _hand_value(hard_total, aces)
    if total > 21:
        return (DealerOutcomeProbability(DealerOutcome.BUST, Fraction(1)),)
    should_hit = total < 17 or (total == 17 and soft and rules.dealer_hits_soft_17)
    if not should_hit:
        return (DealerOutcomeProbability(_outcome_for_total(total), Fraction(1)),)
    draws = composition.draws()
    if not draws:
        raise ValueError("dealer must hit but no cards remain")
    merged: defaultdict[DealerOutcome, Fraction] = defaultdict(Fraction)
    for draw in draws:
        child = _play_dealer(
            draw.composition,
            hard_total + draw.value.hard_value,
            aces + int(draw.value is CardValue.ACE),
            rules,
        )
        for item in child:
            merged[item.outcome] += draw.probability * item.probability
    return tuple(
        DealerOutcomeProbability(outcome, probability)
        for outcome, probability in sorted(
            merged.items(), key=lambda item: item[0].value
        )
    )


@cache
def _dealer_distribution_without_unavailable(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    rules: CasinoRules,
) -> DealerDistribution:
    eligible_total = _eligible_hole_total(composition, upcard, condition)
    if eligible_total == 0:
        raise ValueError("no dealer hole card is compatible with this state")
    merged: defaultdict[DealerOutcome, Fraction] = defaultdict(Fraction)
    for hole in CARD_VALUES:
        count = composition.count(hole)
        if count == 0 or not _hole_is_allowed(hole, upcard, condition):
            continue
        hole_probability = Fraction(count, eligible_total)
        if (upcard is CardValue.ACE and hole is CardValue.TEN) or (
            upcard is CardValue.TEN and hole is CardValue.ACE
        ):
            merged[DealerOutcome.BLACKJACK] += hole_probability
            continue
        child = _play_dealer(
            composition.remove(hole),
            upcard.hard_value + hole.hard_value,
            int(upcard is CardValue.ACE) + int(hole is CardValue.ACE),
            rules,
        )
        for item in child:
            merged[item.outcome] += hole_probability * item.probability
    return DealerDistribution(
        tuple(
            DealerOutcomeProbability(outcome, probability)
            for outcome, probability in sorted(
                merged.items(),
                key=lambda item: item[0].value,
            )
        )
    )


@cache
def dealer_distribution(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition = PeekCondition.NONE,
    rules: CasinoRules = FIXED_RULES,
    unseen_unavailable: int = 0,
) -> DealerDistribution:
    """Average dealer play over every publicly possible hidden-card assignment."""

    if unseen_unavailable < 0:
        raise ValueError("unseen unavailable card count cannot be negative")
    if composition.total <= unseen_unavailable:
        raise ValueError("unavailable cards leave no dealer hole card")
    return _dealer_distribution_without_unavailable(
        composition,
        upcard,
        condition,
        rules,
    )


def dealer_cache_counts() -> tuple[tuple[str, int, int, int], ...]:
    """Expose dealer memoization counters for profiling."""

    play = _play_dealer.cache_info()
    distribution = _dealer_distribution_without_unavailable.cache_info()
    public_distribution = dealer_distribution.cache_info()
    return (
        ("dealer_play", play.hits, play.misses, play.currsize),
        (
            "dealer_distribution_core",
            distribution.hits,
            distribution.misses,
            distribution.currsize,
        ),
        (
            "dealer_distribution_public",
            public_distribution.hits,
            public_distribution.misses,
            public_distribution.currsize,
        ),
    )


def clear_dealer_caches() -> None:
    _play_dealer.cache_clear()
    _dealer_distribution_without_unavailable.cache_clear()
    dealer_distribution.cache_clear()
