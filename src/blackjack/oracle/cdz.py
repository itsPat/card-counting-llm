"""Fast exhaustive round enumeration under an explicit CDZ- split policy.

The casino engine still permits resplitting to four hands.  This production
bet oracle intentionally evaluates the computationally tractable CDZ-,
no-resplit approximation described in the project methodology:

* the composition-dependent non-split policy is fixed before the round;
* that same policy is applied to both post-split hands; and
* a pair produced after the first split is played without resplitting.

The implementation enumerates every compatible card sequence with float64
probabilities.  The rational CDP oracle remains the reference implementation
for small-state verification.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import cache
from math import fsum, isclose
from multiprocessing import get_context

from blackjack.engine.actions import PlayerAction
from blackjack.engine.rules import FIXED_RULES, CasinoRules
from blackjack.oracle.composition import CARD_VALUES, Composition
from blackjack.oracle.dealer import PeekCondition
from blackjack.oracle.distributions import ReturnDistribution
from blackjack.oracle.native_dealer import (
    NativeSplitEndpoint,
    ensure_native_dealer_kernel,
    native_dealer_distribution,
    native_split_distribution,
)

type _Counts = tuple[int, ...]
type _NumericDistribution = tuple[tuple[int, float], ...]
type _DealerDistribution = tuple[tuple[int, float], ...]

_DEALER_BUST = 22
_DEALER_BLACKJACK = 23
_ACE_INDEX = 0
_TEN_INDEX = 9
_PROBABILITY_DENOMINATOR_LIMIT = 10**12


class BetSplitPolicy(StrEnum):
    """Named split-policy concessions supported by the production bet oracle."""

    CDZ_MINUS_NO_RESPLIT = "cdz_minus_no_resplit"


@dataclass(frozen=True, slots=True)
class ExhaustiveNumericRoundAnalysis:
    """A complete round distribution produced by float64 enumeration."""

    distribution: ReturnDistribution
    split_policy: BetSplitPolicy = BetSplitPolicy.CDZ_MINUS_NO_RESPLIT

    @property
    def expected_profit(self) -> Fraction:
        return self.distribution.expected_profit


@dataclass(frozen=True, slots=True)
class _Hand:
    cards: _Counts
    wager_half_units: int = 2
    from_split: bool = False
    split_aces: bool = False
    can_double: bool = True
    can_surrender: bool = True

    @property
    def card_count(self) -> int:
        return sum(self.cards)

    @property
    def total(self) -> int:
        hard = sum(
            count * CARD_VALUES[index].hard_value
            for index, count in enumerate(self.cards)
        )
        return hard + 10 if self.cards[_ACE_INDEX] and hard + 10 <= 21 else hard

    @property
    def is_soft(self) -> bool:
        hard = sum(
            count * CARD_VALUES[index].hard_value
            for index, count in enumerate(self.cards)
        )
        return bool(self.cards[_ACE_INDEX] and hard + 10 <= 21)

    @property
    def is_bust(self) -> bool:
        return self.total > 21

    @property
    def is_natural(self) -> bool:
        return not self.from_split and self.card_count == 2 and self.total == 21

    @property
    def pair_index(self) -> int | None:
        if self.card_count != 2:
            return None
        return next(
            (index for index, count in enumerate(self.cards) if count == 2),
            None,
        )

    def add(self, card_index: int) -> _Hand:
        return _Hand(
            cards=_add_card(self.cards, card_index),
            wager_half_units=self.wager_half_units,
            from_split=self.from_split,
            split_aces=self.split_aces,
            can_double=False,
            can_surrender=False,
        )

    def add_split_card(self, card_index: int) -> _Hand:
        return _Hand(
            cards=_add_card(self.cards, card_index),
            wager_half_units=self.wager_half_units,
            from_split=True,
            split_aces=self.split_aces,
            can_double=self.can_double,
            can_surrender=False,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedHand:
    total: int
    wager_half_units: int
    is_natural: bool
    is_bust: bool
    surrendered: bool = False

    @classmethod
    def from_hand(
        cls,
        hand: _Hand,
        *,
        surrendered: bool = False,
    ) -> _ResolvedHand:
        return cls(
            total=hand.total,
            wager_half_units=hand.wager_half_units,
            is_natural=hand.is_natural,
            is_bust=hand.is_bust,
            surrendered=surrendered,
        )


@dataclass(frozen=True, slots=True)
class _UpcardTask:
    base_counts: _Counts
    remaining_counts: _Counts
    upcard_index: int
    probability: float
    rules: CasinoRules


def exhaustive_cdz_round_return_distribution(
    composition: Composition,
    rules: CasinoRules = FIXED_RULES,
    *,
    unseen_unavailable: int = 1,
    worker_count: int = 1,
) -> ExhaustiveNumericRoundAnalysis:
    """Enumerate the pre-deal return distribution under CDZ-/no-resplit play."""

    if composition.total <= unseen_unavailable + 3:
        raise ValueError("composition is too small for an initial round")
    if worker_count <= 0:
        raise ValueError("worker count must be positive")

    tasks = tuple(
        _UpcardTask(
            base_counts=composition.counts,
            remaining_counts=_remove_card(composition.counts, upcard_index),
            upcard_index=upcard_index,
            probability=count / composition.total,
            rules=rules,
        )
        for upcard_index, count in enumerate(composition.counts)
        if count
    )
    ensure_native_dealer_kernel()
    effective_workers = min(worker_count, len(tasks))
    if effective_workers <= 1:
        solver = _CDZSolver(composition.counts, rules)
        try:
            branches = tuple(
                (
                    task.probability,
                    solver.solve_upcard(task.remaining_counts, task.upcard_index),
                )
                for task in tasks
            )
        finally:
            solver.clear_caches()
    else:
        pool = get_context("spawn").Pool(processes=effective_workers)
        try:
            solved = tuple(pool.map(_solve_upcard_task, tasks))
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()
        branches = tuple(
            (task.probability, distribution)
            for task, distribution in zip(tasks, solved, strict=True)
        )
    numeric = _mixture(branches)
    return ExhaustiveNumericRoundAnalysis(_to_return_distribution(numeric))


def _solve_upcard_task(task: _UpcardTask) -> _NumericDistribution:
    solver = _CDZSolver(task.base_counts, task.rules)
    try:
        return solver.solve_upcard(task.remaining_counts, task.upcard_index)
    finally:
        solver.clear_caches()


class _CDZSolver:
    """One process-local solver with caches shared inside an upcard partition."""

    __slots__ = ("_base_counts", "_rules")

    def __init__(self, base_counts: _Counts, rules: CasinoRules) -> None:
        self._base_counts = base_counts
        self._rules = rules

    def clear_caches(self) -> None:
        """Release one-shot memoized states after a label finishes."""

        self._solve_visible.cache_clear()
        self._play_initial.cache_clear()
        self._play_fixed.cache_clear()
        self._split_endpoints.cache_clear()
        self._policy_action.cache_clear()
        self._policy_optimal_ev.cache_clear()
        self._policy_action_ev.cache_clear()
        self._settle.cache_clear()
        self._settlement_ev.cache_clear()
        self._dealer_distribution.cache_clear()

    def solve_upcard(
        self,
        remaining: _Counts,
        upcard_index: int,
    ) -> _NumericDistribution:
        branches: list[tuple[float, _NumericDistribution]] = []
        total = sum(remaining)
        for first_index, first_count in enumerate(remaining):
            if not first_count:
                continue
            after_first = _remove_card(remaining, first_index)
            first_probability = first_count / total
            second_total = total - 1
            for second_index, second_count in enumerate(after_first):
                if not second_count:
                    continue
                visible_composition = _remove_card(after_first, second_index)
                hand = _Hand(_hand_counts(first_index, second_index))
                probability = first_probability * second_count / second_total
                branches.append(
                    (
                        probability,
                        self._solve_visible(
                            visible_composition,
                            hand,
                            upcard_index,
                        ),
                    )
                )
        return _mixture(branches)

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _solve_visible(
        self,
        composition: _Counts,
        hand: _Hand,
        upcard_index: int,
    ) -> _NumericDistribution:
        upcard_is_ace = upcard_index == _ACE_INDEX
        upcard_is_ten = upcard_index == _TEN_INDEX
        if upcard_is_ace or upcard_is_ten:
            target = _TEN_INDEX if upcard_is_ace else _ACE_INDEX
            blackjack_probability = composition[target] / sum(composition)
            take_insurance = upcard_is_ace and blackjack_probability > (1.0 / 3.0)
            blackjack_profit = 0 if hand.is_natural else -2
            if take_insurance:
                blackjack_profit += 2

            if hand.is_natural:
                no_blackjack = _constant(3)
            else:
                no_blackjack = self._play_initial(
                    composition,
                    hand,
                    upcard_index,
                    PeekCondition.NO_BLACKJACK,
                )
            if take_insurance:
                no_blackjack = _shift(no_blackjack, -1)
            return _mixture(
                (
                    (blackjack_probability, _constant(blackjack_profit)),
                    (1.0 - blackjack_probability, no_blackjack),
                )
            )

        if hand.is_natural:
            return _constant(3)
        return self._play_initial(
            composition,
            hand,
            upcard_index,
            PeekCondition.NONE,
        )

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _play_initial(
        self,
        composition: _Counts,
        hand: _Hand,
        upcard_index: int,
        condition: PeekCondition,
    ) -> _NumericDistribution:
        actions = [
            PlayerAction.HIT,
            PlayerAction.STAND,
            PlayerAction.DOUBLE,
        ]
        if hand.pair_index is not None and self._rules.maximum_player_hands >= 2:
            actions.append(PlayerAction.SPLIT)
        if hand.can_surrender and self._rules.late_surrender:
            actions.append(PlayerAction.SURRENDER)
        evaluations = tuple(
            (
                action,
                self._initial_action_distribution(
                    composition,
                    hand,
                    upcard_index,
                    condition,
                    action,
                ),
            )
            for action in actions
        )
        return max(evaluations, key=lambda item: _expected(item[1]))[1]

    def _initial_action_distribution(
        self,
        composition: _Counts,
        hand: _Hand,
        upcard_index: int,
        condition: PeekCondition,
        action: PlayerAction,
    ) -> _NumericDistribution:
        if action is PlayerAction.STAND:
            return self._settle(
                composition,
                upcard_index,
                condition,
                (_ResolvedHand.from_hand(hand),),
            )
        if action is PlayerAction.SURRENDER:
            return _constant(-1)
        if action is PlayerAction.HIT:
            return _mixture(
                (
                    probability,
                    self._play_fixed(
                        child_composition,
                        hand.add(card_index),
                        upcard_index,
                        condition,
                    ),
                )
                for card_index, probability, child_composition in _visible_draws(
                    composition,
                    upcard_index,
                    condition,
                )
            )
        if action is PlayerAction.DOUBLE:
            return _mixture(
                (
                    probability,
                    self._settle(
                        child_composition,
                        upcard_index,
                        condition,
                        (
                            _ResolvedHand.from_hand(
                                _Hand(
                                    cards=hand.add(card_index).cards,
                                    wager_half_units=4,
                                    can_double=False,
                                    can_surrender=False,
                                )
                            ),
                        ),
                    ),
                )
                for card_index, probability, child_composition in _visible_draws(
                    composition,
                    upcard_index,
                    condition,
                )
            )
        return self._split_once(
            composition,
            hand,
            upcard_index,
            condition,
        )

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _play_fixed(
        self,
        composition: _Counts,
        hand: _Hand,
        upcard_index: int,
        condition: PeekCondition,
    ) -> _NumericDistribution:
        if hand.is_bust or hand.total >= 21 or hand.split_aces:
            return self._settle(
                composition,
                upcard_index,
                condition,
                (_ResolvedHand.from_hand(hand),),
            )
        action = self._policy_action(hand, upcard_index, condition)
        if action is PlayerAction.STAND:
            return self._settle(
                composition,
                upcard_index,
                condition,
                (_ResolvedHand.from_hand(hand),),
            )
        if action is PlayerAction.DOUBLE:
            return _mixture(
                (
                    probability,
                    self._settle(
                        child_composition,
                        upcard_index,
                        condition,
                        (
                            _ResolvedHand.from_hand(
                                _Hand(
                                    cards=hand.add(card_index).cards,
                                    wager_half_units=4,
                                    from_split=hand.from_split,
                                    can_double=False,
                                    can_surrender=False,
                                )
                            ),
                        ),
                    ),
                )
                for card_index, probability, child_composition in _visible_draws(
                    composition,
                    upcard_index,
                    condition,
                )
            )
        return _mixture(
            (
                probability,
                self._play_fixed(
                    child_composition,
                    hand.add(card_index),
                    upcard_index,
                    condition,
                ),
            )
            for card_index, probability, child_composition in _visible_draws(
                composition,
                upcard_index,
                condition,
            )
        )

    def _split_once(
        self,
        composition: _Counts,
        hand: _Hand,
        upcard_index: int,
        condition: PeekCondition,
    ) -> _NumericDistribution:
        pair_index = hand.pair_index
        if pair_index is None:
            raise AssertionError("split action requires a pair")
        endpoints = self._split_endpoints(
            pair_index,
            upcard_index,
            condition,
        )
        distribution = native_split_distribution(
            composition,
            pair_index,
            upcard_index,
            condition,
            tuple(
                NativeSplitEndpoint(
                    cards=endpoint.cards,
                    wager_half_units=endpoint.wager_half_units,
                    multiplicity=multiplicity,
                )
                for endpoint, multiplicity in endpoints
            ),
            hit_soft_17=self._rules.dealer_hits_soft_17,
        )
        if not isclose(
            fsum(probability for _, probability in distribution),
            1.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError("native split distribution is not normalized")
        return distribution

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _split_endpoints(
        self,
        pair_index: int,
        upcard_index: int,
        condition: PeekCondition,
    ) -> tuple[tuple[_Hand, int], ...]:
        base = _Hand(
            cards=_single_card_counts(pair_index),
            from_split=True,
            split_aces=pair_index == _ACE_INDEX,
            can_double=self._rules.double_after_split,
            can_surrender=False,
        )
        endpoints: dict[_Hand, int] = {}

        def finish(endpoint: _Hand, multiplicity: int) -> None:
            endpoints[endpoint] = endpoints.get(endpoint, 0) + multiplicity

        def visit(active: _Hand, multiplicity: int) -> None:
            if active.is_bust or active.total >= 21 or active.split_aces:
                finish(active, multiplicity)
                return
            action = self._policy_action(active, upcard_index, condition)
            if action is PlayerAction.STAND:
                finish(active, multiplicity)
                return
            available = self._policy_composition(active, upcard_index)
            if action is PlayerAction.DOUBLE:
                for card_index, count in enumerate(available):
                    if count:
                        finish(
                            _Hand(
                                cards=active.add(card_index).cards,
                                wager_half_units=4,
                                from_split=True,
                                can_double=False,
                                can_surrender=False,
                            ),
                            multiplicity,
                        )
                return
            for card_index, count in enumerate(available):
                if count:
                    visit(active.add(card_index), multiplicity)

        initial_available = self._policy_composition(base, upcard_index)
        for card_index, count in enumerate(initial_available):
            if count:
                visit(base.add_split_card(card_index), 1)
        return tuple(
            sorted(endpoints.items(), key=lambda item: _hand_sort_key(item[0]))
        )

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _policy_action(
        self,
        hand: _Hand,
        upcard_index: int,
        condition: PeekCondition,
    ) -> PlayerAction:
        """Choose from non-split EVs using only this hand's visible cards."""

        actions = [PlayerAction.HIT, PlayerAction.STAND]
        if hand.card_count == 2 and hand.can_double:
            actions.append(PlayerAction.DOUBLE)
        return max(
            actions,
            key=lambda action: self._policy_action_ev(
                hand,
                upcard_index,
                condition,
                action,
            ),
        )

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _policy_optimal_ev(
        self,
        hand: _Hand,
        upcard_index: int,
        condition: PeekCondition,
    ) -> float:
        if hand.is_bust or hand.total >= 21:
            remaining = self._policy_composition(hand, upcard_index)
            return self._settlement_ev(
                remaining,
                upcard_index,
                condition,
                (_ResolvedHand.from_hand(hand),),
            )
        action = self._policy_action(hand, upcard_index, condition)
        return self._policy_action_ev(hand, upcard_index, condition, action)

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _policy_action_ev(
        self,
        hand: _Hand,
        upcard_index: int,
        condition: PeekCondition,
        action: PlayerAction,
    ) -> float:
        composition = self._policy_composition(hand, upcard_index)
        if action is PlayerAction.STAND:
            return self._settlement_ev(
                composition,
                upcard_index,
                condition,
                (_ResolvedHand.from_hand(hand),),
            )
        if action is PlayerAction.DOUBLE:
            return fsum(
                probability
                * self._settlement_ev(
                    child_composition,
                    upcard_index,
                    condition,
                    (
                        _ResolvedHand.from_hand(
                            _Hand(
                                cards=hand.add(card_index).cards,
                                wager_half_units=4,
                                from_split=hand.from_split,
                                can_double=False,
                                can_surrender=False,
                            )
                        ),
                    ),
                )
                for card_index, probability, child_composition in _visible_draws(
                    composition,
                    upcard_index,
                    condition,
                )
            )
        return fsum(
            probability
            * self._policy_optimal_ev(
                hand.add(card_index),
                upcard_index,
                condition,
            )
            for card_index, probability, _ in _visible_draws(
                composition,
                upcard_index,
                condition,
            )
        )

    def _policy_composition(
        self,
        hand: _Hand,
        upcard_index: int,
    ) -> _Counts:
        counts = list(self._base_counts)
        counts[upcard_index] -= 1
        for index, number in enumerate(hand.cards):
            counts[index] -= number
        if any(count < 0 for count in counts):
            raise ValueError("policy hand is incompatible with the base composition")
        return tuple(counts)

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _settle(
        self,
        composition: _Counts,
        upcard_index: int,
        condition: PeekCondition,
        hands: tuple[_ResolvedHand, ...],
    ) -> _NumericDistribution:
        if all(hand.surrendered or hand.is_bust for hand in hands):
            return _constant(sum(_fixed_profit(hand) for hand in hands))
        dealer = self._dealer_distribution(
            composition,
            upcard_index,
            condition,
        )
        return _from_weighted_profits(
            (
                sum(_profit_against_dealer(hand, outcome) for hand in hands),
                probability,
            )
            for outcome, probability in dealer
        )

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _settlement_ev(
        self,
        composition: _Counts,
        upcard_index: int,
        condition: PeekCondition,
        hands: tuple[_ResolvedHand, ...],
    ) -> float:
        return _expected(self._settle(composition, upcard_index, condition, hands))

    @cache  # noqa: B019 - clear_caches releases this one-shot solver cache.
    def _dealer_distribution(
        self,
        composition: _Counts,
        upcard_index: int,
        condition: PeekCondition,
    ) -> _DealerDistribution:
        return native_dealer_distribution(
            composition,
            upcard_index,
            condition,
            hit_soft_17=self._rules.dealer_hits_soft_17,
        )


def _hole_allowed(
    hole_index: int,
    upcard_index: int,
    condition: PeekCondition,
) -> bool:
    if condition is PeekCondition.NONE:
        return True
    if upcard_index == _ACE_INDEX:
        return hole_index != _TEN_INDEX
    if upcard_index == _TEN_INDEX:
        return hole_index != _ACE_INDEX
    return True


def _visible_draws(
    composition: _Counts,
    upcard_index: int,
    condition: PeekCondition,
) -> tuple[tuple[int, float, _Counts], ...]:
    total = sum(composition)
    if total < 2:
        return ()
    eligible_holes = sum(
        count
        for index, count in enumerate(composition)
        if _hole_allowed(index, upcard_index, condition)
    )
    if eligible_holes <= 0:
        raise ValueError("peek condition leaves no possible dealer hole card")
    denominator = eligible_holes * (total - 1)
    return tuple(
        (
            card_index,
            (
                count * eligible_holes
                - (count if _hole_allowed(card_index, upcard_index, condition) else 0)
            )
            / denominator,
            _remove_card(composition, card_index),
        )
        for card_index, count in enumerate(composition)
        if count
        and (
            count * eligible_holes
            - (count if _hole_allowed(card_index, upcard_index, condition) else 0)
        )
        > 0
    )


def _fixed_profit(hand: _ResolvedHand) -> int:
    return -hand.wager_half_units // 2 if hand.surrendered else -hand.wager_half_units


def _profit_against_dealer(hand: _ResolvedHand, dealer_outcome: int) -> int:
    if hand.surrendered:
        return -hand.wager_half_units // 2
    if hand.is_bust:
        return -hand.wager_half_units
    if dealer_outcome == _DEALER_BLACKJACK:
        return 0 if hand.is_natural else -hand.wager_half_units
    if hand.is_natural:
        return hand.wager_half_units * 3 // 2
    if dealer_outcome == _DEALER_BUST:
        return hand.wager_half_units
    if hand.total > dealer_outcome:
        return hand.wager_half_units
    if hand.total == dealer_outcome:
        return 0
    return -hand.wager_half_units


def _single_card_counts(card_index: int) -> _Counts:
    counts = [0] * len(CARD_VALUES)
    counts[card_index] = 1
    return tuple(counts)


def _hand_counts(first_index: int, second_index: int) -> _Counts:
    return _add_card(_single_card_counts(first_index), second_index)


def _add_card(counts: _Counts, card_index: int) -> _Counts:
    updated = list(counts)
    updated[card_index] += 1
    return tuple(updated)


def _remove_card(counts: _Counts, card_index: int) -> _Counts:
    if counts[card_index] <= 0:
        raise ValueError("cannot remove an unavailable card")
    updated = list(counts)
    updated[card_index] -= 1
    return tuple(updated)


def _hand_sort_key(hand: _Hand) -> tuple[int, _Counts, bool, bool, bool, bool]:
    return (
        hand.wager_half_units,
        hand.cards,
        hand.from_split,
        hand.split_aces,
        hand.can_double,
        hand.can_surrender,
    )


def _constant(profit_half_units: int) -> _NumericDistribution:
    return ((profit_half_units, 1.0),)


def _shift(
    distribution: _NumericDistribution,
    amount_half_units: int,
) -> _NumericDistribution:
    return tuple(
        (profit + amount_half_units, probability)
        for profit, probability in distribution
    )


def _from_weighted_profits(
    pairs: Iterable[tuple[int, float]],
) -> _NumericDistribution:
    merged: dict[int, list[float]] = {}
    for profit, probability in pairs:
        merged.setdefault(profit, []).append(probability)
    return tuple(
        (profit, fsum(probabilities))
        for profit, probabilities in sorted(merged.items())
        if fsum(probabilities) > 0
    )


def _mixture(
    branches: Iterable[tuple[float, _NumericDistribution]],
) -> _NumericDistribution:
    merged: dict[int, list[float]] = {}
    for branch_probability, distribution in branches:
        if branch_probability <= 0:
            continue
        for profit, probability in distribution:
            merged.setdefault(profit, []).append(branch_probability * probability)
    return tuple(
        (profit, fsum(probabilities))
        for profit, probabilities in sorted(merged.items())
        if fsum(probabilities) > 0
    )


def _expected(distribution: _NumericDistribution) -> float:
    return fsum((profit / 2.0) * probability for profit, probability in distribution)


def _to_return_distribution(
    distribution: _NumericDistribution,
) -> ReturnDistribution:
    """Store normalized float probabilities in the exact record container."""

    if not distribution:
        raise ValueError("numeric return distribution cannot be empty")
    total = fsum(probability for _, probability in distribution)
    fractions = [
        Fraction(probability / total).limit_denominator(_PROBABILITY_DENOMINATOR_LIMIT)
        for _, probability in distribution
    ]
    correction = Fraction(1) - sum(fractions, start=Fraction(0))
    largest = max(range(len(fractions)), key=fractions.__getitem__)
    fractions[largest] += correction
    return ReturnDistribution.from_pairs(
        (
            Fraction(profit_half_units, 2),
            probability,
        )
        for (profit_half_units, _), probability in zip(
            distribution,
            fractions,
            strict=True,
        )
    )
