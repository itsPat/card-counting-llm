"""Exact composition-dependent player action evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache

from blackjack.engine.actions import PlayerAction
from blackjack.engine.rules import FIXED_RULES, CasinoRules
from blackjack.oracle.composition import (
    CardValue,
    Composition,
    canonical_values,
)
from blackjack.oracle.dealer import (
    DealerOutcome,
    PeekCondition,
    dealer_distribution,
    hidden_hole_draws,
)
from blackjack.oracle.distributions import ReturnDistribution


@dataclass(frozen=True, slots=True)
class OracleHandValue:
    total: int
    is_soft: bool
    is_bust: bool


def oracle_hand_value(cards: tuple[CardValue, ...]) -> OracleHandValue:
    hard_total = sum(card.hard_value for card in cards)
    aces = sum(card is CardValue.ACE for card in cards)
    is_soft = aces > 0 and hard_total + 10 <= 21
    total = hard_total + 10 if is_soft else hard_total
    return OracleHandValue(total, is_soft, total > 21)


@dataclass(frozen=True, slots=True)
class OracleHand:
    cards: tuple[CardValue, ...]
    wager: Fraction = Fraction(1)
    from_split: bool = False
    split_aces: bool = False
    can_double: bool = True
    can_surrender: bool = True

    def __post_init__(self) -> None:
        if not self.cards:
            raise ValueError("an oracle hand needs at least one card")
        if self.wager <= 0:
            raise ValueError("wager must be positive")
        if self.cards != canonical_values(self.cards):
            object.__setattr__(self, "cards", canonical_values(self.cards))

    @property
    def value(self) -> OracleHandValue:
        return oracle_hand_value(self.cards)

    @property
    def is_natural_blackjack(self) -> bool:
        return not self.from_split and len(self.cards) == 2 and self.value.total == 21

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0] is self.cards[1]

    def add(self, value: CardValue) -> OracleHand:
        return replace(
            self,
            cards=canonical_values((*self.cards, value)),
            can_double=False,
            can_surrender=False,
        )

    def add_split_card(self, value: CardValue) -> OracleHand:
        return replace(
            self,
            cards=canonical_values((*self.cards, value)),
        )


@dataclass(frozen=True, slots=True)
class ResolvedHand:
    total: int
    wager: Fraction
    is_natural_blackjack: bool
    is_bust: bool
    surrendered: bool = False

    @classmethod
    def from_hand(
        cls,
        hand: OracleHand,
        *,
        surrendered: bool = False,
    ) -> ResolvedHand:
        return cls(
            total=hand.value.total,
            wager=hand.wager,
            is_natural_blackjack=hand.is_natural_blackjack,
            is_bust=hand.value.is_bust,
            surrendered=surrendered,
        )


def _add_finished(
    finished: tuple[ResolvedHand, ...],
    resolved: ResolvedHand,
) -> tuple[ResolvedHand, ...]:
    return tuple(
        sorted(
            (*finished, resolved),
            key=lambda hand: (
                hand.surrendered,
                hand.is_bust,
                hand.is_natural_blackjack,
                hand.total,
                hand.wager,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class PlayerSituation:
    composition: Composition
    hand: OracleHand
    dealer_upcard: CardValue
    peek_condition: PeekCondition
    rules: CasinoRules = FIXED_RULES
    unseen_unavailable: int = 0

    def __post_init__(self) -> None:
        if self.composition.total == 0:
            raise ValueError("composition must include the hidden dealer card")
        if self.unseen_unavailable < 0:
            raise ValueError("unseen unavailable card count cannot be negative")
        if self.composition.total <= self.unseen_unavailable:
            raise ValueError("composition must leave a possible dealer hole card")
        if (
            self.dealer_upcard in (CardValue.ACE, CardValue.TEN)
            and self.peek_condition is PeekCondition.NONE
        ):
            raise ValueError(
                "player actions against an Ace or ten require a negative peek"
            )


@dataclass(frozen=True, slots=True)
class RoundPlayerSituation:
    """A complete in-progress split round from the player's information set."""

    composition: Composition
    active_hand: OracleHand
    pending_hands: tuple[OracleHand, ...]
    finished_hands: tuple[ResolvedHand, ...]
    dealer_upcard: CardValue
    peek_condition: PeekCondition
    rules: CasinoRules = FIXED_RULES
    unseen_unavailable: int = 0

    def __post_init__(self) -> None:
        if self.composition.total == 0:
            raise ValueError("composition must include the hidden dealer card")
        if self.unseen_unavailable < 0:
            raise ValueError("unseen unavailable card count cannot be negative")
        if self.composition.total <= self.unseen_unavailable:
            raise ValueError("composition must leave a possible dealer hole card")
        if (
            self.dealer_upcard in (CardValue.ACE, CardValue.TEN)
            and self.peek_condition is PeekCondition.NONE
        ):
            raise ValueError(
                "player actions against an Ace or ten require a negative peek"
            )
        hand_count = 1 + len(self.pending_hands) + len(self.finished_hands)
        if hand_count > self.rules.maximum_player_hands:
            raise ValueError("round has more hands than the rules allow")


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    action: PlayerAction
    distribution: ReturnDistribution

    @property
    def expected_profit(self) -> Fraction:
        return self.distribution.expected_profit


def legal_actions(
    situation: PlayerSituation,
    *,
    hands_in_round: int = 1,
) -> tuple[PlayerAction, ...]:
    return _legal_actions(situation.hand, hands_in_round, situation.rules)


def _legal_actions(
    hand: OracleHand,
    hands_in_round: int,
    rules: CasinoRules,
) -> tuple[PlayerAction, ...]:
    if hand.value.is_bust or hand.value.total >= 21 or hand.split_aces:
        return ()
    actions = [PlayerAction.HIT, PlayerAction.STAND]
    if (
        len(hand.cards) == 2
        and hand.can_double
        and (not hand.from_split or rules.double_after_split)
    ):
        actions.append(PlayerAction.DOUBLE)
    if (
        hand.is_pair
        and len(hand.cards) == 2
        and hands_in_round < rules.maximum_player_hands
        and not (
            hand.from_split
            and hand.cards[0] is CardValue.ACE
            and not rules.resplit_aces
        )
    ):
        actions.append(PlayerAction.SPLIT)
    if (
        hand.can_surrender
        and not hand.from_split
        and len(hand.cards) == 2
        and rules.late_surrender
    ):
        actions.append(PlayerAction.SURRENDER)
    return tuple(actions)


def evaluate_actions(
    situation: PlayerSituation,
) -> tuple[ActionEvaluation, ...]:
    actions = legal_actions(situation)
    return tuple(
        ActionEvaluation(
            action=action,
            distribution=_take_action(
                situation.composition,
                situation.dealer_upcard,
                situation.peek_condition,
                situation.hand,
                (),
                (),
                action,
                situation.rules,
                situation.unseen_unavailable,
            ),
        )
        for action in actions
    )


def evaluate_round_actions(
    situation: RoundPlayerSituation,
) -> tuple[ActionEvaluation, ...]:
    """Evaluate actions while retaining every correlated split-hand outcome."""

    hands_in_round = 1 + len(situation.pending_hands) + len(situation.finished_hands)
    actions = _legal_actions(
        situation.active_hand,
        hands_in_round,
        situation.rules,
    )
    return tuple(
        ActionEvaluation(
            action=action,
            distribution=_take_action(
                situation.composition,
                situation.dealer_upcard,
                situation.peek_condition,
                situation.active_hand,
                situation.pending_hands,
                situation.finished_hands,
                action,
                situation.rules,
                situation.unseen_unavailable,
            ),
        )
        for action in actions
    )


def optimal_action(situation: PlayerSituation) -> ActionEvaluation:
    evaluations = evaluate_actions(situation)
    if not evaluations:
        raise ValueError("the situation does not require a player action")
    return max(evaluations, key=lambda evaluation: evaluation.expected_profit)


def optimal_round_action(situation: RoundPlayerSituation) -> ActionEvaluation:
    evaluations = evaluate_round_actions(situation)
    if not evaluations:
        raise ValueError("the round situation does not require a player action")
    return max(evaluations, key=lambda evaluation: evaluation.expected_profit)


def optimal_return_distribution(
    situation: PlayerSituation,
) -> ReturnDistribution:
    return _optimal_distribution(
        situation.composition,
        situation.dealer_upcard,
        situation.peek_condition,
        situation.hand,
        (),
        (),
        situation.rules,
        situation.unseen_unavailable,
    )


@cache
def _optimal_distribution(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    active: OracleHand,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> ReturnDistribution:
    if active.value.is_bust or active.value.total >= 21 or active.split_aces:
        return _advance(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(finished, ResolvedHand.from_hand(active)),
            rules,
            unseen_unavailable,
        )
    actions = _legal_actions(
        active,
        len(finished) + len(pending) + 1,
        rules,
    )
    if not actions:
        return _advance(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(finished, ResolvedHand.from_hand(active)),
            rules,
            unseen_unavailable,
        )
    best_action = max(
        actions,
        key=lambda action: _action_expected_profit(
            composition,
            upcard,
            condition,
            active,
            pending,
            finished,
            action,
            rules,
            unseen_unavailable,
        ),
    )
    return _take_action(
        composition,
        upcard,
        condition,
        active,
        pending,
        finished,
        best_action,
        rules,
        unseen_unavailable,
    )


@cache
def _take_action(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    active: OracleHand,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    action: PlayerAction,
    rules: CasinoRules,
    unseen_unavailable: int,
) -> ReturnDistribution:
    hands_in_round = len(finished) + len(pending) + 1
    if action not in _legal_actions(active, hands_in_round, rules):
        raise ValueError(f"{action.value} is not legal for this oracle state")

    if action is PlayerAction.STAND:
        return _advance(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(finished, ResolvedHand.from_hand(active)),
            rules,
            unseen_unavailable,
        )
    if action is PlayerAction.SURRENDER:
        return _advance(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(
                finished,
                ResolvedHand.from_hand(active, surrendered=True),
            ),
            rules,
            unseen_unavailable,
        )

    draws = hidden_hole_draws(
        composition,
        upcard,
        condition,
        unseen_unavailable,
    )
    if not draws:
        raise ValueError("the player must draw but no shoe card is available")
    if action is PlayerAction.HIT:
        return ReturnDistribution.mixture(
            (
                draw.probability,
                _optimal_distribution(
                    draw.composition,
                    upcard,
                    condition,
                    active.add(draw.value),
                    pending,
                    finished,
                    rules,
                    unseen_unavailable,
                ),
            )
            for draw in draws
        )
    if action is PlayerAction.DOUBLE:
        doubled = replace(
            active,
            wager=active.wager * 2,
            can_double=False,
            can_surrender=False,
        )
        return ReturnDistribution.mixture(
            (
                draw.probability,
                _advance(
                    draw.composition,
                    upcard,
                    condition,
                    pending,
                    _add_finished(
                        finished,
                        ResolvedHand.from_hand(doubled.add(draw.value)),
                    ),
                    rules,
                    unseen_unavailable,
                ),
            )
            for draw in draws
        )
    return _split_distribution(
        composition,
        upcard,
        condition,
        active,
        pending,
        finished,
        rules,
        unseen_unavailable,
    )


@cache
def _optimal_expected_profit(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    active: OracleHand,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> Fraction:
    if active.value.is_bust or active.value.total >= 21 or active.split_aces:
        return _advance_expected_profit(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(finished, ResolvedHand.from_hand(active)),
            rules,
            unseen_unavailable,
        )
    actions = _legal_actions(
        active,
        len(finished) + len(pending) + 1,
        rules,
    )
    if not actions:
        return _advance_expected_profit(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(finished, ResolvedHand.from_hand(active)),
            rules,
            unseen_unavailable,
        )
    return max(
        _action_expected_profit(
            composition,
            upcard,
            condition,
            active,
            pending,
            finished,
            action,
            rules,
            unseen_unavailable,
        )
        for action in actions
    )


@cache
def _action_expected_profit(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    active: OracleHand,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    action: PlayerAction,
    rules: CasinoRules,
    unseen_unavailable: int,
) -> Fraction:
    hands_in_round = len(finished) + len(pending) + 1
    if action not in _legal_actions(active, hands_in_round, rules):
        raise ValueError(f"{action.value} is not legal for this oracle state")

    if action is PlayerAction.STAND:
        return _advance_expected_profit(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(finished, ResolvedHand.from_hand(active)),
            rules,
            unseen_unavailable,
        )
    if action is PlayerAction.SURRENDER:
        return _advance_expected_profit(
            composition,
            upcard,
            condition,
            pending,
            _add_finished(
                finished,
                ResolvedHand.from_hand(active, surrendered=True),
            ),
            rules,
            unseen_unavailable,
        )

    draws = hidden_hole_draws(
        composition,
        upcard,
        condition,
        unseen_unavailable,
    )
    if not draws:
        raise ValueError("the player must draw but no shoe card is available")
    if action is PlayerAction.HIT:
        return sum(
            (
                draw.probability
                * _optimal_expected_profit(
                    draw.composition,
                    upcard,
                    condition,
                    active.add(draw.value),
                    pending,
                    finished,
                    rules,
                    unseen_unavailable,
                )
                for draw in draws
            ),
            start=Fraction(0),
        )
    if action is PlayerAction.DOUBLE:
        doubled = replace(
            active,
            wager=active.wager * 2,
            can_double=False,
            can_surrender=False,
        )
        return sum(
            (
                draw.probability
                * _advance_expected_profit(
                    draw.composition,
                    upcard,
                    condition,
                    pending,
                    _add_finished(
                        finished,
                        ResolvedHand.from_hand(doubled.add(draw.value)),
                    ),
                    rules,
                    unseen_unavailable,
                )
                for draw in draws
            ),
            start=Fraction(0),
        )
    return _split_expected_profit(
        composition,
        upcard,
        condition,
        active,
        pending,
        finished,
        rules,
        unseen_unavailable,
    )


def _split_expected_profit(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    active: OracleHand,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> Fraction:
    pair_value = active.cards[0]
    split_aces = pair_value is CardValue.ACE
    base = OracleHand(
        cards=(pair_value,),
        wager=active.wager,
        from_split=True,
        split_aces=split_aces,
        can_double=rules.double_after_split,
        can_surrender=False,
    )
    expected = Fraction(0)
    for left_draw in hidden_hole_draws(
        composition,
        upcard,
        condition,
        unseen_unavailable,
    ):
        left = base.add_split_card(left_draw.value)
        for right_draw in hidden_hole_draws(
            left_draw.composition,
            upcard,
            condition,
            unseen_unavailable,
        ):
            right = base.add_split_card(right_draw.value)
            probability = left_draw.probability * right_draw.probability
            if split_aces and rules.split_aces_one_card_only:
                child = _advance_expected_profit(
                    right_draw.composition,
                    upcard,
                    condition,
                    pending,
                    _add_finished(
                        _add_finished(
                            finished,
                            ResolvedHand.from_hand(left),
                        ),
                        ResolvedHand.from_hand(right),
                    ),
                    rules,
                    unseen_unavailable,
                )
            else:
                child = _optimal_expected_profit(
                    right_draw.composition,
                    upcard,
                    condition,
                    left,
                    (right, *pending),
                    finished,
                    rules,
                    unseen_unavailable,
                )
            expected += probability * child
    return expected


def _split_distribution(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    active: OracleHand,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> ReturnDistribution:
    pair_value = active.cards[0]
    split_aces = pair_value is CardValue.ACE
    base = OracleHand(
        cards=(pair_value,),
        wager=active.wager,
        from_split=True,
        split_aces=split_aces,
        can_double=rules.double_after_split,
        can_surrender=False,
    )
    branches: list[tuple[Fraction, ReturnDistribution]] = []
    for left_draw in hidden_hole_draws(
        composition,
        upcard,
        condition,
        unseen_unavailable,
    ):
        left = base.add_split_card(left_draw.value)
        for right_draw in hidden_hole_draws(
            left_draw.composition,
            upcard,
            condition,
            unseen_unavailable,
        ):
            right = base.add_split_card(right_draw.value)
            probability = left_draw.probability * right_draw.probability
            if split_aces and rules.split_aces_one_card_only:
                distribution = _advance(
                    right_draw.composition,
                    upcard,
                    condition,
                    pending,
                    _add_finished(
                        _add_finished(
                            finished,
                            ResolvedHand.from_hand(left),
                        ),
                        ResolvedHand.from_hand(right),
                    ),
                    rules,
                    unseen_unavailable,
                )
            else:
                distribution = _optimal_distribution(
                    right_draw.composition,
                    upcard,
                    condition,
                    left,
                    (right, *pending),
                    finished,
                    rules,
                    unseen_unavailable,
                )
            branches.append((probability, distribution))
    return ReturnDistribution.mixture(branches)


def _advance(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> ReturnDistribution:
    if pending:
        return _optimal_distribution(
            composition,
            upcard,
            condition,
            pending[0],
            pending[1:],
            finished,
            rules,
            unseen_unavailable,
        )
    return _settle_finished(
        composition,
        upcard,
        condition,
        finished,
        rules,
        unseen_unavailable,
    )


def _advance_expected_profit(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> Fraction:
    if pending:
        return _optimal_expected_profit(
            composition,
            upcard,
            condition,
            pending[0],
            pending[1:],
            finished,
            rules,
            unseen_unavailable,
        )
    return _settle_finished_expected_profit(
        composition,
        upcard,
        condition,
        finished,
        rules,
        unseen_unavailable,
    )


@cache
def _settle_finished(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> ReturnDistribution:
    if all(resolved.surrendered or resolved.is_bust for resolved in finished):
        return ReturnDistribution.constant(
            sum(
                (_fixed_loss(resolved) for resolved in finished),
                start=Fraction(0),
            )
        )
    dealer = dealer_distribution(
        composition,
        upcard,
        condition,
        rules,
        unseen_unavailable,
    )
    return ReturnDistribution.from_pairs(
        (
            sum(
                (
                    _profit_against_dealer(resolved, item.outcome, rules)
                    for resolved in finished
                ),
                start=Fraction(0),
            ),
            item.probability,
        )
        for item in dealer.outcomes
    )


@cache
def _settle_finished_expected_profit(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
    unseen_unavailable: int,
) -> Fraction:
    if all(resolved.surrendered or resolved.is_bust for resolved in finished):
        return sum(
            (_fixed_loss(resolved) for resolved in finished),
            start=Fraction(0),
        )
    dealer = dealer_distribution(
        composition,
        upcard,
        condition,
        rules,
        unseen_unavailable,
    )
    return sum(
        (
            item.probability
            * sum(
                (
                    _profit_against_dealer(resolved, item.outcome, rules)
                    for resolved in finished
                ),
                start=Fraction(0),
            )
            for item in dealer.outcomes
        ),
        start=Fraction(0),
    )


def _fixed_loss(resolved: ResolvedHand) -> Fraction:
    return -resolved.wager / 2 if resolved.surrendered else -resolved.wager


def _profit_against_dealer(
    resolved: ResolvedHand,
    dealer: DealerOutcome,
    rules: CasinoRules,
) -> Fraction:
    if resolved.surrendered:
        return -resolved.wager / 2
    if resolved.is_bust:
        return -resolved.wager
    if dealer is DealerOutcome.BLACKJACK:
        return Fraction(0) if resolved.is_natural_blackjack else -resolved.wager
    if resolved.is_natural_blackjack:
        return resolved.wager * rules.blackjack_profit
    if dealer is DealerOutcome.BUST:
        return resolved.wager * rules.ordinary_win_profit
    dealer_total = dealer.total
    if dealer_total is None:
        raise AssertionError("live dealer outcome must have a total")
    if resolved.total > dealer_total:
        return resolved.wager * rules.ordinary_win_profit
    if resolved.total == dealer_total:
        return Fraction(0)
    return -resolved.wager


def player_cache_counts() -> tuple[tuple[str, int, int, int], ...]:
    """Expose memoization counters without leaking cached state."""

    optimal = _optimal_distribution.cache_info()
    action = _take_action.cache_info()
    settlement = _settle_finished.cache_info()
    optimal_value = _optimal_expected_profit.cache_info()
    action_value = _action_expected_profit.cache_info()
    settlement_value = _settle_finished_expected_profit.cache_info()
    return (
        ("player_optimal", optimal.hits, optimal.misses, optimal.currsize),
        ("player_action", action.hits, action.misses, action.currsize),
        (
            "player_settlement",
            settlement.hits,
            settlement.misses,
            settlement.currsize,
        ),
        (
            "player_optimal_value",
            optimal_value.hits,
            optimal_value.misses,
            optimal_value.currsize,
        ),
        (
            "player_action_value",
            action_value.hits,
            action_value.misses,
            action_value.currsize,
        ),
        (
            "player_settlement_value",
            settlement_value.hits,
            settlement_value.misses,
            settlement_value.currsize,
        ),
    )


def clear_player_caches() -> None:
    _optimal_distribution.cache_clear()
    _take_action.cache_clear()
    _settle_finished.cache_clear()
    _optimal_expected_profit.cache_clear()
    _action_expected_profit.cache_clear()
    _settle_finished_expected_profit.cache_clear()
