"""Exact composition-dependent player action evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache

from blackjack.actions import PlayerAction
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
from blackjack.rules import FIXED_RULES, CasinoRules


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
    hand: OracleHand
    surrendered: bool = False


@dataclass(frozen=True, slots=True)
class PlayerSituation:
    composition: Composition
    hand: OracleHand
    dealer_upcard: CardValue
    peek_condition: PeekCondition
    rules: CasinoRules = FIXED_RULES

    def __post_init__(self) -> None:
        if self.composition.total == 0:
            raise ValueError("composition must include the hidden dealer card")
        if (
            self.dealer_upcard in (CardValue.ACE, CardValue.TEN)
            and self.peek_condition is PeekCondition.NONE
        ):
            raise ValueError(
                "player actions against an Ace or ten require a negative peek"
            )


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
            ),
        )
        for action in actions
    )


def optimal_action(situation: PlayerSituation) -> ActionEvaluation:
    evaluations = evaluate_actions(situation)
    if not evaluations:
        raise ValueError("the situation does not require a player action")
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
) -> ReturnDistribution:
    if active.value.is_bust or active.value.total >= 21 or active.split_aces:
        return _advance(
            composition,
            upcard,
            condition,
            pending,
            (*finished, ResolvedHand(active)),
            rules,
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
            (*finished, ResolvedHand(active)),
            rules,
        )
    choices = tuple(
        _take_action(
            composition,
            upcard,
            condition,
            active,
            pending,
            finished,
            action,
            rules,
        )
        for action in actions
    )
    return max(choices, key=lambda distribution: distribution.expected_profit)


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
            (*finished, ResolvedHand(active)),
            rules,
        )
    if action is PlayerAction.SURRENDER:
        return _advance(
            composition,
            upcard,
            condition,
            pending,
            (*finished, ResolvedHand(active, surrendered=True)),
            rules,
        )

    draws = hidden_hole_draws(composition, upcard, condition)
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
                    (*finished, ResolvedHand(doubled.add(draw.value))),
                    rules,
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
    )


def _split_distribution(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    active: OracleHand,
    pending: tuple[OracleHand, ...],
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
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
    for left_draw in hidden_hole_draws(composition, upcard, condition):
        left = base.add_split_card(left_draw.value)
        for right_draw in hidden_hole_draws(
            left_draw.composition,
            upcard,
            condition,
        ):
            right = base.add_split_card(right_draw.value)
            probability = left_draw.probability * right_draw.probability
            if split_aces and rules.split_aces_one_card_only:
                distribution = _advance(
                    right_draw.composition,
                    upcard,
                    condition,
                    pending,
                    (
                        *finished,
                        ResolvedHand(left),
                        ResolvedHand(right),
                    ),
                    rules,
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
        )
    return _settle_finished(composition, upcard, condition, finished, rules)


@cache
def _settle_finished(
    composition: Composition,
    upcard: CardValue,
    condition: PeekCondition,
    finished: tuple[ResolvedHand, ...],
    rules: CasinoRules,
) -> ReturnDistribution:
    if all(
        resolved.surrendered or resolved.hand.value.is_bust for resolved in finished
    ):
        return ReturnDistribution.constant(
            sum(
                (_fixed_loss(resolved) for resolved in finished),
                start=Fraction(0),
            )
        )
    dealer = dealer_distribution(composition, upcard, condition, rules)
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


def _fixed_loss(resolved: ResolvedHand) -> Fraction:
    return -resolved.hand.wager / 2 if resolved.surrendered else -resolved.hand.wager


def _profit_against_dealer(
    resolved: ResolvedHand,
    dealer: DealerOutcome,
    rules: CasinoRules,
) -> Fraction:
    hand = resolved.hand
    if resolved.surrendered:
        return -hand.wager / 2
    if hand.value.is_bust:
        return -hand.wager
    if dealer is DealerOutcome.BLACKJACK:
        return Fraction(0) if hand.is_natural_blackjack else -hand.wager
    if hand.is_natural_blackjack:
        return hand.wager * rules.blackjack_profit
    if dealer is DealerOutcome.BUST:
        return hand.wager * rules.ordinary_win_profit
    dealer_total = dealer.total
    if dealer_total is None:
        raise AssertionError("live dealer outcome must have a total")
    if hand.value.total > dealer_total:
        return hand.wager * rules.ordinary_win_profit
    if hand.value.total == dealer_total:
        return Fraction(0)
    return -hand.wager
