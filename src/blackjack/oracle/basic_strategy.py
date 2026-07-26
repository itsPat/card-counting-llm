"""Documented six-deck H17 basic strategy used by rollout continuations."""

from __future__ import annotations

from blackjack.engine import PlayerAction
from blackjack.oracle.composition import CardValue
from blackjack.oracle.player import oracle_hand_value


def _dealer_value(card: CardValue) -> int:
    return 11 if card is CardValue.ACE else card.hard_value


def basic_strategy_action(
    cards: tuple[CardValue, ...],
    dealer_upcard: CardValue,
    legal_actions: tuple[PlayerAction, ...],
) -> PlayerAction:
    """Choose six-deck H17, DAS, late-surrender basic strategy."""

    if not cards:
        raise ValueError("basic strategy needs a player hand")
    if not legal_actions:
        raise ValueError("basic strategy needs at least one legal action")
    value = oracle_hand_value(cards)
    dealer = _dealer_value(dealer_upcard)

    if PlayerAction.SURRENDER in legal_actions and not value.is_soft:
        surrender = (
            (value.total == 17 and dealer == 11)
            or (value.total == 16 and dealer in (9, 10, 11))
            or (value.total == 15 and dealer in (10, 11))
        )
        if surrender:
            return PlayerAction.SURRENDER

    if (
        PlayerAction.SPLIT in legal_actions
        and len(cards) == 2
        and cards[0] is cards[1]
    ):
        pair = cards[0]
        should_split = (
            pair in (CardValue.ACE, CardValue.EIGHT)
            or (
                pair is CardValue.NINE
                and dealer in (2, 3, 4, 5, 6, 8, 9)
            )
            or (
                pair is CardValue.SEVEN
                and dealer in (2, 3, 4, 5, 6, 7)
            )
            or (
                pair is CardValue.SIX
                and dealer in (2, 3, 4, 5, 6)
            )
            or (pair is CardValue.FOUR and dealer in (5, 6))
            or (
                pair in (CardValue.TWO, CardValue.THREE)
                and dealer in (2, 3, 4, 5, 6, 7)
            )
        )
        if should_split:
            return PlayerAction.SPLIT

    can_double = PlayerAction.DOUBLE in legal_actions
    if value.is_soft:
        if value.total >= 20:
            return PlayerAction.STAND
        if value.total == 19:
            return (
                PlayerAction.DOUBLE
                if can_double and dealer == 6
                else PlayerAction.STAND
            )
        if value.total == 18:
            if can_double and dealer in (2, 3, 4, 5, 6):
                return PlayerAction.DOUBLE
            return (
                PlayerAction.STAND
                if dealer in (2, 3, 4, 5, 6, 7, 8)
                else PlayerAction.HIT
            )
        double_ranges = {
            17: (3, 4, 5, 6),
            16: (4, 5, 6),
            15: (4, 5, 6),
            14: (5, 6),
            13: (5, 6),
        }
        if can_double and dealer in double_ranges.get(value.total, ()):
            return PlayerAction.DOUBLE
        return PlayerAction.HIT

    if value.total >= 17:
        return PlayerAction.STAND
    if 13 <= value.total <= 16:
        return (
            PlayerAction.STAND
            if dealer in (2, 3, 4, 5, 6)
            else PlayerAction.HIT
        )
    if value.total == 12:
        return (
            PlayerAction.STAND
            if dealer in (4, 5, 6)
            else PlayerAction.HIT
        )
    if can_double and value.total == 11:
        return PlayerAction.DOUBLE
    if can_double and value.total == 10 and dealer in range(2, 10):
        return PlayerAction.DOUBLE
    if can_double and value.total == 9 and dealer in (3, 4, 5, 6):
        return PlayerAction.DOUBLE
    return PlayerAction.HIT
