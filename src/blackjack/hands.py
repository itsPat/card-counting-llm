"""Immutable blackjack hands and Ace-aware hand valuation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from blackjack.cards import Card


@dataclass(frozen=True, slots=True)
class HandValue:
    """The best blackjack value for a collection of cards."""

    total: int
    is_soft: bool
    is_bust: bool


def calculate_hand_value(cards: Iterable[Card]) -> HandValue:
    """Calculate the highest non-busting total, or the minimum bust total.

    All Aces begin at one. Exactly one Ace is promoted to eleven when the
    extra ten points keep the hand at or below 21.
    """

    card_tuple = tuple(cards)
    hard_total = sum(card.value for card in card_tuple)
    has_ace = any(card.rank.value == "A" for card in card_tuple)
    is_soft = has_ace and hard_total + 10 <= 21
    total = hard_total + 10 if is_soft else hard_total
    return HandValue(total=total, is_soft=is_soft, is_bust=total > 21)


@dataclass(frozen=True, slots=True)
class Hand:
    """An immutable collection of cards with blackjack semantics."""

    cards: tuple[Card, ...]
    from_split: bool = False

    @property
    def value(self) -> HandValue:
        return calculate_hand_value(self.cards)

    @property
    def is_natural_blackjack(self) -> bool:
        return not self.from_split and len(self.cards) == 2 and self.value.total == 21

    @property
    def is_pair(self) -> bool:
        if len(self.cards) != 2:
            return False
        left, right = self.cards
        return left.value == right.value

    def add(self, card: Card) -> Hand:
        return Hand(cards=(*self.cards, card), from_split=self.from_split)
