"""Immutable decision-relevant composition of an unseen shoe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from blackjack.engine.cards import Card, Rank


class CardValue(StrEnum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"

    @property
    def hard_value(self) -> int:
        return 1 if self is CardValue.ACE else int(self.value)

    @classmethod
    def from_card(cls, card: Card) -> CardValue:
        return cls.TEN if card.is_ten_valued else cls(card.rank.value)


CARD_VALUES: tuple[CardValue, ...] = tuple(CardValue)
_VALUE_INDEX: dict[CardValue, int] = {
    value: index for index, value in enumerate(CARD_VALUES)
}


@dataclass(frozen=True, slots=True)
class Draw:
    value: CardValue
    probability: Fraction
    composition: Composition


@dataclass(frozen=True, slots=True)
class Composition:
    """Counts for A, 2..9, and all ten-valued ranks combined."""

    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.counts) != len(CARD_VALUES):
            raise ValueError("composition needs exactly ten value counts")
        if any(count < 0 for count in self.counts):
            raise ValueError("composition counts cannot be negative")

    @classmethod
    def empty(cls) -> Composition:
        return cls((0,) * len(CARD_VALUES))

    @classmethod
    def full_shoe(cls, decks: int = 6) -> Composition:
        if decks <= 0:
            raise ValueError("deck count must be positive")
        per_rank = 4 * decks
        return cls((*((per_rank,) * 9), per_rank * 4))

    @classmethod
    def from_values(cls, values: Iterable[CardValue]) -> Composition:
        counts = Counter(values)
        return cls(tuple(counts[value] for value in CARD_VALUES))

    @classmethod
    def from_cards(cls, cards: Iterable[Card]) -> Composition:
        return cls.from_values(CardValue.from_card(card) for card in cards)

    @property
    def total(self) -> int:
        return sum(self.counts)

    def count(self, value: CardValue) -> int:
        return self.counts[_VALUE_INDEX[value]]

    def remove(self, value: CardValue, number: int = 1) -> Composition:
        if number <= 0:
            raise ValueError("number removed must be positive")
        index = _VALUE_INDEX[value]
        if self.counts[index] < number:
            raise ValueError(f"cannot remove {number} {value.value} card(s)")
        updated = list(self.counts)
        updated[index] -= number
        return Composition(tuple(updated))

    def remove_cards(self, cards: Iterable[Card]) -> Composition:
        result = self
        for card in cards:
            result = result.remove(CardValue.from_card(card))
        return result

    def draws(self) -> tuple[Draw, ...]:
        if self.total == 0:
            return ()
        return tuple(
            Draw(
                value=value,
                probability=Fraction(count, self.total),
                composition=self.remove(value),
            )
            for value, count in self
            if count > 0
        )

    def __iter__(self) -> Iterator[tuple[CardValue, int]]:
        return iter(zip(CARD_VALUES, self.counts, strict=True))


def canonical_values(values: Iterable[CardValue]) -> tuple[CardValue, ...]:
    return tuple(sorted(values, key=_VALUE_INDEX.__getitem__))


def cards_to_values(cards: Iterable[Card]) -> tuple[CardValue, ...]:
    return canonical_values(CardValue.from_card(card) for card in cards)


def value_to_card(value: CardValue) -> Card:
    rank = Rank.TEN if value is CardValue.TEN else Rank(value.value)
    return Card(rank)
