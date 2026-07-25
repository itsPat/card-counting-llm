"""Card ranks and values used by the blackjack engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Rank(StrEnum):
    """The thirteen ranks in a standard deck.

    Suits are intentionally absent: without side bets, they cannot affect a
    blackjack decision.
    """

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
    JACK = "J"
    QUEEN = "Q"
    KING = "K"

    @property
    def hard_value(self) -> int:
        """Return the rank's value when every Ace counts as one."""

        if self is Rank.ACE:
            return 1
        if self in TEN_VALUED_RANKS:
            return 10
        return int(self.value)

    @property
    def is_ten_valued(self) -> bool:
        return self in TEN_VALUED_RANKS

    @classmethod
    def parse(cls, value: str) -> Rank:
        """Parse a compact rank label such as ``"A"`` or ``"10"``."""

        return cls(value.strip().upper())


TEN_VALUED_RANKS: frozenset[Rank] = frozenset(
    {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING}
)


@dataclass(frozen=True, slots=True)
class Card:
    """A physical card represented only by its decision-relevant rank."""

    rank: Rank

    @property
    def value(self) -> int:
        return self.rank.hard_value

    @property
    def is_ten_valued(self) -> bool:
        return self.rank.is_ten_valued

    def __str__(self) -> str:
        return self.rank.value


def cards(*ranks: Rank | str) -> tuple[Card, ...]:
    """Create a compact immutable card tuple for examples and tests."""

    return tuple(
        Card(rank if isinstance(rank, Rank) else Rank.parse(rank)) for rank in ranks
    )
