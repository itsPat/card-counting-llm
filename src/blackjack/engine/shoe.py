"""Six-deck shoe creation, deterministic shuffling, and explicit replay."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, floor
from random import Random

from blackjack.engine.cards import Card, Rank
from blackjack.engine.rules import FIXED_RULES, CasinoRules

DEFAULT_BURN_CARD = Card(Rank.TWO)


class ShoeExhaustedError(RuntimeError):
    """Raised when a replay has no card available for the next deal."""


class InvalidReplayError(ValueError):
    """Raised when replay data cannot describe a valid shoe."""


@dataclass(frozen=True, slots=True)
class ShoeReplay:
    """Complete immutable data required to reproduce a shoe exactly.

    ``cards`` is the physical top-to-bottom order before the unseen burn card.
    ``cut_card_position`` is the zero-based boundary in that same order.
    """

    cards: tuple[Card, ...]
    cut_card_position: int

    def __post_init__(self) -> None:
        if len(self.cards) < 2:
            raise InvalidReplayError("a replay needs a burn card and a dealable card")
        if not 1 <= self.cut_card_position <= len(self.cards):
            raise InvalidReplayError("cut card position must lie within the replay")

    @property
    def burn_card(self) -> Card:
        return self.cards[0]

    @property
    def deal_order(self) -> tuple[Card, ...]:
        return self.cards[1:]


class Shoe:
    """Mutable dealing cursor over immutable replay data."""

    __slots__ = ("_next_position", "_replay")

    def __init__(self, replay: ShoeReplay) -> None:
        self._replay = replay
        self._next_position = 1

    @classmethod
    def shuffled(
        cls,
        seed: int,
        rules: CasinoRules = FIXED_RULES,
    ) -> Shoe:
        rng = Random(seed)
        cards = [Card(rank) for rank in Rank for _ in range(4 * rules.decks)]
        rng.shuffle(cards)
        minimum_cut = ceil(len(cards) * rules.minimum_penetration)
        maximum_cut = floor(len(cards) * rules.maximum_penetration)
        cut = rng.randint(minimum_cut, maximum_cut)
        return cls(ShoeReplay(cards=tuple(cards), cut_card_position=cut))

    @classmethod
    def from_replay(cls, replay: ShoeReplay) -> Shoe:
        return cls(replay)

    @classmethod
    def arranged(
        cls,
        deal_order: Iterable[Card],
        *,
        burn_card: Card = DEFAULT_BURN_CARD,
        cut_card_position: int | None = None,
    ) -> Shoe:
        """Build a compact deliberate replay for a deterministic rule example."""

        replay_cards = (burn_card, *tuple(deal_order))
        cut = cut_card_position if cut_card_position is not None else len(replay_cards)
        return cls(ShoeReplay(cards=replay_cards, cut_card_position=cut))

    @property
    def replay(self) -> ShoeReplay:
        return self._replay

    @property
    def burn_card(self) -> Card:
        return self._replay.burn_card

    @property
    def dealt_count(self) -> int:
        return self._next_position - 1

    @property
    def remaining(self) -> int:
        return len(self._replay.cards) - self._next_position

    @property
    def reached_cut_card(self) -> bool:
        return self._next_position >= self._replay.cut_card_position

    def deal(self) -> Card:
        if self._next_position >= len(self._replay.cards):
            raise ShoeExhaustedError("the shoe has no cards remaining")
        card = self._replay.cards[self._next_position]
        self._next_position += 1
        return card


def six_deck_rank_counts() -> Counter[Rank]:
    """Return the exact initial rank inventory for the fixed shoe."""

    return Counter({rank: 4 * FIXED_RULES.decks for rank in Rank})
