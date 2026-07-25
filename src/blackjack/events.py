"""Typed internal and player-visible event records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from blackjack.actions import InsuranceAction, PlayerAction
from blackjack.cards import Card


class EventType(str, Enum):
    SHOE_BURNED = "shoe_burned"
    PLAYER_CARD_DEALT = "player_card_dealt"
    DEALER_UPCARD_DEALT = "dealer_upcard_dealt"
    DEALER_HOLE_CARD_DEALT = "dealer_hole_card_dealt"
    INSURANCE_DECIDED = "insurance_decided"
    DEALER_PEEKED = "dealer_peeked"
    PLAYER_ACTED = "player_acted"
    HAND_SPLIT = "hand_split"
    DEALER_HOLE_REVEALED = "dealer_hole_revealed"
    DEALER_HIT = "dealer_hit"
    ROUND_SETTLED = "round_settled"


class EventVisibility(str, Enum):
    INTERNAL = "internal"
    PUBLIC = "public"


@dataclass(frozen=True, slots=True)
class InternalEvent:
    """An authoritative event. Optional fields are type-specific payloads."""

    sequence: int
    event_type: EventType
    visibility: EventVisibility
    card: Card | None = None
    hand_index: int | None = None
    player_action: PlayerAction | None = None
    insurance_action: InsuranceAction | None = None
    amount: Fraction | None = None


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """A redacted event safe to expose to a player or model pipeline."""

    sequence: int
    event_type: EventType
    card: Card | None = None
    hand_index: int | None = None
    player_action: PlayerAction | None = None
    insurance_action: InsuranceAction | None = None
    amount: Fraction | None = None


def public_events(events: tuple[InternalEvent, ...]) -> tuple[PublicEvent, ...]:
    """Redact internal-only events without transforming hidden payloads."""

    return tuple(
        PublicEvent(
            sequence=event.sequence,
            event_type=event.event_type,
            card=event.card,
            hand_index=event.hand_index,
            player_action=event.player_action,
            insurance_action=event.insurance_action,
            amount=event.amount,
        )
        for event in events
        if event.visibility is EventVisibility.PUBLIC
    )
