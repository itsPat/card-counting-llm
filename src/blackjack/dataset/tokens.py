"""Model-facing tokens for dataset examples."""

from __future__ import annotations

from enum import StrEnum

from blackjack.analysis import BetAction
from blackjack.engine import Card, InsuranceAction, ModelContext, PlayerAction
from blackjack.oracle import CardValue


class StructureToken(StrEnum):
    HISTORY = "<HISTORY>"
    CURRENT_HAND = "<CURRENT_HAND>"
    PLAYER = "<PLAYER>"
    DEALER = "<DEALER>"
    BET_QUERY = "<BET_QUERY>"
    PLAY_QUERY = "<PLAY_QUERY>"
    INSURANCE_QUERY = "<INSURANCE_QUERY>"


class PlayToken(StrEnum):
    HIT = "<HIT>"
    STAND = "<STAND>"
    DOUBLE = "<DOUBLE>"
    SPLIT = "<SPLIT>"
    SURRENDER = "<SURRENDER>"


class InsuranceToken(StrEnum):
    TAKE = "<INSURANCE>"
    DECLINE = "<NO_INSURANCE>"


DecisionToken = BetAction | PlayToken | InsuranceToken

_PLAY_TOKENS: dict[PlayerAction, PlayToken] = {
    PlayerAction.HIT: PlayToken.HIT,
    PlayerAction.STAND: PlayToken.STAND,
    PlayerAction.DOUBLE: PlayToken.DOUBLE,
    PlayerAction.SPLIT: PlayToken.SPLIT,
    PlayerAction.SURRENDER: PlayToken.SURRENDER,
}
_INSURANCE_TOKENS: dict[InsuranceAction, InsuranceToken] = {
    InsuranceAction.TAKE: InsuranceToken.TAKE,
    InsuranceAction.DECLINE: InsuranceToken.DECLINE,
}


def card_token(card: Card) -> str:
    """Collapse physical face-card ranks into the decision-equivalent 10."""

    return CardValue.from_card(card).value


def play_token(action: PlayerAction) -> PlayToken:
    return _PLAY_TOKENS[action]


def insurance_token(action: InsuranceAction) -> InsuranceToken:
    return _INSURANCE_TOKENS[action]


def encode_bet_input(history: tuple[Card, ...]) -> tuple[str, ...]:
    return (
        StructureToken.HISTORY.value,
        *(card_token(card) for card in history),
        StructureToken.BET_QUERY.value,
    )


def encode_decision_input(
    prior_history: tuple[Card, ...],
    context: ModelContext,
) -> tuple[str, ...]:
    query = (
        StructureToken.PLAY_QUERY
        if context.legal_player_actions
        else StructureToken.INSURANCE_QUERY
    )
    return (
        StructureToken.HISTORY.value,
        *(card_token(card) for card in (*prior_history, *context.history)),
        StructureToken.CURRENT_HAND.value,
        StructureToken.PLAYER.value,
        *(card_token(card) for card in context.current_hand),
        StructureToken.DEALER.value,
        card_token(context.dealer_upcard),
        query.value,
    )
