"""Closed action and phase sets for a blackjack round."""

from enum import StrEnum


class PlayerAction(StrEnum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"


class InsuranceAction(StrEnum):
    TAKE = "take"
    DECLINE = "decline"


class RoundPhase(StrEnum):
    INSURANCE = "insurance"
    PLAYER_ACTIONS = "player_actions"
    DEALER_PLAY = "dealer_play"
    SETTLED = "settled"


class DecisionType(StrEnum):
    PLAY = "play"
    INSURANCE = "insurance"
