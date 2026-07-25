"""Closed action and phase sets for a blackjack round."""

from enum import Enum


class PlayerAction(str, Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"


class InsuranceAction(str, Enum):
    TAKE = "take"
    DECLINE = "decline"


class RoundPhase(str, Enum):
    INSURANCE = "insurance"
    PLAYER_ACTIONS = "player_actions"
    DEALER_PLAY = "dealer_play"
    SETTLED = "settled"
