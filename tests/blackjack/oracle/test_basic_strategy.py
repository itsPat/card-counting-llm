from __future__ import annotations

import pytest

from blackjack.engine import PlayerAction
from blackjack.oracle import CardValue, basic_strategy_action

_PLAY = (PlayerAction.HIT, PlayerAction.STAND)
_INITIAL = (
    PlayerAction.HIT,
    PlayerAction.STAND,
    PlayerAction.DOUBLE,
    PlayerAction.SPLIT,
    PlayerAction.SURRENDER,
)


@pytest.mark.parametrize(
    ("cards", "dealer", "legal", "expected"),
    [
        (
            (CardValue.ACE, CardValue.ACE),
            CardValue.TEN,
            _INITIAL,
            PlayerAction.SPLIT,
        ),
        (
            (CardValue.TEN, CardValue.TEN),
            CardValue.SIX,
            _INITIAL,
            PlayerAction.STAND,
        ),
        (
            (CardValue.TEN, CardValue.SIX),
            CardValue.TEN,
            _INITIAL,
            PlayerAction.SURRENDER,
        ),
        (
            (CardValue.TEN, CardValue.SIX),
            CardValue.TEN,
            _PLAY,
            PlayerAction.HIT,
        ),
        (
            (CardValue.ACE, CardValue.SIX),
            CardValue.THREE,
            _INITIAL,
            PlayerAction.DOUBLE,
        ),
        (
            (CardValue.ACE, CardValue.TWO, CardValue.FOUR),
            CardValue.THREE,
            _PLAY,
            PlayerAction.HIT,
        ),
        (
            (CardValue.TEN, CardValue.TWO),
            CardValue.FOUR,
            _PLAY,
            PlayerAction.STAND,
        ),
        (
            (CardValue.FIVE, CardValue.SIX),
            CardValue.ACE,
            _INITIAL,
            PlayerAction.DOUBLE,
        ),
        (
            (CardValue.FOUR, CardValue.FIVE),
            CardValue.TWO,
            _INITIAL,
            PlayerAction.HIT,
        ),
    ],
)
def test_basic_strategy_boundaries(
    cards: tuple[CardValue, ...],
    dealer: CardValue,
    legal: tuple[PlayerAction, ...],
    expected: PlayerAction,
) -> None:
    assert basic_strategy_action(cards, dealer, legal) is expected


def test_basic_strategy_requires_a_hand_and_legal_action() -> None:
    with pytest.raises(ValueError, match="hand"):
        basic_strategy_action((), CardValue.SIX, _PLAY)
    with pytest.raises(ValueError, match="legal"):
        basic_strategy_action(
            (CardValue.TEN, CardValue.SIX),
            CardValue.SIX,
            (),
        )
