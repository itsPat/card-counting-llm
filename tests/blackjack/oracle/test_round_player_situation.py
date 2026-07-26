from __future__ import annotations

from blackjack import PlayerAction
from blackjack.oracle import (
    CardValue,
    Composition,
    OracleHand,
    PeekCondition,
    ResolvedHand,
    RoundPlayerSituation,
    evaluate_round_actions,
)


def test_existing_hands_count_toward_the_resplit_limit() -> None:
    pair = OracleHand(
        (CardValue.EIGHT, CardValue.EIGHT),
        from_split=True,
        can_surrender=False,
    )
    resolved = ResolvedHand(
        total=18,
        wager=pair.wager,
        is_natural_blackjack=False,
        is_bust=False,
    )
    situation = RoundPlayerSituation(
        composition=Composition.from_values((CardValue.TEN,) * 8),
        active_hand=pair,
        pending_hands=(OracleHand((CardValue.TEN, CardValue.SEVEN)),),
        finished_hands=(resolved, resolved),
        dealer_upcard=CardValue.SIX,
        peek_condition=PeekCondition.NONE,
    )
    actions = tuple(
        evaluation.action for evaluation in evaluate_round_actions(situation)
    )
    assert PlayerAction.SPLIT not in actions
    assert actions == (
        PlayerAction.HIT,
        PlayerAction.STAND,
        PlayerAction.DOUBLE,
    )
