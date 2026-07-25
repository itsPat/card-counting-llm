from __future__ import annotations

from fractions import Fraction

from blackjack import InsuranceAction, PlayerAction
from blackjack.oracle import (
    CardValue,
    Composition,
    OracleHand,
    PeekCondition,
    PlayerSituation,
    evaluate_actions,
    evaluate_insurance,
    optimal_action,
    optimal_insurance,
)


def situation(
    hand: tuple[CardValue, ...],
    upcard: CardValue,
    unseen: tuple[CardValue, ...],
    *,
    negative_peek: bool = False,
    from_split: bool = False,
) -> PlayerSituation:
    return PlayerSituation(
        composition=Composition.from_values(unseen),
        hand=OracleHand(
            hand,
            from_split=from_split,
            can_surrender=not from_split,
        ),
        dealer_upcard=upcard,
        peek_condition=(
            PeekCondition.NO_BLACKJACK if negative_peek else PeekCondition.NONE
        ),
    )


def test_surrender_has_constant_half_wager_loss() -> None:
    state = situation(
        (CardValue.TEN, CardValue.SIX),
        CardValue.TEN,
        (CardValue.TEN, CardValue.TEN, CardValue.TEN),
        negative_peek=True,
    )
    evaluations = {item.action: item for item in evaluate_actions(state)}
    surrender = evaluations[PlayerAction.SURRENDER]
    assert surrender.distribution.probability(Fraction(-1, 2)) == 1
    assert optimal_action(state).action is PlayerAction.SURRENDER


def test_double_is_optimal_when_every_available_card_makes_twenty_one() -> None:
    state = situation(
        (CardValue.FIVE, CardValue.SIX),
        CardValue.SIX,
        (
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
        ),
    )
    evaluations = {item.action: item for item in evaluate_actions(state)}
    assert evaluations[PlayerAction.DOUBLE].distribution.probability(2) == 1
    assert evaluations[PlayerAction.HIT].distribution.probability(1) == 1
    assert optimal_action(state).action is PlayerAction.DOUBLE


def test_split_evaluates_both_hands_against_the_shared_dealer() -> None:
    state = situation(
        (CardValue.EIGHT, CardValue.EIGHT),
        CardValue.SIX,
        (
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
        ),
    )
    evaluations = {item.action: item for item in evaluate_actions(state)}
    split = evaluations[PlayerAction.SPLIT]
    assert split.distribution.probability(2) == 1
    assert optimal_action(state).action is PlayerAction.SPLIT


def test_double_after_split_remains_legal() -> None:
    state = situation(
        (CardValue.FIVE, CardValue.SIX),
        CardValue.SIX,
        (
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
        ),
        from_split=True,
    )
    assert PlayerAction.DOUBLE in {
        evaluation.action for evaluation in evaluate_actions(state)
    }
    assert PlayerAction.SURRENDER not in {
        evaluation.action for evaluation in evaluate_actions(state)
    }


def test_split_aces_receive_one_card_and_pay_as_ordinary_hands() -> None:
    state = situation(
        (CardValue.ACE, CardValue.ACE),
        CardValue.SIX,
        (
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
            CardValue.TEN,
        ),
    )
    split = next(
        evaluation
        for evaluation in evaluate_actions(state)
        if evaluation.action is PlayerAction.SPLIT
    )
    assert split.distribution.probability(2) == 1


def test_insurance_break_even_and_recommendation() -> None:
    one_third_tens = Composition.from_values(
        (CardValue.TEN, CardValue.SIX, CardValue.SEVEN)
    )
    take, decline = evaluate_insurance(one_third_tens, CardValue.ACE)
    assert take.expected_profit == 0
    assert decline.expected_profit == 0
    assert (
        optimal_insurance(
            one_third_tens,
            CardValue.ACE,
        ).action
        is InsuranceAction.DECLINE
    )

    rich = Composition.from_values((CardValue.TEN, CardValue.TEN, CardValue.SIX))
    assert optimal_insurance(rich, CardValue.ACE).action is InsuranceAction.TAKE
