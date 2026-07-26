from __future__ import annotations

from fractions import Fraction

from blackjack import BlackjackRound, PlayerAction, Shoe, cards
from blackjack.dataset import (
    EvaluationMethod,
    ExactDatasetOracle,
    InsuranceToken,
    PlayToken,
    ProductionDatasetOracle,
    StructureToken,
    card_token,
    encode_bet_input,
    encode_decision_input,
    labeled_decision_from_json,
    labeled_decision_to_json,
)
from blackjack.oracle import (
    CardValue,
    Composition,
    OracleHand,
    PeekCondition,
    RoundPlayerSituation,
)


def test_model_inputs_match_the_minimal_documented_shape() -> None:
    game = BlackjackRound(
        Shoe.arranged(cards("J", "6", "Q", "9")),
        wager=1,
    )
    context = game.public_state.model_context
    assert context is not None
    assert encode_bet_input(cards("K", "2")) == (
        StructureToken.HISTORY,
        "10",
        "2",
        StructureToken.BET_QUERY,
    )
    assert encode_decision_input(cards("K", "2"), context) == (
        StructureToken.HISTORY,
        "10",
        "2",
        StructureToken.CURRENT_HAND,
        StructureToken.PLAYER,
        "10",
        "10",
        StructureToken.DEALER,
        "6",
        StructureToken.PLAY_QUERY,
    )
    assert card_token(cards("J")[0]) == card_token(cards("K")[0]) == "10"


def test_exact_insurance_label_retains_both_action_distributions() -> None:
    label = ExactDatasetOracle().label_insurance(
        Composition.from_values((CardValue.TEN, CardValue.TEN, CardValue.SIX)),
        unseen_unavailable=0,
    )
    assert label.target_token == InsuranceToken.TAKE
    assert label.metadata.legal_target_tokens == (
        InsuranceToken.TAKE,
        InsuranceToken.DECLINE,
    )
    assert all(
        value.return_distribution is not None for value in label.metadata.action_values
    )


def test_break_even_insurance_tie_resolves_to_decline() -> None:
    label = ExactDatasetOracle().label_insurance(
        Composition.from_values((CardValue.TEN, CardValue.SIX, CardValue.SEVEN)),
        unseen_unavailable=0,
    )
    assert label.target_token == InsuranceToken.DECLINE


def test_exact_play_label_uses_round_level_correlated_state() -> None:
    situation = RoundPlayerSituation(
        composition=Composition.from_values((CardValue.TEN,) * 8),
        active_hand=OracleHand((CardValue.FIVE, CardValue.SIX)),
        pending_hands=(
            OracleHand(
                (CardValue.TEN, CardValue.SEVEN),
                from_split=True,
                can_surrender=False,
            ),
        ),
        finished_hands=(),
        dealer_upcard=CardValue.SIX,
        peek_condition=PeekCondition.NONE,
    )
    legal = (
        PlayerAction.HIT,
        PlayerAction.STAND,
        PlayerAction.DOUBLE,
        PlayerAction.SURRENDER,
    )
    label = ExactDatasetOracle().label_play(situation, legal)
    assert label.target_token == PlayToken.DOUBLE
    assert label.metadata.legal_target_tokens == tuple(
        token.value
        for token in (
            PlayToken.HIT,
            PlayToken.STAND,
            PlayToken.DOUBLE,
            PlayToken.SURRENDER,
        )
    )
    assert all(
        value.expected_profit is not None and value.return_distribution is not None
        for value in label.metadata.action_values
    )


def test_exact_bet_label_uses_full_return_distribution_and_half_kelly() -> None:
    label = ExactDatasetOracle().label_bet(
        Composition.from_values((CardValue.TEN,) * 20),
        unseen_unavailable=0,
    )
    assert label.target_token == "<BET_MIN>"
    assert label.metadata.continuous_half_kelly == 0
    assert label.metadata.selected_bet_fraction == 0.001
    distribution = label.metadata.round_return_distribution
    assert distribution is not None
    assert distribution.outcomes[0].profit == Fraction(0)
    assert label.metadata.evaluation_method is EvaluationMethod.RATIONAL_EXACT_CDP


def test_production_bet_label_records_replay_and_uncertainty_metadata() -> None:
    label = ProductionDatasetOracle(
        bet_rollout_seed=71,
        bet_rollouts=1_000,
    ).label_bet(
        Composition.from_values((CardValue.TEN,) * 24),
        unseen_unavailable=2,
    )
    assert label.target_token == "<BET_MIN>"
    assert (
        label.metadata.evaluation_method
        is EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_H17
    )
    simulation = label.metadata.monte_carlo
    assert simulation is not None
    assert simulation.rollouts == 1_000
    assert simulation.expected_profit_standard_error == 0
    assert labeled_decision_from_json(labeled_decision_to_json(label)) == label


def test_production_play_label_records_per_action_rollout_metadata() -> None:
    composition = Composition.full_shoe()
    for card in (CardValue.FIVE, CardValue.SIX, CardValue.SIX):
        composition = composition.remove(card)
    situation = RoundPlayerSituation(
        composition=composition,
        active_hand=OracleHand((CardValue.FIVE, CardValue.SIX)),
        pending_hands=(),
        finished_hands=(),
        dealer_upcard=CardValue.SIX,
        peek_condition=PeekCondition.NONE,
    )
    legal = (
        PlayerAction.HIT,
        PlayerAction.STAND,
        PlayerAction.DOUBLE,
        PlayerAction.SURRENDER,
    )
    label = ProductionDatasetOracle(
        play_rollout_seed=72,
        play_rollouts=10_000,
    ).label_play(situation, legal)
    assert label.target_token == PlayToken.DOUBLE
    assert (
        label.metadata.evaluation_method
        is EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_CONTINUATION_H17
    )
    assert label.metadata.monte_carlo is None
    assert all(
        value.monte_carlo is not None
        and value.monte_carlo.rollouts == 10_000
        for value in label.metadata.action_values
    )
    assert labeled_decision_from_json(labeled_decision_to_json(label)) == label
