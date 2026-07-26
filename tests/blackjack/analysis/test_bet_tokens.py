from __future__ import annotations

import pytest

from blackjack import PlayerAction
from blackjack.analysis import (
    SELECTED_BET_VOCABULARY,
    BetAction,
    BetVocabulary,
    EmpiricalReturnDistribution,
    EmpiricalReturnOutcome,
    ExactPlayState,
    PilotConfiguration,
    PilotObservation,
    PlayRolloutConfiguration,
    PlaySamplingConfiguration,
    SampledComposition,
    analyze_vocabulary,
    candidate_vocabularies,
    empirical_round_return_distribution,
    run_bet_token_pilot,
    run_play_rollout_validation,
    run_play_sampling_validation,
    sample_representative_compositions,
)
from blackjack.oracle import (
    ActionEvaluation,
    CardValue,
    Composition,
    OracleHand,
    PeekCondition,
    PlayerSituation,
    ReturnDistribution,
)


def test_representative_composition_sampling_is_reproducible_and_stratified() -> None:
    configuration = PilotConfiguration(
        seed=41,
        shoe_count=2,
        samples_per_shoe=3,
        rollouts_per_composition=10,
    )
    first = sample_representative_compositions(configuration)
    second = sample_representative_compositions(configuration)
    assert first == second
    assert len(first) == 6
    assert all(
        sample.composition.total == 312 - sample.visible_cards for sample in first
    )
    for shoe_seed in (41, 42):
        penetrations = [
            sample.penetration for sample in first if sample.shoe_seed == shoe_seed
        ]
        assert penetrations == sorted(penetrations)


def test_empirical_round_distribution_is_seeded_and_normalized() -> None:
    composition = Composition.from_values(
        (
            *((CardValue.ACE,) * 4),
            *((CardValue.FIVE,) * 8),
            *((CardValue.SIX,) * 8),
            *((CardValue.TEN,) * 24),
        )
    )
    first = empirical_round_return_distribution(
        composition,
        seed=99,
        rollouts=500,
    )
    second = empirical_round_return_distribution(
        composition,
        seed=99,
        rollouts=500,
    )
    assert first == second
    assert sum(outcome.probability for outcome in first.outcomes) == pytest.approx(1)


def test_all_ten_rounds_are_deterministic_pushes() -> None:
    distribution = empirical_round_return_distribution(
        Composition.from_values((CardValue.TEN,) * 20),
        seed=7,
        rollouts=50,
    )
    assert distribution.probability(0) == 1


def test_small_pilot_is_reproducible() -> None:
    configuration = PilotConfiguration(
        seed=17,
        shoe_count=1,
        samples_per_shoe=2,
        rollouts_per_composition=100,
    )
    assert run_bet_token_pilot(configuration) == run_bet_token_pilot(configuration)


def test_vocabulary_analysis_counts_classes_and_quantization_error() -> None:
    sample = SampledComposition(
        sample_index=0,
        shoe_seed=1,
        visible_cards=0,
        penetration=0,
        unseen_unavailable=1,
        composition=Composition.full_shoe(),
    )
    distribution = EmpiricalReturnDistribution(
        (
            EmpiricalReturnOutcome(-1, 0.4),
            EmpiricalReturnOutcome(1, 0.6),
        )
    )
    observations = (
        PilotObservation(sample, distribution, 0.2, 0.01, 0.001),
        PilotObservation(sample, distribution, 0.2, 0.01, 0.009),
    )
    vocabulary = BetVocabulary("test", (0.001, 0.005, 0.01))
    metrics = analyze_vocabulary(observations, vocabulary)
    assert metrics.class_counts == (1, 0, 1)
    assert metrics.mean_absolute_rounding_error == pytest.approx(0.0005)
    assert metrics.occupied_classes == 2


def test_candidate_vocabularies_have_stable_unique_tokens() -> None:
    vocabularies = candidate_vocabularies()
    assert SELECTED_BET_VOCABULARY in vocabularies
    assert [token.token for token in SELECTED_BET_VOCABULARY.tokens] == [
        BetAction.MINIMUM,
        BetAction.LOW,
        BetAction.MEDIUM,
        BetAction.HIGH,
    ]
    assert len({vocabulary.name for vocabulary in vocabularies}) == len(vocabularies)


def test_play_sampling_validation_is_reproducible_on_exact_distributions() -> None:
    situation = PlayerSituation(
        composition=Composition.from_values((CardValue.TEN,) * 8),
        hand=OracleHand((CardValue.FIVE, CardValue.SIX)),
        dealer_upcard=CardValue.SIX,
        peek_condition=PeekCondition.NONE,
    )
    evaluations = (
        ActionEvaluation(PlayerAction.HIT, ReturnDistribution.constant(1)),
        ActionEvaluation(PlayerAction.STAND, ReturnDistribution.constant(0)),
    )
    state = ExactPlayState(
        name="deterministic",
        situation=situation,
        evaluations=evaluations,
        optimal_action=PlayerAction.HIT,
        optimal_expected_profit=1,
        runner_up_gap=1,
    )
    configuration = PlaySamplingConfiguration(
        seed=42,
        sample_budgets=(10, 100),
        replications=5,
    )
    first = run_play_sampling_validation(configuration, states=(state,))
    second = run_play_sampling_validation(configuration, states=(state,))
    assert first == second
    assert all(metric.action_agreement == 1 for metric in first.metrics)
    assert all(metric.maximum_exact_regret == 0 for metric in first.metrics)


def test_real_play_rollout_validation_is_reproducible() -> None:
    situation = PlayerSituation(
        composition=Composition.from_values((CardValue.TEN,) * 20),
        hand=OracleHand((CardValue.FIVE, CardValue.SIX)),
        dealer_upcard=CardValue.SIX,
        peek_condition=PeekCondition.NONE,
    )
    evaluations = (
        ActionEvaluation(PlayerAction.HIT, ReturnDistribution.constant(1)),
        ActionEvaluation(PlayerAction.STAND, ReturnDistribution.constant(-1)),
    )
    state = ExactPlayState(
        name="deterministic rollout",
        situation=situation,
        evaluations=evaluations,
        optimal_action=PlayerAction.HIT,
        optimal_expected_profit=1,
        runner_up_gap=2,
    )
    configuration = PlayRolloutConfiguration(
        seed=43,
        rollout_budgets_per_action=(10, 100),
    )
    first = run_play_rollout_validation(configuration, states=(state,))
    second = run_play_rollout_validation(configuration, states=(state,))
    assert first == second
    assert all(metric.action_agreement == 1 for metric in first.metrics)
    assert all(metric.maximum_exact_regret == 0 for metric in first.metrics)
