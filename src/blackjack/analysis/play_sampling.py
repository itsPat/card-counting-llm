"""Bounded validation of Monte Carlo sampling against exact play values."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
from numpy.typing import NDArray

from blackjack.engine import PlayerAction
from blackjack.oracle import (
    ActionEvaluation,
    CardValue,
    Composition,
    OracleHand,
    PeekCondition,
    PlayerSituation,
    RoundPlayerSituation,
    evaluate_actions,
    fixed_policy_play_action_estimates,
    fixed_policy_play_rollout_seed,
)


@dataclass(frozen=True, slots=True)
class PlaySamplingConfiguration:
    """Every stochastic choice in the bounded sampling experiment."""

    seed: int = 20250729
    sample_budgets: tuple[int, ...] = (1_000, 10_000, 100_000, 1_000_000)
    replications: int = 200
    material_gap: float = 0.01

    def __post_init__(self) -> None:
        if not self.sample_budgets:
            raise ValueError("sampling validation needs at least one budget")
        if any(budget <= 0 for budget in self.sample_budgets):
            raise ValueError("sample budgets must be positive")
        if self.replications <= 0:
            raise ValueError("replication count must be positive")
        if self.material_gap < 0:
            raise ValueError("material action gap cannot be negative")


@dataclass(frozen=True, slots=True)
class ExactPlayState:
    """One tractable exact state retained as sampling ground truth."""

    name: str
    situation: PlayerSituation
    evaluations: tuple[ActionEvaluation, ...]
    optimal_action: PlayerAction
    optimal_expected_profit: float
    runner_up_gap: float


@dataclass(frozen=True, slots=True)
class PlaySamplingMetrics:
    sample_budget_per_action: int
    comparisons: int
    action_agreement: float
    material_comparisons: int
    material_action_agreement: float
    mean_exact_regret: float
    p95_exact_regret: float
    maximum_exact_regret: float


@dataclass(frozen=True, slots=True)
class PlaySamplingValidation:
    configuration: PlaySamplingConfiguration
    states: tuple[ExactPlayState, ...]
    metrics: tuple[PlaySamplingMetrics, ...]


@dataclass(frozen=True, slots=True)
class PlayRolloutConfiguration:
    """Budgets for real state rollouts under the fixed continuation policy."""

    seed: int = 20250730
    rollout_budgets_per_action: tuple[int, ...] = (
        10_000,
        100_000,
        1_000_000,
    )
    material_gap: float = 0.01

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**64:
            raise ValueError("play rollout seed must fit in unsigned 64 bits")
        if not self.rollout_budgets_per_action:
            raise ValueError("play rollout validation needs at least one budget")
        if any(budget <= 0 for budget in self.rollout_budgets_per_action):
            raise ValueError("play rollout budgets must be positive")
        if self.material_gap < 0:
            raise ValueError("material action gap cannot be negative")


@dataclass(frozen=True, slots=True)
class PlayRolloutMetrics:
    rollouts_per_action: int
    comparisons: int
    action_agreement: float
    material_comparisons: int
    material_action_agreement: float
    mean_exact_regret: float
    maximum_exact_regret: float
    mean_action_value_absolute_error: float
    maximum_action_value_absolute_error: float


@dataclass(frozen=True, slots=True)
class PlayRolloutValidation:
    configuration: PlayRolloutConfiguration
    states: tuple[ExactPlayState, ...]
    metrics: tuple[PlayRolloutMetrics, ...]


_CORPUS: tuple[
    tuple[str, tuple[CardValue, ...], CardValue],
    ...,
] = (
    ("hard 16 vs 10", (CardValue.TEN, CardValue.SIX), CardValue.TEN),
    ("hard 16 vs 6", (CardValue.TEN, CardValue.SIX), CardValue.SIX),
    ("hard 12 vs 4", (CardValue.TEN, CardValue.TWO), CardValue.FOUR),
    ("hard 11 vs 6", (CardValue.FIVE, CardValue.SIX), CardValue.SIX),
    ("hard 9 vs 2", (CardValue.FIVE, CardValue.FOUR), CardValue.TWO),
    ("hard 12 vs 3", (CardValue.SEVEN, CardValue.FIVE), CardValue.THREE),
    ("hard 17 vs A", (CardValue.TEN, CardValue.SEVEN), CardValue.ACE),
    ("hard 15 vs 10", (CardValue.TEN, CardValue.FIVE), CardValue.TEN),
    ("soft 18 vs 9", (CardValue.ACE, CardValue.SEVEN), CardValue.NINE),
    ("soft 18 vs 6", (CardValue.ACE, CardValue.SEVEN), CardValue.SIX),
    ("soft 17 vs 3", (CardValue.ACE, CardValue.SIX), CardValue.THREE),
    ("soft 13 vs 5", (CardValue.ACE, CardValue.TWO), CardValue.FIVE),
    (
        "three-card hard 12 vs 2",
        (CardValue.FIVE, CardValue.THREE, CardValue.FOUR),
        CardValue.TWO,
    ),
    (
        "three-card hard 16 vs 10",
        (CardValue.TEN, CardValue.TWO, CardValue.FOUR),
        CardValue.TEN,
    ),
)


def exact_play_sampling_corpus() -> tuple[ExactPlayState, ...]:
    """Evaluate a fixed full-shoe hard/soft corpus with the rational oracle."""

    states: list[ExactPlayState] = []
    for name, cards, upcard in _CORPUS:
        composition = Composition.full_shoe()
        for card in (*cards, upcard):
            composition = composition.remove(card)
        situation = PlayerSituation(
            composition=composition,
            hand=OracleHand(
                cards,
                can_double=len(cards) == 2,
                can_surrender=len(cards) == 2,
            ),
            dealer_upcard=upcard,
            peek_condition=(
                PeekCondition.NO_BLACKJACK
                if upcard in (CardValue.ACE, CardValue.TEN)
                else PeekCondition.NONE
            ),
        )
        evaluations = evaluate_actions(situation)
        ordered = sorted(
            (float(evaluation.expected_profit) for evaluation in evaluations),
            reverse=True,
        )
        optimal = max(
            evaluations,
            key=lambda evaluation: evaluation.expected_profit,
        )
        states.append(
            ExactPlayState(
                name=name,
                situation=situation,
                evaluations=evaluations,
                optimal_action=optimal.action,
                optimal_expected_profit=float(optimal.expected_profit),
                runner_up_gap=ordered[0] - ordered[1],
            )
        )
    return tuple(states)


def run_play_sampling_validation(
    configuration: PlaySamplingConfiguration | None = None,
    *,
    states: tuple[ExactPlayState, ...] | None = None,
) -> PlaySamplingValidation:
    """Measure sampling-only action error against exact return distributions."""

    active_configuration = (
        PlaySamplingConfiguration()
        if configuration is None
        else configuration
    )
    evaluated_states = exact_play_sampling_corpus() if states is None else states
    if not evaluated_states:
        raise ValueError("sampling validation needs at least one exact state")
    generator = np.random.default_rng(active_configuration.seed)
    metrics: list[PlaySamplingMetrics] = []
    for budget in active_configuration.sample_budgets:
        agreements = 0
        material_agreements = 0
        material_comparisons = 0
        regrets: list[float] = []
        for state in evaluated_states:
            exact_values = {
                evaluation.action: float(evaluation.expected_profit)
                for evaluation in state.evaluations
            }
            is_material = (
                state.runner_up_gap >= active_configuration.material_gap
            )
            for _ in range(active_configuration.replications):
                sampled_values = tuple(
                    (
                        evaluation.action,
                        _sample_mean(evaluation, budget, generator),
                    )
                    for evaluation in state.evaluations
                )
                selected = max(sampled_values, key=lambda item: item[1])[0]
                if selected is state.optimal_action:
                    agreements += 1
                    if is_material:
                        material_agreements += 1
                regrets.append(
                    state.optimal_expected_profit - exact_values[selected]
                )
                if is_material:
                    material_comparisons += 1

        comparisons = (
            len(evaluated_states) * active_configuration.replications
        )
        metrics.append(
            PlaySamplingMetrics(
                sample_budget_per_action=budget,
                comparisons=comparisons,
                action_agreement=agreements / comparisons,
                material_comparisons=material_comparisons,
                material_action_agreement=(
                    material_agreements / material_comparisons
                    if material_comparisons
                    else 1.0
                ),
                mean_exact_regret=sum(regrets) / comparisons,
                p95_exact_regret=_percentile(tuple(regrets), 0.95),
                maximum_exact_regret=max(regrets),
            )
        )
    return PlaySamplingValidation(
        configuration=active_configuration,
        states=evaluated_states,
        metrics=tuple(metrics),
    )


def run_play_rollout_validation(
    configuration: PlayRolloutConfiguration | None = None,
    *,
    states: tuple[ExactPlayState, ...] | None = None,
) -> PlayRolloutValidation:
    """Compare real fixed-continuation rollouts with exact rational values."""

    active_configuration = (
        PlayRolloutConfiguration()
        if configuration is None
        else configuration
    )
    evaluated_states = exact_play_sampling_corpus() if states is None else states
    if not evaluated_states:
        raise ValueError("play rollout validation needs at least one exact state")

    metrics: list[PlayRolloutMetrics] = []
    for budget in active_configuration.rollout_budgets_per_action:
        agreements = 0
        material_agreements = 0
        material_comparisons = 0
        regrets: list[float] = []
        value_errors: list[float] = []
        for state in evaluated_states:
            round_situation = _as_round_situation(state.situation)
            actions = tuple(evaluation.action for evaluation in state.evaluations)
            seed = fixed_policy_play_rollout_seed(
                round_situation,
                active_configuration.seed,
            )
            estimates = fixed_policy_play_action_estimates(
                round_situation,
                actions,
                seed=seed,
                rollouts=budget,
            )
            selected = max(
                estimates,
                key=lambda estimate: estimate.expected_profit,
            ).action
            exact_values = {
                evaluation.action: float(evaluation.expected_profit)
                for evaluation in state.evaluations
            }
            if selected is state.optimal_action:
                agreements += 1
            is_material = (
                state.runner_up_gap >= active_configuration.material_gap
            )
            if is_material:
                material_comparisons += 1
                if selected is state.optimal_action:
                    material_agreements += 1
            regrets.append(
                state.optimal_expected_profit - exact_values[selected]
            )
            value_errors.extend(
                abs(float(estimate.expected_profit) - exact_values[estimate.action])
                for estimate in estimates
            )

        comparisons = len(evaluated_states)
        metrics.append(
            PlayRolloutMetrics(
                rollouts_per_action=budget,
                comparisons=comparisons,
                action_agreement=agreements / comparisons,
                material_comparisons=material_comparisons,
                material_action_agreement=(
                    material_agreements / material_comparisons
                    if material_comparisons
                    else 1.0
                ),
                mean_exact_regret=sum(regrets) / comparisons,
                maximum_exact_regret=max(regrets),
                mean_action_value_absolute_error=(
                    sum(value_errors) / len(value_errors)
                ),
                maximum_action_value_absolute_error=max(value_errors),
            )
        )
    return PlayRolloutValidation(
        configuration=active_configuration,
        states=evaluated_states,
        metrics=tuple(metrics),
    )


def _as_round_situation(situation: PlayerSituation) -> RoundPlayerSituation:
    return RoundPlayerSituation(
        composition=situation.composition,
        active_hand=situation.hand,
        pending_hands=(),
        finished_hands=(),
        dealer_upcard=situation.dealer_upcard,
        peek_condition=situation.peek_condition,
        rules=situation.rules,
        unseen_unavailable=situation.unseen_unavailable,
    )


def _sample_mean(
    evaluation: ActionEvaluation,
    budget: int,
    generator: np.random.Generator,
) -> float:
    probabilities: NDArray[np.float64] = np.asarray(
        [
            float(outcome.probability)
            for outcome in evaluation.distribution.outcomes
        ],
        dtype=np.float64,
    )
    counts: NDArray[np.int64] = generator.multinomial(budget, probabilities)
    profits: NDArray[np.float64] = np.asarray(
        [float(outcome.profit) for outcome in evaluation.distribution.outcomes],
        dtype=np.float64,
    )
    return float(np.dot(counts, profits) / budget)


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, floor(percentile * len(ordered)))
    return ordered[index]
