"""Reference and production oracle adapters for model targets."""

from __future__ import annotations

from dataclasses import dataclass

from blackjack.analysis import SELECTED_BET_VOCABULARY, BetAction
from blackjack.dataset.records import (
    ActionValue,
    EvaluationMetadata,
    EvaluationMethod,
    MonteCarloMetadata,
    ReturnDistributionRecord,
)
from blackjack.dataset.tokens import insurance_token, play_token
from blackjack.engine import InsuranceAction, PlayerAction
from blackjack.oracle import (
    CardValue,
    Composition,
    InsuranceEvaluation,
    PlayerSituation,
    ReturnDistribution,
    RoundPlayerSituation,
    evaluate_insurance,
    evaluate_round_actions,
    expected_log_growth,
    fixed_policy_play_action_estimates,
    fixed_policy_play_rollout_seed,
    fixed_policy_rollout_seed,
    fixed_policy_round_return_estimate,
    kelly_recommendation,
    optimal_insurance,
    round_return_distribution,
)
from blackjack.oracle import (
    legal_actions as oracle_legal_actions,
)


class OracleStateMismatchError(RuntimeError):
    """Raised when the engine and oracle disagree about a decision state."""


@dataclass(frozen=True, slots=True)
class LabeledDecision:
    target_token: str
    metadata: EvaluationMetadata


class ExactDatasetOracle:
    """Translate engine decision states into exact composition-dependent labels."""

    __slots__ = ("_bet_worker_count",)

    def __init__(self, *, bet_worker_count: int = 1) -> None:
        if bet_worker_count <= 0:
            raise ValueError("bet worker count must be positive")
        self._bet_worker_count = bet_worker_count

    def label_bet(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        analysis = round_return_distribution(
            composition,
            unseen_unavailable=unseen_unavailable,
            worker_count=self._bet_worker_count,
        )
        return _bet_label(
            composition,
            unseen_unavailable,
            analysis.distribution,
            EvaluationMethod.RATIONAL_EXACT_CDP,
        )

    def label_insurance(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        evaluations = evaluate_insurance(
            composition,
            dealer_upcard=CardValue.ACE,
            unseen_unavailable=unseen_unavailable,
        )
        values = tuple(_insurance_value(evaluation) for evaluation in evaluations)
        target = optimal_insurance(
            composition,
            CardValue.ACE,
            unseen_unavailable,
        )
        return LabeledDecision(
            target_token=insurance_token(target.action).value,
            metadata=EvaluationMetadata(
                shoe_composition=composition,
                unseen_unavailable=unseen_unavailable,
                legal_target_tokens=tuple(value.token for value in values),
                action_values=values,
            ),
        )

    def label_play(
        self,
        situation: RoundPlayerSituation,
        legal_actions: tuple[PlayerAction, ...],
    ) -> LabeledDecision:
        evaluations = evaluate_round_actions(situation)
        oracle_actions = tuple(evaluation.action for evaluation in evaluations)
        if oracle_actions != legal_actions:
            raise OracleStateMismatchError(
                "engine legal actions do not match the exact oracle: "
                f"engine={legal_actions!r}, oracle={oracle_actions!r}"
            )
        values = tuple(
            ActionValue(
                token=play_token(evaluation.action).value,
                expected_profit=evaluation.expected_profit,
                return_distribution=ReturnDistributionRecord.from_oracle(
                    evaluation.distribution
                ),
            )
            for evaluation in evaluations
        )
        target = max(
            evaluations,
            key=lambda evaluation: evaluation.expected_profit,
        )
        return LabeledDecision(
            target_token=play_token(target.action).value,
            metadata=EvaluationMetadata(
                shoe_composition=situation.composition,
                unseen_unavailable=situation.unseen_unavailable,
                legal_target_tokens=tuple(value.token for value in values),
                action_values=values,
            ),
        )


class ProductionDatasetOracle(ExactDatasetOracle):
    """Use seeded native rollouts for production bet and play labels."""

    __slots__ = (
        "_bet_rollout_seed",
        "_bet_rollouts",
        "_play_rollout_seed",
        "_play_rollouts",
    )

    def __init__(
        self,
        *,
        bet_rollout_seed: int = 20250728,
        bet_rollouts: int = 1_000_000,
        play_rollout_seed: int = 20250730,
        play_rollouts: int = 1_000_000,
    ) -> None:
        super().__init__()
        if not 0 <= bet_rollout_seed < 2**64:
            raise ValueError("bet rollout seed must fit in unsigned 64 bits")
        if bet_rollouts <= 0:
            raise ValueError("bet rollout count must be positive")
        if not 0 <= play_rollout_seed < 2**64:
            raise ValueError("play rollout seed must fit in unsigned 64 bits")
        if play_rollouts <= 0:
            raise ValueError("play rollout count must be positive")
        self._bet_rollout_seed = bet_rollout_seed
        self._bet_rollouts = bet_rollouts
        self._play_rollout_seed = play_rollout_seed
        self._play_rollouts = play_rollouts

    def label_bet(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        seed = fixed_policy_rollout_seed(
            composition,
            unseen_unavailable,
            self._bet_rollout_seed,
        )
        estimate = fixed_policy_round_return_estimate(
            composition,
            unseen_unavailable=unseen_unavailable,
            seed=seed,
            rollouts=self._bet_rollouts,
        )
        return _bet_label(
            composition,
            unseen_unavailable,
            estimate.distribution,
            EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_H17,
            monte_carlo=MonteCarloMetadata(
                seed=estimate.seed,
                rollouts=estimate.rollouts,
                expected_profit_standard_error=(
                    estimate.expected_profit_standard_error
                ),
                expected_profit_confidence_interval_95=(
                    estimate.expected_profit_confidence_interval_95
                ),
            ),
        )

    def label_play(
        self,
        situation: RoundPlayerSituation,
        legal_actions: tuple[PlayerAction, ...],
    ) -> LabeledDecision:
        oracle_actions = oracle_legal_actions(
            PlayerSituation(
                composition=situation.composition,
                hand=situation.active_hand,
                dealer_upcard=situation.dealer_upcard,
                peek_condition=situation.peek_condition,
                rules=situation.rules,
                unseen_unavailable=situation.unseen_unavailable,
            ),
            hands_in_round=(
                1
                + len(situation.pending_hands)
                + len(situation.finished_hands)
            ),
        )
        if oracle_actions != legal_actions:
            raise OracleStateMismatchError(
                "engine legal actions do not match the rollout oracle: "
                f"engine={legal_actions!r}, oracle={oracle_actions!r}"
            )
        seed = fixed_policy_play_rollout_seed(
            situation,
            self._play_rollout_seed,
        )
        estimates = fixed_policy_play_action_estimates(
            situation,
            legal_actions,
            seed=seed,
            rollouts=self._play_rollouts,
        )
        values = tuple(
            ActionValue(
                token=play_token(estimate.action).value,
                expected_profit=estimate.expected_profit,
                return_distribution=ReturnDistributionRecord.from_oracle(
                    estimate.distribution
                ),
                monte_carlo=MonteCarloMetadata(
                    seed=estimate.seed,
                    rollouts=estimate.rollouts,
                    expected_profit_standard_error=(
                        estimate.expected_profit_standard_error
                    ),
                    expected_profit_confidence_interval_95=(
                        estimate.expected_profit_confidence_interval_95
                    ),
                ),
            )
            for estimate in estimates
        )
        target = max(
            estimates,
            key=lambda estimate: estimate.expected_profit,
        )
        return LabeledDecision(
            target_token=play_token(target.action).value,
            metadata=EvaluationMetadata(
                shoe_composition=situation.composition,
                unseen_unavailable=situation.unseen_unavailable,
                legal_target_tokens=tuple(value.token for value in values),
                action_values=values,
                evaluation_method=(
                    EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_CONTINUATION_H17
                ),
            ),
        )


def _bet_label(
    composition: Composition,
    unseen_unavailable: int,
    distribution: ReturnDistribution,
    evaluation_method: EvaluationMethod,
    *,
    monte_carlo: MonteCarloMetadata | None = None,
) -> LabeledDecision:
    recommendation = kelly_recommendation(distribution)
    selected_fraction = SELECTED_BET_VOCABULARY.nearest_fraction(
        recommendation.half_kelly
    )
    selected_token = next(
        token.token
        for token in SELECTED_BET_VOCABULARY.tokens
        if token.bankroll_fraction == selected_fraction
    )
    values = tuple(
        ActionValue(
            token=token.token.value,
            expected_log_growth=expected_log_growth(
                distribution,
                token.bankroll_fraction,
            ),
        )
        for token in SELECTED_BET_VOCABULARY.tokens
    )
    return LabeledDecision(
        target_token=selected_token.value,
        metadata=EvaluationMetadata(
            shoe_composition=composition,
            unseen_unavailable=unseen_unavailable,
            legal_target_tokens=tuple(value.token for value in values),
            action_values=values,
            evaluation_method=evaluation_method,
            round_return_distribution=ReturnDistributionRecord.from_oracle(
                distribution
            ),
            continuous_half_kelly=recommendation.half_kelly,
            selected_bet_fraction=selected_fraction,
            monte_carlo=monte_carlo,
        ),
    )


def _insurance_value(evaluation: InsuranceEvaluation) -> ActionValue:
    return ActionValue(
        token=insurance_token(evaluation.action).value,
        expected_profit=evaluation.expected_profit,
        return_distribution=ReturnDistributionRecord.from_oracle(
            evaluation.distribution
        ),
    )


def bet_action_for_token(token: str) -> BetAction:
    return BetAction(token)


def insurance_action_for_token(token: str) -> InsuranceAction:
    return next(
        action for action in InsuranceAction if insurance_token(action).value == token
    )


def player_action_for_token(token: str) -> PlayerAction:
    return next(action for action in PlayerAction if play_token(action).value == token)
