"""Immutable dataset rows, manifests, and evaluation-only metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from blackjack.analysis import SELECTED_BET_VOCABULARY, BetVocabulary
from blackjack.engine import FIXED_RULES, CasinoRules, Rank
from blackjack.oracle import Composition, ReturnDistribution


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DecisionKind(StrEnum):
    BET = "bet"
    INSURANCE = "insurance"
    PLAY = "play"


class EvaluationMethod(StrEnum):
    """The numerical and policy contract used to construct one target."""

    RATIONAL_EXACT_CDP = "rational_exact_cdp"
    FLOAT64_EXHAUSTIVE_CDZ_NO_RESPLIT = "float64_exhaustive_cdz_no_resplit"
    SEEDED_MONTE_CARLO_FIXED_H17 = (
        "seeded_monte_carlo_fixed_h17_basic_strategy"
    )
    SEEDED_MONTE_CARLO_FIXED_CONTINUATION_H17 = (
        "seeded_monte_carlo_fixed_h17_continuation"
    )


@dataclass(frozen=True, slots=True)
class ReturnOutcomeRecord:
    profit: Fraction
    probability: Fraction


@dataclass(frozen=True, slots=True)
class ReturnDistributionRecord:
    outcomes: tuple[ReturnOutcomeRecord, ...]

    @classmethod
    def from_oracle(
        cls,
        distribution: ReturnDistribution,
    ) -> ReturnDistributionRecord:
        return cls(
            tuple(
                ReturnOutcomeRecord(outcome.profit, outcome.probability)
                for outcome in distribution.outcomes
            )
        )


@dataclass(frozen=True, slots=True)
class ActionValue:
    """Evaluation data for one legal output token."""

    token: str
    expected_profit: Fraction | None = None
    expected_log_growth: float | None = None
    return_distribution: ReturnDistributionRecord | None = None
    monte_carlo: MonteCarloMetadata | None = None

    def __post_init__(self) -> None:
        if (self.expected_profit is None) == (self.expected_log_growth is None):
            raise ValueError("an action value needs exactly one objective value")


@dataclass(frozen=True, slots=True)
class MonteCarloMetadata:
    """Replay and uncertainty data for one empirical return distribution."""

    seed: int
    rollouts: int
    expected_profit_standard_error: float
    expected_profit_confidence_interval_95: tuple[float, float]

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**64:
            raise ValueError("Monte Carlo seed must fit in unsigned 64 bits")
        if self.rollouts <= 0:
            raise ValueError("Monte Carlo rollout count must be positive")
        if self.expected_profit_standard_error < 0:
            raise ValueError("Monte Carlo standard error cannot be negative")
        lower, upper = self.expected_profit_confidence_interval_95
        if lower > upper:
            raise ValueError("Monte Carlo confidence interval is reversed")


@dataclass(frozen=True, slots=True)
class EvaluationMetadata:
    """Data retained for analysis but excluded from the model input."""

    shoe_composition: Composition
    unseen_unavailable: int
    legal_target_tokens: tuple[str, ...]
    action_values: tuple[ActionValue, ...]
    evaluation_method: EvaluationMethod = EvaluationMethod.RATIONAL_EXACT_CDP
    round_return_distribution: ReturnDistributionRecord | None = None
    continuous_half_kelly: float | None = None
    selected_bet_fraction: float | None = None
    monte_carlo: MonteCarloMetadata | None = None

    def __post_init__(self) -> None:
        if self.unseen_unavailable < 0:
            raise ValueError("unseen unavailable count cannot be negative")
        if not self.legal_target_tokens:
            raise ValueError("a decision needs at least one legal target")
        if tuple(value.token for value in self.action_values) != (
            self.legal_target_tokens
        ):
            raise ValueError("action values must follow the legal-target order")
        uses_monte_carlo = (
            self.evaluation_method
            is EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_H17
        )
        if uses_monte_carlo != (self.monte_carlo is not None):
            raise ValueError(
                "round Monte Carlo metadata must match the evaluation method"
            )
        uses_action_monte_carlo = (
            self.evaluation_method
            is EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_CONTINUATION_H17
        )
        all_actions_have_monte_carlo = all(
            value.monte_carlo is not None for value in self.action_values
        )
        any_action_has_monte_carlo = any(
            value.monte_carlo is not None for value in self.action_values
        )
        if (
            uses_action_monte_carlo
            and not all_actions_have_monte_carlo
        ) or (
            not uses_action_monte_carlo
            and any_action_has_monte_carlo
        ):
            raise ValueError(
                "action Monte Carlo metadata must match the evaluation method"
            )


@dataclass(frozen=True, slots=True)
class DecisionExample:
    schema_version: int
    dataset_id: str
    shoe_id: int
    shoe_seed: int
    split: DatasetSplit
    round_index: int
    decision_index: int
    kind: DecisionKind
    input_tokens: tuple[str, ...]
    target_token: str
    behavior_token: str
    metadata: EvaluationMetadata

    def __post_init__(self) -> None:
        if self.target_token not in self.metadata.legal_target_tokens:
            raise ValueError("target token must be legal")
        if self.behavior_token not in self.metadata.legal_target_tokens:
            raise ValueError("behavior token must be legal")


@dataclass(frozen=True, slots=True)
class DatasetConfiguration:
    """Every stochastic and policy input needed to regenerate a dataset."""

    master_seed: int = 20250725
    split_seed: int = 20250726
    exploration_seed: int = 20250727
    shoe_count: int = 100
    train_fraction: Fraction = Fraction(4, 5)
    validation_fraction: Fraction = Fraction(1, 10)
    test_fraction: Fraction = Fraction(1, 10)
    exploration_probability: Fraction = Fraction(1, 5)
    bet_rollout_seed: int = 20250728
    bet_rollouts: int = 1_000_000
    play_rollout_seed: int = 20250730
    play_rollouts: int = 1_000_000
    bet_vocabulary: BetVocabulary = SELECTED_BET_VOCABULARY
    bet_evaluation_method: EvaluationMethod = (
        EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_H17
    )
    play_evaluation_method: EvaluationMethod = (
        EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_CONTINUATION_H17
    )
    rules: CasinoRules = FIXED_RULES

    def __post_init__(self) -> None:
        if self.shoe_count < 3:
            raise ValueError("at least three shoes are needed for three splits")
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(fraction <= 0 for fraction in fractions):
            raise ValueError("every dataset split must have positive weight")
        if sum(fractions, start=Fraction(0)) != 1:
            raise ValueError("dataset split fractions must sum to one")
        if not 0 <= self.exploration_probability <= 1:
            raise ValueError("exploration probability must lie in [0, 1]")
        if not 0 <= self.bet_rollout_seed < 2**64:
            raise ValueError("bet rollout seed must fit in unsigned 64 bits")
        if self.bet_rollouts <= 0:
            raise ValueError("bet rollout count must be positive")
        if not 0 <= self.play_rollout_seed < 2**64:
            raise ValueError("play rollout seed must fit in unsigned 64 bits")
        if self.play_rollouts <= 0:
            raise ValueError("play rollout count must be positive")
        if self.rules != FIXED_RULES:
            raise ValueError("dataset generation uses the fixed experiment rules")
        if self.bet_vocabulary != SELECTED_BET_VOCABULARY:
            raise ValueError("dataset generation uses the selected bet vocabulary")
        if (
            self.bet_evaluation_method
            is not EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_H17
        ):
            raise ValueError("dataset generation uses the production bet oracle")
        if (
            self.play_evaluation_method
            is not EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_CONTINUATION_H17
        ):
            raise ValueError("dataset generation uses the production play oracle")


@dataclass(frozen=True, slots=True)
class ShoeManifest:
    shoe_id: int
    seed: int
    split: DatasetSplit
    cards: tuple[Rank, ...]
    cut_card_position: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    dataset_id: str
    configuration: DatasetConfiguration
    shoes: tuple[ShoeManifest, ...]


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    manifest: DatasetManifest
    examples: tuple[DecisionExample, ...]

    def examples_for(self, split: DatasetSplit) -> tuple[DecisionExample, ...]:
        return tuple(example for example in self.examples if example.split is split)
