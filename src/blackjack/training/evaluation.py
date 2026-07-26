"""Evaluation-only objective values kept outside model training items."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import log1p
from pathlib import Path

from blackjack.analysis import SELECTED_BET_VOCABULARY
from blackjack.dataset import (
    DatasetSplit,
    DecisionKind,
    EvaluationMetadata,
    ReturnDistributionRecord,
    decision_example_from_json,
)
from blackjack.training.vocabulary import BlackjackVocabulary


class DecisionObjective(StrEnum):
    EXPECTED_PROFIT = "expected_profit"


@dataclass(frozen=True, slots=True)
class BetPolicyReference:
    """Half-Kelly policy errors for every legal discrete bet token."""

    fraction_errors: tuple[tuple[int, float], ...]
    absolute_log_growth_changes: tuple[tuple[int, float], ...]

    def errors(self, predicted_token_id: int) -> tuple[float, float]:
        fraction_errors = dict(self.fraction_errors)
        log_growth_changes = dict(self.absolute_log_growth_changes)
        if (
            predicted_token_id not in fraction_errors
            or predicted_token_id not in log_growth_changes
        ):
            raise ValueError("prediction is not legal for this reference")
        return (
            fraction_errors[predicted_token_id],
            log_growth_changes[predicted_token_id],
        )


@dataclass(frozen=True, slots=True)
class EvaluationReference:
    shoe_id: int
    decision_index: int
    kind: DecisionKind
    profit_values: tuple[tuple[int, float], ...] = ()
    bet_policy: BetPolicyReference | None = None

    def __post_init__(self) -> None:
        if (self.kind is DecisionKind.BET) != (self.bet_policy is not None):
            raise ValueError("only bet references may contain bet policy data")
        if (self.kind is DecisionKind.BET) == bool(self.profit_values):
            raise ValueError(
                "bet references need policy errors; other references need "
                "profit values"
            )

    def regret(self, predicted_token_id: int) -> float:
        values = dict(self.profit_values)
        if predicted_token_id not in values:
            raise ValueError("prediction is not legal for this reference")
        return max(values.values()) - values[predicted_token_id]


class EvaluationReferenceIndex:
    """Objective values addressed by dataset-row provenance."""

    def __init__(
        self,
        references: tuple[EvaluationReference, ...],
    ) -> None:
        keyed = {
            (reference.shoe_id, reference.decision_index): reference
            for reference in references
        }
        if len(keyed) != len(references):
            raise ValueError("evaluation references contain duplicate rows")
        self._keyed = keyed

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        *,
        expected_split: DatasetSplit,
        vocabulary: BlackjackVocabulary,
    ) -> EvaluationReferenceIndex:
        references: list[EvaluationReference] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                example = decision_example_from_json(line)
                if example.split is not expected_split:
                    raise ValueError(
                        f"{path}:{line_number} contains a "
                        f"{example.split.value} row"
                    )
                if example.kind is DecisionKind.BET:
                    references.append(
                        EvaluationReference(
                            shoe_id=example.shoe_id,
                            decision_index=example.decision_index,
                            kind=example.kind,
                            bet_policy=_bet_policy_reference(
                                example.metadata,
                                vocabulary,
                                path,
                                line_number,
                            ),
                        )
                    )
                    continue
                values: list[tuple[int, float]] = []
                for action in example.metadata.action_values:
                    if action.expected_profit is None:
                        raise ValueError(
                            f"{path}:{line_number} lacks expected_profit"
                        )
                    values.append(
                        (
                            vocabulary.id_for(action.token),
                            float(action.expected_profit),
                        )
                    )
                references.append(
                    EvaluationReference(
                        shoe_id=example.shoe_id,
                        decision_index=example.decision_index,
                        kind=example.kind,
                        profit_values=tuple(values),
                    )
                )
        return cls(tuple(references))

    def get(
        self,
        shoe_id: int,
        decision_index: int,
    ) -> EvaluationReference:
        try:
            return self._keyed[(shoe_id, decision_index)]
        except KeyError as error:
            raise ValueError(
                f"no evaluation reference for shoe {shoe_id} "
                f"decision {decision_index}"
            ) from error


def _bet_policy_reference(
    metadata: EvaluationMetadata,
    vocabulary: BlackjackVocabulary,
    path: Path,
    line_number: int,
) -> BetPolicyReference:
    half_kelly = metadata.continuous_half_kelly
    distribution = metadata.round_return_distribution
    if half_kelly is None or distribution is None:
        raise ValueError(
            f"{path}:{line_number} lacks half-Kelly bet policy metadata"
        )
    continuous_growth = _recorded_expected_log_growth(
        distribution,
        half_kelly,
    )
    fractions = {
        token.token.value: token.bankroll_fraction
        for token in SELECTED_BET_VOCABULARY.tokens
    }
    fraction_errors: list[tuple[int, float]] = []
    growth_changes: list[tuple[int, float]] = []
    for action in metadata.action_values:
        if action.expected_log_growth is None:
            raise ValueError(
                f"{path}:{line_number} lacks expected_log_growth"
            )
        try:
            fraction = fractions[action.token]
        except KeyError as error:
            raise ValueError(
                f"{path}:{line_number} has unknown bet token {action.token}"
            ) from error
        token_id = vocabulary.id_for(action.token)
        fraction_errors.append((token_id, abs(fraction - half_kelly)))
        growth_changes.append(
            (
                token_id,
                abs(action.expected_log_growth - continuous_growth),
            )
        )
    return BetPolicyReference(
        fraction_errors=tuple(fraction_errors),
        absolute_log_growth_changes=tuple(growth_changes),
    )


def _recorded_expected_log_growth(
    distribution: ReturnDistributionRecord,
    bankroll_fraction: float,
) -> float:
    return sum(
        float(outcome.probability)
        * log1p(bankroll_fraction * float(outcome.profit))
        for outcome in distribution.outcomes
    )
