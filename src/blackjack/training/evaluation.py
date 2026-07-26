"""Evaluation-only objective values kept outside model training items."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from blackjack.dataset import (
    DatasetSplit,
    DecisionKind,
    decision_example_from_json,
)
from blackjack.training.vocabulary import BlackjackVocabulary


class DecisionObjective(StrEnum):
    EXPECTED_LOG_GROWTH = "expected_log_growth"
    EXPECTED_PROFIT = "expected_profit"


@dataclass(frozen=True, slots=True)
class EvaluationReference:
    shoe_id: int
    decision_index: int
    kind: DecisionKind
    objective: DecisionObjective
    values: tuple[tuple[int, float], ...]

    def regret(self, predicted_token_id: int) -> float:
        values = dict(self.values)
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
                objective = (
                    DecisionObjective.EXPECTED_LOG_GROWTH
                    if example.kind is DecisionKind.BET
                    else DecisionObjective.EXPECTED_PROFIT
                )
                values: list[tuple[int, float]] = []
                for action in example.metadata.action_values:
                    raw_value = (
                        action.expected_log_growth
                        if objective
                        is DecisionObjective.EXPECTED_LOG_GROWTH
                        else action.expected_profit
                    )
                    if raw_value is None:
                        raise ValueError(
                            f"{path}:{line_number} lacks {objective.value}"
                        )
                    values.append(
                        (
                            vocabulary.id_for(action.token),
                            float(raw_value),
                        )
                    )
                references.append(
                    EvaluationReference(
                        shoe_id=example.shoe_id,
                        decision_index=example.decision_index,
                        kind=example.kind,
                        objective=objective,
                        values=tuple(values),
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
