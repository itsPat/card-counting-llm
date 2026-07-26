"""Decision-level training metrics that preserve rare-target visibility."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from torch import Tensor

from blackjack.dataset import DecisionKind
from blackjack.training.data import DecisionBatch, legal_decision_logits
from blackjack.training.evaluation import (
    DecisionObjective,
    EvaluationReferenceIndex,
)
from blackjack.training.vocabulary import BlackjackVocabulary


@dataclass(frozen=True, slots=True)
class CategoryAccuracy:
    category: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True, slots=True)
class DecisionMetrics:
    mean_loss: float
    correct: int
    total: int
    by_kind: tuple[CategoryAccuracy, ...]
    by_target: tuple[CategoryAccuracy, ...]
    regret_by_kind: tuple[ObjectiveRegret, ...] = ()
    basic_strategy_comparison: StrategyComparison | None = None

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True, slots=True)
class ObjectiveRegret:
    category: str
    objective: DecisionObjective
    total: int
    mean_regret: float
    percentile_95_regret: float
    maximum_regret: float


@dataclass(frozen=True, slots=True)
class StrategyComparison:
    agreement_total: int
    agreement_model_correct: int
    deviation_total: int
    deviation_model_correct: int

    @property
    def agreement_model_accuracy(self) -> float:
        return self.agreement_model_correct / self.agreement_total

    @property
    def deviation_model_accuracy(self) -> float:
        return self.deviation_model_correct / self.deviation_total


@dataclass(slots=True)
class _MutableAccuracy:
    correct: int = 0
    total: int = 0

    def add(self, correct: bool) -> None:
        self.total += 1
        self.correct += int(correct)


class DecisionMetricAccumulator:
    """Accumulate exact accuracy without retaining model predictions."""

    def __init__(
        self,
        vocabulary: BlackjackVocabulary,
        references: EvaluationReferenceIndex | None = None,
    ) -> None:
        self._vocabulary = vocabulary
        self._references = references
        self._loss_sum = 0.0
        self._overall = _MutableAccuracy()
        self._by_kind = {
            kind: _MutableAccuracy() for kind in DecisionKind
        }
        self._by_target: dict[int, _MutableAccuracy] = {}
        self._regrets: dict[
            tuple[DecisionKind, DecisionObjective],
            list[float],
        ] = {}
        self._control_agreement = _MutableAccuracy()
        self._control_deviation = _MutableAccuracy()

    def update(
        self,
        loss: Tensor,
        logits: Tensor,
        batch: DecisionBatch,
        control_logits: Tensor | None = None,
    ) -> None:
        selected = legal_decision_logits(logits.detach(), batch)
        predictions = selected.argmax(dim=1).to("cpu")
        targets = batch.target_ids.to("cpu")
        control_predictions = (
            None
            if control_logits is None
            else legal_decision_logits(
                control_logits.detach(),
                batch,
            )
            .argmax(dim=1)
            .to("cpu")
        )
        self._loss_sum += float(loss.detach().item()) * batch.batch_size
        for row, kind in enumerate(batch.kinds):
            target = int(targets[row].item())
            correct = int(predictions[row].item()) == target
            self._overall.add(correct)
            self._by_kind[kind].add(correct)
            self._by_target.setdefault(target, _MutableAccuracy()).add(correct)
            if (
                control_predictions is not None
                and kind is DecisionKind.PLAY
            ):
                control_agrees = (
                    int(control_predictions[row].item()) == target
                )
                destination = (
                    self._control_agreement
                    if control_agrees
                    else self._control_deviation
                )
                destination.add(correct)
            if self._references is not None:
                reference = self._references.get(
                    int(batch.shoe_ids[row].item()),
                    int(batch.decision_indices[row].item()),
                )
                if reference.kind is not kind:
                    raise ValueError(
                        "evaluation reference decision kind differs"
                    )
                key = (kind, reference.objective)
                self._regrets.setdefault(key, []).append(
                    reference.regret(int(predictions[row].item()))
                )

    def finish(self) -> DecisionMetrics:
        if self._overall.total == 0:
            raise ValueError("cannot finish empty decision metrics")
        return DecisionMetrics(
            mean_loss=self._loss_sum / self._overall.total,
            correct=self._overall.correct,
            total=self._overall.total,
            by_kind=tuple(
                CategoryAccuracy(
                    category=kind.value,
                    correct=counts.correct,
                    total=counts.total,
                )
                for kind, counts in self._by_kind.items()
                if counts.total
            ),
            by_target=tuple(
                CategoryAccuracy(
                    category=self._vocabulary.token_for(target),
                    correct=counts.correct,
                    total=counts.total,
                )
                for target, counts in sorted(self._by_target.items())
            ),
            regret_by_kind=tuple(
                _objective_regret(kind, objective, regrets)
                for (kind, objective), regrets in self._regrets.items()
            ),
            basic_strategy_comparison=self._strategy_comparison(),
        )

    def _strategy_comparison(self) -> StrategyComparison | None:
        if (
            self._control_agreement.total == 0
            or self._control_deviation.total == 0
        ):
            return None
        return StrategyComparison(
            agreement_total=self._control_agreement.total,
            agreement_model_correct=self._control_agreement.correct,
            deviation_total=self._control_deviation.total,
            deviation_model_correct=self._control_deviation.correct,
        )


def _objective_regret(
    kind: DecisionKind,
    objective: DecisionObjective,
    regrets: list[float],
) -> ObjectiveRegret:
    ordered = sorted(regrets)
    percentile_index = ceil(0.95 * len(ordered)) - 1
    return ObjectiveRegret(
        category=kind.value,
        objective=objective,
        total=len(ordered),
        mean_regret=sum(ordered) / len(ordered),
        percentile_95_regret=ordered[percentile_index],
        maximum_regret=ordered[-1],
    )


def _category_data(
    metric: CategoryAccuracy,
) -> dict[str, int | float | str]:
    return {
        "category": metric.category,
        "correct": metric.correct,
        "total": metric.total,
        "accuracy": metric.accuracy,
    }


def _regret_data(
    metric: ObjectiveRegret,
) -> dict[str, int | float | str]:
    return {
        "category": metric.category,
        "objective": metric.objective.value,
        "total": metric.total,
        "mean_regret": metric.mean_regret,
        "percentile_95_regret": metric.percentile_95_regret,
        "maximum_regret": metric.maximum_regret,
    }


def decision_metrics_data(metrics: DecisionMetrics) -> dict[str, object]:
    """Convert typed metrics to a transparent JSON-compatible object."""

    result: dict[str, object] = {
        "mean_loss": metrics.mean_loss,
        "correct": metrics.correct,
        "total": metrics.total,
        "accuracy": metrics.accuracy,
        "by_kind": [_category_data(metric) for metric in metrics.by_kind],
        "by_target": [_category_data(metric) for metric in metrics.by_target],
        "regret_by_kind": [
            _regret_data(metric) for metric in metrics.regret_by_kind
        ],
    }
    comparison = metrics.basic_strategy_comparison
    if comparison is not None:
        result["basic_strategy_comparison"] = {
            "agreement_total": comparison.agreement_total,
            "agreement_model_correct": (
                comparison.agreement_model_correct
            ),
            "agreement_model_accuracy": (
                comparison.agreement_model_accuracy
            ),
            "deviation_total": comparison.deviation_total,
            "deviation_model_correct": (
                comparison.deviation_model_correct
            ),
            "deviation_model_accuracy": (
                comparison.deviation_model_accuracy
            ),
        }
    return result
