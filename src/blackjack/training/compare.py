"""Evaluate model behavior where an interpretable control agrees or differs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from blackjack.dataset import DatasetSplit, DecisionKind
from blackjack.training.baseline import BasicStrategyBaseline
from blackjack.training.data import (
    DecisionDataset,
    build_decision_loader,
    legal_decision_logits,
)
from blackjack.training.model import (
    BlackjackTransformer,
    PositionScheme,
    TransformerConfiguration,
)


class DecisionModel(Protocol):
    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    baseline_agreement_total: int
    baseline_agreement_model_correct: int
    baseline_deviation_total: int
    baseline_deviation_model_correct: int

    @property
    def baseline_agreement_model_accuracy(self) -> float:
        return (
            self.baseline_agreement_model_correct
            / self.baseline_agreement_total
        )

    @property
    def baseline_deviation_model_accuracy(self) -> float:
        return (
            self.baseline_deviation_model_correct
            / self.baseline_deviation_total
        )


def compare_model_with_basic_strategy(
    model: DecisionModel,
    dataset: DecisionDataset,
    *,
    batch_size: int = 256,
) -> BaselineComparison:
    """Split model play accuracy by basic-strategy agreement."""

    control = BasicStrategyBaseline(dataset)
    agreement_total = 0
    agreement_correct = 0
    deviation_total = 0
    deviation_correct = 0
    with torch.no_grad():
        for batch in build_decision_loader(
            dataset,
            batch_size=batch_size,
        ).batches(0):
            model_predictions = legal_decision_logits(
                model(batch.input_ids, batch.attention_mask),
                batch,
            ).argmax(dim=1)
            control_predictions = legal_decision_logits(
                control.logits(batch),
                batch,
            ).argmax(dim=1)
            for row, kind in enumerate(batch.kinds):
                if kind is not DecisionKind.PLAY:
                    continue
                target = int(batch.target_ids[row].item())
                prediction = int(model_predictions[row].item())
                control_prediction = int(
                    control_predictions[row].item()
                )
                if control_prediction == target:
                    agreement_total += 1
                    agreement_correct += int(prediction == target)
                else:
                    deviation_total += 1
                    deviation_correct += int(prediction == target)
    if agreement_total == 0 or deviation_total == 0:
        raise ValueError(
            "comparison needs both basic-strategy agreements and deviations"
        )
    return BaselineComparison(
        baseline_agreement_total=agreement_total,
        baseline_agreement_model_correct=agreement_correct,
        baseline_deviation_total=deviation_total,
        baseline_deviation_model_correct=deviation_correct,
    )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    untyped = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in untyped):
        raise ValueError(f"{field} keys must be strings")
    return {str(key): item for key, item in untyped.items()}


def _integer(data: dict[str, object], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(data: dict[str, object], field: str) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _model_configuration(artifact_directory: Path) -> TransformerConfiguration:
    raw: object = json.loads(
        (artifact_directory / "metrics.json").read_text(encoding="utf-8")
    )
    report = _mapping(raw, "training report")
    configuration = _mapping(
        report.get("model_configuration"),
        "model_configuration",
    )
    position = configuration.get(
        "position_scheme",
        PositionScheme.ABSOLUTE.value,
    )
    if not isinstance(position, str):
        raise ValueError("position_scheme must be a string")
    return TransformerConfiguration(
        vocabulary_size=_integer(configuration, "vocabulary_size"),
        context_length=_integer(configuration, "context_length"),
        embedding_dimension=_integer(
            configuration,
            "embedding_dimension",
        ),
        head_count=_integer(configuration, "head_count"),
        layer_count=_integer(configuration, "layer_count"),
        feed_forward_dimension=_integer(
            configuration,
            "feed_forward_dimension",
        ),
        dropout=_number(configuration, "dropout"),
        position_scheme=PositionScheme(position),
    )


def load_model_artifact(
    artifact_directory: Path,
    dataset: DecisionDataset,
) -> BlackjackTransformer:
    configuration = _model_configuration(artifact_directory)
    if configuration.vocabulary_size != len(dataset.vocabulary):
        raise ValueError("artifact and dataset vocabulary sizes differ")
    model = BlackjackTransformer(
        configuration,
        padding_index=dataset.vocabulary.pad_id,
    )
    loaded: object = torch.load(
        artifact_directory / "model.pt",
        weights_only=True,
        map_location="cpu",
    )
    if not isinstance(loaded, dict):
        raise ValueError("model artifact must contain a state dictionary")
    untyped_state = cast(dict[object, object], loaded)
    state: dict[str, Tensor] = {}
    for key, value in untyped_state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("model state dictionary is malformed")
        state[key] = value
    model.load_state_dict(state)
    model.eval()
    return model


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a trained model on basic-strategy deviations.",
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("artifact_directory", type=Path)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    validation = DecisionDataset.from_directory(
        arguments.dataset_directory,
        DatasetSplit.VALIDATION,
    )
    model = load_model_artifact(
        arguments.artifact_directory,
        validation,
    )
    comparison = compare_model_with_basic_strategy(model, validation)
    print(
        json.dumps(
            {
                "basic_strategy_agreement": {
                    "total": comparison.baseline_agreement_total,
                    "model_correct": (
                        comparison.baseline_agreement_model_correct
                    ),
                    "model_accuracy": (
                        comparison.baseline_agreement_model_accuracy
                    ),
                },
                "basic_strategy_deviation": {
                    "total": comparison.baseline_deviation_total,
                    "model_correct": (
                        comparison.baseline_deviation_model_correct
                    ),
                    "model_accuracy": (
                        comparison.baseline_deviation_model_accuracy
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
