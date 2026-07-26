"""Rescore a retained model artifact against one dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blackjack.dataset import DatasetSplit
from blackjack.training.compare import load_model_artifact
from blackjack.training.data import DecisionDataset
from blackjack.training.evaluation import EvaluationReferenceIndex
from blackjack.training.metrics import (
    DecisionMetrics,
    decision_metrics_data,
)
from blackjack.training.run import (
    TrainingDevice,
    evaluate_model,
    resolve_device,
)


def evaluate_artifact(
    dataset_directory: Path,
    artifact_directory: Path,
    *,
    batch_size: int = 64,
    device_selection: TrainingDevice = TrainingDevice.AUTO,
) -> DecisionMetrics:
    """Evaluate the selected checkpoint without retraining it."""

    dataset = DecisionDataset.from_directory(
        dataset_directory,
        DatasetSplit.VALIDATION,
    )
    references = EvaluationReferenceIndex.from_jsonl(
        dataset_directory / "validation.jsonl",
        expected_split=DatasetSplit.VALIDATION,
        vocabulary=dataset.vocabulary,
    )
    device = resolve_device(device_selection)
    model = load_model_artifact(artifact_directory, dataset).to(device)
    return evaluate_model(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        references=references,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore a selected transformer checkpoint on validation data."
        ),
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=TrainingDevice,
        choices=tuple(TrainingDevice),
        default=TrainingDevice.AUTO,
    )
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    metrics = evaluate_artifact(
        arguments.dataset_directory,
        arguments.artifact_directory,
        batch_size=arguments.batch_size,
        device_selection=arguments.device,
    )
    print(
        json.dumps(
            {
                "artifact_directory": str(arguments.artifact_directory),
                "validation": decision_metrics_data(metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
