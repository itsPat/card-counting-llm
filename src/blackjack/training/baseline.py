"""Simple legal-set-only baseline for interpreting transformer accuracy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import log
from pathlib import Path

import torch
from torch import Tensor

from blackjack.dataset import DatasetSplit
from blackjack.training.data import (
    DecisionBatch,
    DecisionDataset,
    build_decision_loader,
    decision_cross_entropy,
)
from blackjack.training.evaluation import EvaluationReferenceIndex
from blackjack.training.metrics import (
    DecisionMetricAccumulator,
    DecisionMetrics,
    decision_metrics_data,
)


@dataclass(frozen=True, slots=True)
class LegalFrequencyBaseline:
    """Score each legal token by its natural training-set frequency."""

    log_token_counts: Tensor

    @classmethod
    def fit(cls, dataset: DecisionDataset) -> LegalFrequencyBaseline:
        counts = [0] * len(dataset.vocabulary)
        for target in dataset.targets:
            counts[target] += 1
        return cls(
            torch.tensor(
                tuple(log(count) if count else 0.0 for count in counts),
                dtype=torch.float32,
            )
        )

    def logits(self, batch: DecisionBatch) -> Tensor:
        scores = self.log_token_counts.to(batch.input_ids.device)
        return scores.view(1, 1, -1).expand(
            batch.batch_size,
            batch.input_ids.shape[1],
            -1,
        )


def evaluate_frequency_baseline(
    baseline: LegalFrequencyBaseline,
    dataset: DecisionDataset,
    *,
    batch_size: int = 256,
    references: EvaluationReferenceIndex | None = None,
) -> DecisionMetrics:
    accumulator = DecisionMetricAccumulator(dataset.vocabulary, references)
    loader = build_decision_loader(dataset, batch_size=batch_size)
    for batch in loader.batches(epoch=0):
        logits = baseline.logits(batch)
        loss = decision_cross_entropy(logits, batch)
        accumulator.update(loss, logits, batch)
    return accumulator.finish()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the legal-set target-frequency control.",
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument(
        "--training-shoe-prefix",
        type=int,
        help="fit counts only on training rows below this shoe ID",
    )
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    training = DecisionDataset.from_directory(
        arguments.dataset_directory,
        DatasetSplit.TRAIN,
    )
    if arguments.training_shoe_prefix is not None:
        training = training.before_shoe_id(
            arguments.training_shoe_prefix
        )
    validation = DecisionDataset.from_directory(
        arguments.dataset_directory,
        DatasetSplit.VALIDATION,
    )
    references = EvaluationReferenceIndex.from_jsonl(
        arguments.dataset_directory / "validation.jsonl",
        expected_split=DatasetSplit.VALIDATION,
        vocabulary=validation.vocabulary,
    )
    metrics = evaluate_frequency_baseline(
        LegalFrequencyBaseline.fit(training),
        validation,
        references=references,
    )
    report: dict[str, object] = {
        "training_shoe_count": len(training.shoe_ids),
        "training_decision_count": len(training),
        "validation_shoe_count": len(validation.shoe_ids),
        "validation_decision_count": len(validation),
        "metrics": decision_metrics_data(metrics),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
