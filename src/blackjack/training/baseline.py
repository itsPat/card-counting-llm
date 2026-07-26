"""Simple legal-set-only baseline for interpreting transformer accuracy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import StrEnum
from math import log
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from blackjack.analysis import BetAction
from blackjack.dataset import (
    DatasetSplit,
    DecisionKind,
    InsuranceToken,
    PlayToken,
    play_token,
)
from blackjack.engine import PlayerAction
from blackjack.oracle import CardValue, basic_strategy_action
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


class BasicStrategyBaseline:
    """Use fixed basic strategy while ignoring all visible-card history."""

    def __init__(self, dataset: DecisionDataset) -> None:
        self._vocabulary = dataset.vocabulary

    def logits(self, batch: DecisionBatch) -> Tensor:
        logits = torch.zeros(
            (
                batch.batch_size,
                batch.input_ids.shape[1],
                len(self._vocabulary),
            ),
            dtype=torch.float32,
            device=batch.input_ids.device,
        )
        for row, kind in enumerate(batch.kinds):
            if kind is DecisionKind.BET:
                token = BetAction.MINIMUM.value
            elif kind is DecisionKind.INSURANCE:
                token = InsuranceToken.DECLINE.value
            else:
                token = play_token(
                    self._play_action(batch, row)
                ).value
            position = int(batch.prediction_positions[row].item())
            logits[row, position, self._vocabulary.id_for(token)] = 1
        return logits

    def _play_action(
        self,
        batch: DecisionBatch,
        row: int,
    ) -> PlayerAction:
        length = int(batch.prediction_positions[row].item()) + 1
        token_ids = tuple(
            int(token_id)
            for token_id in batch.input_ids[row, :length]
        )
        tokens = self._vocabulary.decode(token_ids)
        try:
            player_start = tokens.index("<PLAYER>") + 1
            dealer_index = tokens.index("<DEALER>", player_start)
            dealer_token = tokens[dealer_index + 1]
        except (ValueError, IndexError) as error:
            raise ValueError("play input lacks player/dealer structure") from error
        cards = tuple(
            CardValue(token) for token in tokens[player_start:dealer_index]
        )
        legal_actions = tuple(
            action
            for token, action in _PLAY_ACTIONS.items()
            if bool(
                batch.legal_token_mask[
                    row,
                    self._vocabulary.id_for(token.value),
                ].item()
            )
        )
        return basic_strategy_action(
            cards,
            CardValue(dealer_token),
            legal_actions,
        )


_PLAY_ACTIONS: dict[PlayToken, PlayerAction] = {
    PlayToken.HIT: PlayerAction.HIT,
    PlayToken.STAND: PlayerAction.STAND,
    PlayToken.DOUBLE: PlayerAction.DOUBLE,
    PlayToken.SPLIT: PlayerAction.SPLIT,
    PlayToken.SURRENDER: PlayerAction.SURRENDER,
}


class DecisionBaseline(Protocol):
    def logits(self, batch: DecisionBatch) -> Tensor: ...


def evaluate_decision_baseline(
    baseline: DecisionBaseline,
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


def evaluate_frequency_baseline(
    baseline: LegalFrequencyBaseline,
    dataset: DecisionDataset,
    *,
    batch_size: int = 256,
    references: EvaluationReferenceIndex | None = None,
) -> DecisionMetrics:
    return evaluate_decision_baseline(
        baseline,
        dataset,
        batch_size=batch_size,
        references=references,
    )


class BaselineKind(StrEnum):
    FREQUENCY = "frequency"
    BASIC_STRATEGY = "basic-strategy"


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
    parser.add_argument(
        "--policy",
        type=BaselineKind,
        choices=tuple(BaselineKind),
        default=BaselineKind.FREQUENCY,
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
    baseline: DecisionBaseline = (
        LegalFrequencyBaseline.fit(training)
        if arguments.policy is BaselineKind.FREQUENCY
        else BasicStrategyBaseline(training)
    )
    metrics = evaluate_decision_baseline(
        baseline,
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
