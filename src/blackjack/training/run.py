"""Deterministic training and evaluation for the blackjack transformer."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from blackjack.dataset import DatasetSplit
from blackjack.training.data import (
    DecisionBatch,
    DecisionDataset,
    SamplingConfiguration,
    SamplingStrategy,
    build_decision_loader,
    decision_cross_entropy,
)
from blackjack.training.evaluation import EvaluationReferenceIndex
from blackjack.training.metrics import (
    CategoryAccuracy,
    DecisionMetricAccumulator,
    DecisionMetrics,
    ObjectiveRegret,
)
from blackjack.training.model import (
    BlackjackTransformer,
    TransformerConfiguration,
)


class TrainingDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    MPS = "mps"


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    epoch_count: int = 8
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    seed: int = 20250801
    sampling_strategy: SamplingStrategy = SamplingStrategy.NATURAL
    maximum_class_amplification: float = 10.0
    device: TrainingDevice = TrainingDevice.AUTO

    def __post_init__(self) -> None:
        if self.epoch_count <= 0:
            raise ValueError("epoch count must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight decay cannot be negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient clip norm must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.maximum_class_amplification < 1:
            raise ValueError("class amplification cap must be at least one")


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    elapsed_seconds: float
    training: DecisionMetrics
    validation: DecisionMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_configuration: TransformerConfiguration
    training_configuration: TrainingConfiguration
    vocabulary_tokens: tuple[str, ...]
    parameter_count: int
    device: str
    best_epoch: int
    epochs: tuple[EpochResult, ...]


def resolve_device(selection: TrainingDevice) -> torch.device:
    if selection is TrainingDevice.MPS:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if (
        selection is TrainingDevice.AUTO
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch(batch: DecisionBatch, device: torch.device) -> DecisionBatch:
    return DecisionBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        prediction_positions=batch.prediction_positions.to(device),
        target_ids=batch.target_ids.to(device),
        legal_token_mask=batch.legal_token_mask.to(device),
        kinds=batch.kinds,
        shoe_ids=batch.shoe_ids.to(device),
        decision_indices=batch.decision_indices.to(device),
    )


def _seed_torch(seed: int) -> None:
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.set_rng_state(generator.get_state())


def evaluate_model(
    model: BlackjackTransformer,
    dataset: DecisionDataset,
    *,
    batch_size: int,
    device: torch.device,
    references: EvaluationReferenceIndex | None = None,
) -> DecisionMetrics:
    model.eval()
    accumulator = DecisionMetricAccumulator(dataset.vocabulary, references)
    loader = build_decision_loader(
        dataset,
        batch_size=batch_size,
        sampling=SamplingConfiguration(seed=0),
    )
    with torch.no_grad():
        for cpu_batch in loader.batches(epoch=0):
            batch = _move_batch(cpu_batch, device)
            logits = model(batch.input_ids, batch.attention_mask)
            loss = decision_cross_entropy(logits, batch)
            accumulator.update(loss, logits, batch)
    return accumulator.finish()


def train_model(
    training_dataset: DecisionDataset,
    validation_dataset: DecisionDataset,
    model_configuration: TransformerConfiguration,
    training_configuration: TrainingConfiguration,
    *,
    validation_references: EvaluationReferenceIndex | None = None,
    progress: bool = True,
) -> tuple[BlackjackTransformer, TrainingResult]:
    """Train from one recorded seed and return every epoch's metrics."""

    if training_dataset.vocabulary != validation_dataset.vocabulary:
        raise ValueError("training and validation vocabularies differ")
    if (
        model_configuration.vocabulary_size
        != len(training_dataset.vocabulary)
    ):
        raise ValueError("model and dataset vocabulary sizes differ")

    _seed_torch(training_configuration.seed)
    device = resolve_device(training_configuration.device)
    if device.type == "mps":
        torch.mps.manual_seed(training_configuration.seed)
    model = BlackjackTransformer(
        model_configuration,
        padding_index=training_dataset.vocabulary.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_configuration.learning_rate,
        weight_decay=training_configuration.weight_decay,
    )
    loader = build_decision_loader(
        training_dataset,
        batch_size=training_configuration.batch_size,
        sampling=SamplingConfiguration(
            strategy=training_configuration.sampling_strategy,
            seed=training_configuration.seed,
            maximum_class_amplification=(
                training_configuration.maximum_class_amplification
            ),
        ),
    )
    epochs: list[EpochResult] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None

    for epoch in range(training_configuration.epoch_count):
        started = perf_counter()
        model.train()
        accumulator = DecisionMetricAccumulator(
            training_dataset.vocabulary
        )
        for cpu_batch in loader.batches(epoch):
            batch = _move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.input_ids, batch.attention_mask)
            loss = decision_cross_entropy(logits, batch)
            accumulator.update(loss, logits, batch)
            # PyTorch leaves optional callback types unknown in its stubs.
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_configuration.gradient_clip_norm,
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        training_metrics = accumulator.finish()
        validation_metrics = evaluate_model(
            model,
            validation_dataset,
            batch_size=training_configuration.batch_size,
            device=device,
            references=validation_references,
        )
        result = EpochResult(
            epoch=epoch + 1,
            elapsed_seconds=perf_counter() - started,
            training=training_metrics,
            validation=validation_metrics,
        )
        epochs.append(result)
        if validation_metrics.mean_loss < best_validation_loss:
            best_validation_loss = validation_metrics.mean_loss
            best_epoch = result.epoch
            best_state = copy.deepcopy(model.state_dict())
        if progress:
            print(
                f"epoch {result.epoch:02d}: "
                f"train loss={training_metrics.mean_loss:.4f} "
                f"accuracy={training_metrics.accuracy:.1%}; "
                f"validation loss={validation_metrics.mean_loss:.4f} "
                f"accuracy={validation_metrics.accuracy:.1%}; "
                f"{result.elapsed_seconds:.1f}s",
                flush=True,
            )

    if best_state is None:
        raise AssertionError("positive epoch count must select a checkpoint")
    model.load_state_dict(best_state)
    return model, TrainingResult(
        model_configuration=model_configuration,
        training_configuration=training_configuration,
        vocabulary_tokens=training_dataset.vocabulary.tokens,
        parameter_count=model.parameter_count,
        device=str(device),
        best_epoch=best_epoch,
        epochs=tuple(epochs),
    )


def _category_data(metric: CategoryAccuracy) -> dict[str, int | float | str]:
    return {
        "category": metric.category,
        "correct": metric.correct,
        "total": metric.total,
        "accuracy": metric.accuracy,
    }


def _metrics_data(metrics: DecisionMetrics) -> dict[str, object]:
    return {
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


def _regret_data(metric: ObjectiveRegret) -> dict[str, int | float | str]:
    return {
        "category": metric.category,
        "objective": metric.objective.value,
        "total": metric.total,
        "mean_regret": metric.mean_regret,
        "percentile_95_regret": metric.percentile_95_regret,
        "maximum_regret": metric.maximum_regret,
    }


def _result_data(result: TrainingResult) -> dict[str, object]:
    return {
        "model_configuration": asdict(result.model_configuration),
        "training_configuration": asdict(result.training_configuration),
        "vocabulary_tokens": list(result.vocabulary_tokens),
        "parameter_count": result.parameter_count,
        "device": result.device,
        "best_epoch": result.best_epoch,
        "epochs": [
            {
                "epoch": epoch.epoch,
                "elapsed_seconds": epoch.elapsed_seconds,
                "training": _metrics_data(epoch.training),
                "validation": _metrics_data(epoch.validation),
            }
            for epoch in result.epochs
        ],
    }


def write_training_artifacts(
    model: BlackjackTransformer,
    result: TrainingResult,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_model = output_directory / ".model.pt.tmp"
    torch.save(model.state_dict(), temporary_model)
    temporary_model.replace(output_directory / "model.pt")
    temporary_metrics = output_directory / ".metrics.json.tmp"
    temporary_metrics.write_text(
        json.dumps(_result_data(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_metrics.replace(output_directory / "metrics.json")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the causal blackjack decision transformer.",
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--sampling",
        type=SamplingStrategy,
        choices=tuple(SamplingStrategy),
        default=SamplingStrategy.NATURAL,
    )
    parser.add_argument(
        "--device",
        type=TrainingDevice,
        choices=tuple(TrainingDevice),
        default=TrainingDevice.AUTO,
    )
    parser.add_argument("--embedding-dimension", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--feed-forward-dimension", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20250801)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    training_dataset = DecisionDataset.from_directory(
        arguments.dataset_directory,
        DatasetSplit.TRAIN,
    )
    validation_dataset = DecisionDataset.from_directory(
        arguments.dataset_directory,
        DatasetSplit.VALIDATION,
    )
    validation_references = EvaluationReferenceIndex.from_jsonl(
        arguments.dataset_directory / "validation.jsonl",
        expected_split=DatasetSplit.VALIDATION,
        vocabulary=validation_dataset.vocabulary,
    )
    model_configuration = TransformerConfiguration(
        vocabulary_size=len(training_dataset.vocabulary),
        context_length=training_dataset.maximum_context_length,
        embedding_dimension=arguments.embedding_dimension,
        head_count=arguments.heads,
        layer_count=arguments.layers,
        feed_forward_dimension=arguments.feed_forward_dimension,
        dropout=arguments.dropout,
    )
    training_configuration = TrainingConfiguration(
        epoch_count=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        sampling_strategy=arguments.sampling,
        device=arguments.device,
    )
    model, result = train_model(
        training_dataset,
        validation_dataset,
        model_configuration,
        training_configuration,
        validation_references=validation_references,
    )
    write_training_artifacts(model, result, arguments.output_directory)
    print(
        f"saved {result.parameter_count:,}-parameter model and metrics to "
        f"{arguments.output_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()
