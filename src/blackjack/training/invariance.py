"""Measure model invariance to order-preserving blackjack information."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from blackjack.dataset import DatasetSplit
from blackjack.training.compare import load_model_artifact
from blackjack.training.data import (
    CardOrderAugmentation,
    DecisionBatch,
    DecisionDataset,
    SamplingConfiguration,
    build_decision_loader,
    legal_decision_logits,
)
from blackjack.training.run import TrainingDevice, resolve_device


class DecisionModel(Protocol):
    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class PermutationConsistency:
    permutation_count: int
    total_comparisons: int
    changed_input_comparisons: int
    prediction_agreements: int
    changed_input_prediction_agreements: int
    original_correct: int
    permuted_correct: int

    @property
    def prediction_agreement(self) -> float:
        return self.prediction_agreements / self.total_comparisons

    @property
    def changed_input_prediction_agreement(self) -> float:
        return (
            self.changed_input_prediction_agreements
            / self.changed_input_comparisons
        )

    @property
    def original_accuracy(self) -> float:
        return self.original_correct / self.total_comparisons

    @property
    def permuted_accuracy(self) -> float:
        return self.permuted_correct / self.total_comparisons


def evaluate_permutation_consistency(
    model: DecisionModel,
    dataset: DecisionDataset,
    *,
    permutation_count: int = 4,
    batch_size: int = 256,
    seed: int = 20250802,
    device: torch.device | None = None,
) -> PermutationConsistency:
    """Compare predictions before and after deterministic valid permutations."""

    if permutation_count <= 0:
        raise ValueError("permutation count must be positive")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if seed < 0:
        raise ValueError("seed cannot be negative")
    selected_device = torch.device("cpu") if device is None else device
    original_loader = build_decision_loader(
        dataset,
        batch_size=batch_size,
        sampling=SamplingConfiguration(seed=seed),
    )
    permuted_loader = build_decision_loader(
        dataset,
        batch_size=batch_size,
        sampling=SamplingConfiguration(
            seed=seed,
            card_order_augmentation=CardOrderAugmentation.PERMUTE,
        ),
    )
    total = 0
    changed_total = 0
    agreements = 0
    changed_agreements = 0
    original_correct = 0
    permuted_correct = 0
    with torch.no_grad():
        for permutation in range(permutation_count):
            paired_batches = zip(
                original_loader.batches(permutation),
                permuted_loader.batches(permutation),
                strict=True,
            )
            for original_cpu, permuted_cpu in paired_batches:
                _validate_pair(original_cpu, permuted_cpu)
                changed = torch.any(
                    original_cpu.input_ids != permuted_cpu.input_ids,
                    dim=1,
                )
                original = _move_batch(original_cpu, selected_device)
                permuted = _move_batch(permuted_cpu, selected_device)
                original_predictions = legal_decision_logits(
                    model(original.input_ids, original.attention_mask),
                    original,
                ).argmax(dim=1)
                permuted_predictions = legal_decision_logits(
                    model(permuted.input_ids, permuted.attention_mask),
                    permuted,
                ).argmax(dim=1)
                matching = original_predictions == permuted_predictions
                changed_device = changed.to(selected_device)
                total += original.batch_size
                changed_total += int(changed.sum().item())
                agreements += int(matching.sum().item())
                changed_agreements += int(
                    (matching & changed_device).sum().item()
                )
                original_correct += int(
                    (original_predictions == original.target_ids).sum().item()
                )
                permuted_correct += int(
                    (permuted_predictions == permuted.target_ids).sum().item()
                )
    if changed_total == 0:
        raise ValueError("permutations did not change any decision input")
    return PermutationConsistency(
        permutation_count=permutation_count,
        total_comparisons=total,
        changed_input_comparisons=changed_total,
        prediction_agreements=agreements,
        changed_input_prediction_agreements=changed_agreements,
        original_correct=original_correct,
        permuted_correct=permuted_correct,
    )


def permutation_consistency_data(
    result: PermutationConsistency,
) -> dict[str, object]:
    return {
        **asdict(result),
        "prediction_agreement": result.prediction_agreement,
        "changed_input_prediction_agreement": (
            result.changed_input_prediction_agreement
        ),
        "original_accuracy": result.original_accuracy,
        "permuted_accuracy": result.permuted_accuracy,
    }


def _validate_pair(
    original: DecisionBatch,
    permuted: DecisionBatch,
) -> None:
    if (
        original.kinds != permuted.kinds
        or not torch.equal(original.shoe_ids, permuted.shoe_ids)
        or not torch.equal(
            original.decision_indices,
            permuted.decision_indices,
        )
        or not torch.equal(original.target_ids, permuted.target_ids)
        or not torch.equal(
            original.legal_token_mask,
            permuted.legal_token_mask,
        )
    ):
        raise AssertionError("permutation changed batch identity or labels")


def _move_batch(
    batch: DecisionBatch,
    device: torch.device,
) -> DecisionBatch:
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


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure validation prediction consistency under valid card-order "
            "permutations."
        ),
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("artifact_directory", type=Path, nargs="+")
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20250802)
    parser.add_argument(
        "--device",
        type=TrainingDevice,
        choices=tuple(TrainingDevice),
        default=TrainingDevice.AUTO,
    )
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    validation = DecisionDataset.from_directory(
        arguments.dataset_directory,
        DatasetSplit.VALIDATION,
    )
    device = resolve_device(arguments.device)
    reports: list[dict[str, object]] = []
    for artifact_directory in arguments.artifact_directory:
        model = load_model_artifact(artifact_directory, validation).to(device)
        result = evaluate_permutation_consistency(
            model,
            validation,
            permutation_count=arguments.permutations,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
            device=device,
        )
        reports.append(
            {
                "artifact_directory": str(artifact_directory),
                **permutation_consistency_data(result),
            }
        )
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
