"""A deliberately disposable model for verifying the training boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from blackjack.dataset import DatasetSplit
from blackjack.training.data import (
    DecisionCollator,
    DecisionDataset,
    decision_accuracy,
    decision_cross_entropy,
)


class PipelineSmokeModel(nn.Module):
    """A small GRU used only to prove that a tiny batch can be memorized."""

    def __init__(
        self,
        vocabulary_size: int,
        *,
        embedding_dimension: int,
        hidden_dimension: int,
        padding_index: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
            padding_idx=padding_index,
        )
        self.recurrent = nn.GRU(
            embedding_dimension,
            hidden_dimension,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dimension, vocabulary_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        embedded = self.embedding(input_ids)
        hidden_states, _ = self.recurrent(embedded)
        return self.output(hidden_states)


@dataclass(frozen=True, slots=True)
class OverfitConfiguration:
    example_count: int = 16
    update_count: int = 100
    learning_rate: float = 0.01
    embedding_dimension: int = 32
    hidden_dimension: int = 64
    seed: int = 20250731

    def __post_init__(self) -> None:
        if self.example_count <= 0:
            raise ValueError("example count must be positive")
        if self.update_count <= 0:
            raise ValueError("update count must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if self.hidden_dimension <= 0:
            raise ValueError("hidden dimension must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")


@dataclass(frozen=True, slots=True)
class OverfitResult:
    example_count: int
    update_count: int
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float


def run_tiny_overfit(
    dataset: DecisionDataset,
    configuration: OverfitConfiguration | None = None,
) -> OverfitResult:
    """Memorize a few rows to catch wiring errors before real training."""

    if configuration is None:
        configuration = OverfitConfiguration()
    if configuration.example_count > len(dataset):
        raise ValueError("overfit example count exceeds the dataset size")

    generator = torch.Generator()
    generator.manual_seed(configuration.seed)
    torch.set_rng_state(generator.get_state())
    examples = tuple(
        dataset[index] for index in range(configuration.example_count)
    )
    batch = DecisionCollator(dataset.vocabulary)(examples)
    model = PipelineSmokeModel(
        len(dataset.vocabulary),
        embedding_dimension=configuration.embedding_dimension,
        hidden_dimension=configuration.hidden_dimension,
        padding_index=dataset.vocabulary.pad_id,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
    )

    model.train()
    initial_logits = model(batch.input_ids)
    initial_loss = float(
        decision_cross_entropy(initial_logits, batch).detach().item()
    )
    initial_accuracy = decision_accuracy(initial_logits, batch)

    for _ in range(configuration.update_count):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids)
        loss = decision_cross_entropy(logits, batch)
        # PyTorch's public stubs leave these two otherwise typed methods'
        # optional callback parameters unknown to Pyright.
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    model.eval()
    with torch.no_grad():
        final_logits = model(batch.input_ids)
        final_loss = float(decision_cross_entropy(final_logits, batch).item())
        final_accuracy = decision_accuracy(final_logits, batch)

    return OverfitResult(
        example_count=configuration.example_count,
        update_count=configuration.update_count,
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Intentionally overfit a few generated training rows.",
    )
    parser.add_argument(
        "dataset_directory",
        type=Path,
        help="directory containing the assembled train.jsonl split",
    )
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20250731)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    dataset = DecisionDataset.from_directory(
        arguments.dataset_directory,
        DatasetSplit.TRAIN,
    )
    result = run_tiny_overfit(
        dataset,
        OverfitConfiguration(
            example_count=arguments.examples,
            update_count=arguments.updates,
            seed=arguments.seed,
        ),
    )
    print(
        "tiny overfit: "
        f"{result.example_count} examples, {result.update_count} updates, "
        f"loss {result.initial_loss:.4f} -> {result.final_loss:.6f}, "
        f"accuracy {result.initial_accuracy:.1%} -> {result.final_accuracy:.1%}"
    )


if __name__ == "__main__":
    main()
