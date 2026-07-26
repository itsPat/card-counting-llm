"""Decision-only batches and deterministic training samplers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.data import Dataset

from blackjack.analysis import BetAction
from blackjack.dataset import (
    DatasetSplit,
    DecisionKind,
    DecisionToken,
    InsuranceToken,
    PlayToken,
    decision_example_from_json,
)
from blackjack.training.vocabulary import (
    BLACKJACK_VOCABULARY,
    BlackjackVocabulary,
)


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    input_ids: tuple[int, ...]
    target_id: int
    legal_token_ids: tuple[int, ...]
    kind: DecisionKind
    shoe_id: int
    decision_index: int

    def __post_init__(self) -> None:
        if not self.input_ids:
            raise ValueError("a training decision needs at least one input token")
        if self.target_id not in self.legal_token_ids:
            raise ValueError("training target must be a legal decision token")
        if len(set(self.legal_token_ids)) != len(self.legal_token_ids):
            raise ValueError("legal decision tokens must be unique")


class DecisionDataset(Dataset[EncodedDecision]):
    """An in-memory encoded split with no evaluation metadata in its items."""

    def __init__(
        self,
        examples: tuple[EncodedDecision, ...],
        *,
        vocabulary: BlackjackVocabulary = BLACKJACK_VOCABULARY,
        maximum_context_length: int = 256,
    ) -> None:
        if not examples:
            raise ValueError("decision dataset cannot be empty")
        if maximum_context_length <= 0:
            raise ValueError("maximum context length must be positive")
        if any(
            len(example.input_ids) > maximum_context_length
            for example in examples
        ):
            raise ValueError(
                "decision input exceeds the configured context length"
            )
        self._examples = examples
        self._vocabulary = vocabulary
        self._maximum_context_length = maximum_context_length

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        *,
        expected_split: DatasetSplit,
        vocabulary: BlackjackVocabulary = BLACKJACK_VOCABULARY,
        maximum_context_length: int = 256,
    ) -> DecisionDataset:
        examples: list[EncodedDecision] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = decision_example_from_json(line)
                if record.split is not expected_split:
                    raise ValueError(
                        f"{path}:{line_number} contains a "
                        f"{record.split.value} row"
                    )
                examples.append(
                    EncodedDecision(
                        input_ids=vocabulary.encode(record.input_tokens),
                        target_id=vocabulary.id_for(record.target_token),
                        legal_token_ids=vocabulary.encode(
                            record.metadata.legal_target_tokens
                        ),
                        kind=record.kind,
                        shoe_id=record.shoe_id,
                        decision_index=record.decision_index,
                    )
                )
        return cls(
            tuple(examples),
            vocabulary=vocabulary,
            maximum_context_length=maximum_context_length,
        )

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        split: DatasetSplit,
        *,
        vocabulary: BlackjackVocabulary = BLACKJACK_VOCABULARY,
        maximum_context_length: int = 256,
    ) -> DecisionDataset:
        return cls.from_jsonl(
            directory / f"{split.value}.jsonl",
            expected_split=split,
            vocabulary=vocabulary,
            maximum_context_length=maximum_context_length,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> EncodedDecision:
        return self._examples[index]

    @property
    def vocabulary(self) -> BlackjackVocabulary:
        return self._vocabulary

    @property
    def maximum_context_length(self) -> int:
        return self._maximum_context_length

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(example.target_id for example in self._examples)

    @property
    def shoe_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted({example.shoe_id for example in self._examples})
        )

    def before_shoe_id(self, exclusive_limit: int) -> DecisionDataset:
        """Return rows from the nested original shoe-ID prefix."""

        if exclusive_limit <= 0:
            raise ValueError("shoe ID limit must be positive")
        selected = tuple(
            example
            for example in self._examples
            if example.shoe_id < exclusive_limit
        )
        if not selected:
            raise ValueError("shoe ID limit selects no decisions")
        return DecisionDataset(
            selected,
            vocabulary=self._vocabulary,
            maximum_context_length=self._maximum_context_length,
        )


@dataclass(frozen=True, slots=True)
class DecisionBatch:
    input_ids: Tensor
    attention_mask: Tensor
    prediction_positions: Tensor
    target_ids: Tensor
    legal_token_mask: Tensor
    kinds: tuple[DecisionKind, ...]
    shoe_ids: Tensor
    decision_indices: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])


@dataclass(frozen=True, slots=True)
class DecisionCollator:
    vocabulary: BlackjackVocabulary = BLACKJACK_VOCABULARY

    def __call__(self, examples: Sequence[EncodedDecision]) -> DecisionBatch:
        if not examples:
            raise ValueError("cannot collate an empty decision batch")
        batch_size = len(examples)
        sequence_length = max(len(example.input_ids) for example in examples)
        input_ids = torch.full(
            (batch_size, sequence_length),
            self.vocabulary.pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, sequence_length),
            dtype=torch.bool,
        )
        prediction_positions = torch.empty(batch_size, dtype=torch.long)
        target_ids = torch.empty(batch_size, dtype=torch.long)
        legal_token_mask = torch.zeros(
            (batch_size, len(self.vocabulary)),
            dtype=torch.bool,
        )
        shoe_ids = torch.empty(batch_size, dtype=torch.long)
        decision_indices = torch.empty(batch_size, dtype=torch.long)

        for row, example in enumerate(examples):
            length = len(example.input_ids)
            input_ids[row, :length] = torch.tensor(
                example.input_ids,
                dtype=torch.long,
            )
            attention_mask[row, :length] = True
            prediction_positions[row] = length - 1
            target_ids[row] = example.target_id
            legal_token_mask[row, list(example.legal_token_ids)] = True
            shoe_ids[row] = example.shoe_id
            decision_indices[row] = example.decision_index

        return DecisionBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prediction_positions=prediction_positions,
            target_ids=target_ids,
            legal_token_mask=legal_token_mask,
            kinds=tuple(example.kind for example in examples),
            shoe_ids=shoe_ids,
            decision_indices=decision_indices,
        )


class SamplingStrategy(StrEnum):
    NATURAL = "natural"
    BALANCED = "balanced"


class CardOrderAugmentation(StrEnum):
    NONE = "none"
    PERMUTE = "permute"


@dataclass(frozen=True, slots=True)
class SamplingConfiguration:
    strategy: SamplingStrategy = SamplingStrategy.NATURAL
    seed: int = 20250731
    maximum_class_amplification: float = 10.0
    card_order_augmentation: CardOrderAugmentation = (
        CardOrderAugmentation.NONE
    )

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("sampling seed cannot be negative")
        if self.maximum_class_amplification < 1:
            raise ValueError("class amplification cap must be at least one")


def target_sampling_weights(
    dataset: DecisionDataset,
    *,
    maximum_class_amplification: float = 10.0,
) -> tuple[float, ...]:
    """Return capped inverse-square-root target-frequency weights."""

    if maximum_class_amplification < 1:
        raise ValueError("class amplification cap must be at least one")
    counts = Counter(dataset.targets)
    most_common_count = max(counts.values())
    return tuple(
        min(
            sqrt(most_common_count / counts[target]),
            maximum_class_amplification,
        )
        for target in dataset.targets
    )


class DecisionLoader:
    """Small deterministic epoch-aware loader with fully typed batches."""

    def __init__(
        self,
        dataset: DecisionDataset,
        *,
        batch_size: int,
        sampling: SamplingConfiguration,
        drop_last: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self._dataset = dataset
        self._batch_size = batch_size
        self._sampling = sampling
        self._drop_last = drop_last
        self._collator = DecisionCollator(dataset.vocabulary)

    def __len__(self) -> int:
        complete, remainder = divmod(len(self._dataset), self._batch_size)
        return complete if self._drop_last or remainder == 0 else complete + 1

    def batches(self, epoch: int) -> Iterator[DecisionBatch]:
        """Yield one deterministic epoch whose order changes by epoch."""

        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        generator = torch.Generator()
        generator.manual_seed(self._sampling.seed + epoch)
        if self._sampling.strategy is SamplingStrategy.NATURAL:
            sampled = torch.randperm(
                len(self._dataset),
                generator=generator,
            )
        else:
            weights = torch.tensor(
                target_sampling_weights(
                    self._dataset,
                    maximum_class_amplification=(
                        self._sampling.maximum_class_amplification
                    ),
                ),
                dtype=torch.float64,
            )
            sampled = torch.multinomial(
                weights,
                len(self._dataset),
                replacement=True,
                generator=generator,
            )
        indices = tuple(int(index) for index in sampled)
        for start in range(0, len(indices), self._batch_size):
            batch_indices = indices[start : start + self._batch_size]
            if self._drop_last and len(batch_indices) < self._batch_size:
                break
            yield self._collator(
                tuple(
                    _augment_card_order(
                        self._dataset[index],
                        self._dataset.vocabulary,
                        self._sampling,
                        epoch=epoch,
                        sample_position=start + offset,
                    )
                    for offset, index in enumerate(batch_indices)
                )
            )


def build_decision_loader(
    dataset: DecisionDataset,
    *,
    batch_size: int,
    sampling: SamplingConfiguration | None = None,
    drop_last: bool = False,
) -> DecisionLoader:
    """Build a deterministic natural or target-balanced decision loader."""

    configuration = (
        SamplingConfiguration() if sampling is None else sampling
    )
    return DecisionLoader(
        dataset,
        batch_size=batch_size,
        sampling=configuration,
        drop_last=drop_last,
    )


def _augment_card_order(
    example: EncodedDecision,
    vocabulary: BlackjackVocabulary,
    sampling: SamplingConfiguration,
    *,
    epoch: int,
    sample_position: int,
) -> EncodedDecision:
    if (
        sampling.card_order_augmentation
        is CardOrderAugmentation.NONE
    ):
        return example
    tokens = list(vocabulary.decode(example.input_ids))
    seed = _augmentation_seed(
        sampling.seed,
        epoch,
        example.shoe_id,
        example.decision_index,
        sample_position,
    )
    try:
        history_start = tokens.index("<HISTORY>") + 1
    except ValueError as error:
        raise ValueError("decision input lacks history structure") from error
    if example.kind is DecisionKind.BET:
        try:
            history_end = tokens.index("<BET_QUERY>", history_start)
        except ValueError as error:
            raise ValueError("bet input lacks query structure") from error
        _shuffle_slice(tokens, history_start, history_end, seed)
    else:
        try:
            history_end = tokens.index("<CURRENT_HAND>", history_start)
            player_start = tokens.index("<PLAYER>", history_end) + 1
            player_end = tokens.index("<DEALER>", player_start)
        except ValueError as error:
            raise ValueError(
                "play input lacks history/player/dealer structure"
            ) from error
        seed = _shuffle_slice(tokens, history_start, history_end, seed)
        _shuffle_slice(tokens, player_start, player_end, seed)
    return EncodedDecision(
        input_ids=vocabulary.encode(tuple(tokens)),
        target_id=example.target_id,
        legal_token_ids=example.legal_token_ids,
        kind=example.kind,
        shoe_id=example.shoe_id,
        decision_index=example.decision_index,
    )


def _augmentation_seed(
    seed: int,
    epoch: int,
    shoe_id: int,
    decision_index: int,
    sample_position: int,
) -> int:
    state = seed & ((1 << 64) - 1)
    for value in (epoch, shoe_id, decision_index, sample_position):
        state = _splitmix64(state ^ (value & ((1 << 64) - 1)))
    return state


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    mixed = (value + 0x9E3779B97F4A7C15) & mask
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
    return (mixed ^ (mixed >> 31)) & mask


def _shuffle_slice(
    tokens: list[str],
    start: int,
    end: int,
    state: int,
) -> int:
    for index in range(end - 1, start, -1):
        state = _splitmix64(state)
        selected = start + state % (index - start + 1)
        tokens[index], tokens[selected] = tokens[selected], tokens[index]
    return state


def legal_decision_logits(logits: Tensor, batch: DecisionBatch) -> Tensor:
    """Select query-position logits and mask every illegal output token."""

    if logits.ndim != 3:
        raise ValueError("model logits must have shape [batch, time, vocabulary]")
    if logits.shape[0] != batch.batch_size:
        raise ValueError("model logits and decision batch sizes differ")
    if logits.shape[2] != batch.legal_token_mask.shape[1]:
        raise ValueError("model and legal-mask vocabulary sizes differ")
    rows = torch.arange(batch.batch_size, device=logits.device)
    selected = logits[
        rows,
        batch.prediction_positions.to(logits.device),
    ]
    legal_mask = batch.legal_token_mask.to(logits.device)
    return selected.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def decision_cross_entropy(logits: Tensor, batch: DecisionBatch) -> Tensor:
    """Train only the single decision after each query token."""

    selected = legal_decision_logits(logits, batch)
    return functional.cross_entropy(
        selected,
        batch.target_ids.to(logits.device),
    )


def decision_accuracy(logits: Tensor, batch: DecisionBatch) -> float:
    """Measure exact target accuracy after applying the legal-token mask."""

    selected = legal_decision_logits(logits, batch)
    predictions = selected.argmax(dim=1)
    correct = predictions.eq(batch.target_ids.to(logits.device))
    return float(correct.float().mean().item())


def decode_decisions(
    logits: Tensor,
    batch: DecisionBatch,
    vocabulary: BlackjackVocabulary = BLACKJACK_VOCABULARY,
) -> tuple[DecisionToken, ...]:
    """Decode the highest-scoring legal token into its closed enum type."""

    selected = legal_decision_logits(logits, batch)
    predicted_ids = selected.argmax(dim=1).to("cpu")
    decoded: list[DecisionToken] = []
    for row, kind in enumerate(batch.kinds):
        token = vocabulary.token_for(int(predicted_ids[row].item()))
        if kind is DecisionKind.BET:
            decoded.append(BetAction(token))
        elif kind is DecisionKind.PLAY:
            decoded.append(PlayToken(token))
        else:
            decoded.append(InsuranceToken(token))
    return tuple(decoded)
