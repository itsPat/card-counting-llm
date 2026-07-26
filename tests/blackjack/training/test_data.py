from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from blackjack.dataset import (
    ActionValue,
    DatasetSplit,
    DecisionExample,
    DecisionKind,
    EvaluationMetadata,
    decision_example_to_json,
)
from blackjack.oracle import Composition
from blackjack.training import (
    BLACKJACK_VOCABULARY,
    CardOrderAugmentation,
    DecisionBatch,
    DecisionCollator,
    DecisionDataset,
    DecisionLoader,
    EncodedDecision,
    SamplingConfiguration,
    SamplingStrategy,
    build_decision_loader,
    decision_cross_entropy,
    decode_decisions,
    legal_decision_logits,
    target_sampling_weights,
)


def _encoded(
    decision_index: int,
    target: str,
    *,
    inputs: tuple[str, ...] = ("<HISTORY>", "<PLAY_QUERY>"),
    legal: tuple[str, ...] = ("<HIT>", "<STAND>"),
) -> EncodedDecision:
    return EncodedDecision(
        input_ids=BLACKJACK_VOCABULARY.encode(inputs),
        target_id=BLACKJACK_VOCABULARY.id_for(target),
        legal_token_ids=BLACKJACK_VOCABULARY.encode(legal),
        kind=DecisionKind.PLAY,
        shoe_id=decision_index // 2,
        decision_index=decision_index,
    )


def _decision_order(batches: Iterable[DecisionBatch]) -> tuple[int, ...]:
    return tuple(
        int(index)
        for batch in batches
        for index in batch.decision_indices
    )


def test_vocabulary_has_stable_round_trip_and_padding_index() -> None:
    vocabulary = BLACKJACK_VOCABULARY
    assert len(vocabulary) == 29
    assert vocabulary.pad_id == 0
    tokens = ("<HISTORY>", "A", "10", "<PLAY_QUERY>", "<SPLIT>")
    assert vocabulary.decode(vocabulary.encode(tokens)) == tokens
    assert vocabulary.id_for("<SPLIT>") in vocabulary.decision_token_ids
    with pytest.raises(ValueError, match="unknown"):
        vocabulary.id_for("<UNKNOWN>")


def test_collator_right_pads_and_builds_legal_decision_masks() -> None:
    first = _encoded(0, "<HIT>")
    second = _encoded(
        1,
        "<STAND>",
        inputs=("<HISTORY>", "A", "<PLAY_QUERY>"),
    )
    batch = DecisionCollator()((first, second))
    assert batch.input_ids.shape == (2, 3)
    assert torch.equal(
        batch.attention_mask,
        torch.tensor(
            (
                (True, True, False),
                (True, True, True),
            ),
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        batch.prediction_positions,
        torch.tensor((1, 2), dtype=torch.long),
    )
    assert torch.equal(
        batch.target_ids,
        torch.tensor((first.target_id, second.target_id), dtype=torch.long),
    )
    assert batch.input_ids[0, 2].item() == BLACKJACK_VOCABULARY.pad_id
    assert torch.equal(
        batch.legal_token_mask.sum(dim=1),
        torch.tensor((2, 2)),
    )


def test_loss_uses_only_query_positions_and_masks_illegal_tokens() -> None:
    examples = (_encoded(0, "<HIT>"), _encoded(1, "<STAND>"))
    batch = DecisionCollator()(examples)
    logits = torch.zeros(
        (2, 3, len(BLACKJACK_VOCABULARY)),
        dtype=torch.float32,
    )
    logits[:, 0, :] = 100
    selected = legal_decision_logits(logits, batch)
    assert selected.shape == (2, len(BLACKJACK_VOCABULARY))
    assert torch.isfinite(
        selected[:, BLACKJACK_VOCABULARY.id_for("<HIT>")]
    ).all()
    assert selected[:, BLACKJACK_VOCABULARY.id_for("<BET_HIGH>")].max() < -1e30
    assert decision_cross_entropy(logits, batch).item() == pytest.approx(
        0.693147,
        rel=1e-5,
    )
    logits[0, 1, BLACKJACK_VOCABULARY.id_for("<HIT>")] = 2
    logits[1, 1, BLACKJACK_VOCABULARY.id_for("<STAND>")] = 2
    assert tuple(token.value for token in decode_decisions(logits, batch)) == (
        "<HIT>",
        "<STAND>",
    )


def test_balancing_uses_capped_inverse_square_root_frequency() -> None:
    examples = tuple(
        _encoded(index, "<HIT>" if index < 4 else "<STAND>")
        for index in range(5)
    )
    dataset = DecisionDataset(examples)
    assert target_sampling_weights(dataset) == (1, 1, 1, 1, 2)
    assert target_sampling_weights(
        dataset,
        maximum_class_amplification=1.5,
    ) == (1, 1, 1, 1, 1.5)


def test_epoch_loaders_are_seeded_and_change_order_between_epochs() -> None:
    dataset = DecisionDataset(
        tuple(
            _encoded(index, "<HIT>" if index < 6 else "<STAND>")
            for index in range(8)
        )
    )
    configuration = SamplingConfiguration(seed=91)
    first = build_decision_loader(
        dataset,
        batch_size=3,
        sampling=configuration,
    )
    second = build_decision_loader(
        dataset,
        batch_size=3,
        sampling=configuration,
    )
    first_epoch = _decision_order(first.batches(0))
    assert first_epoch == _decision_order(second.batches(0))
    assert first_epoch != _decision_order(first.batches(1))
    assert len(first) == 3

    balanced = build_decision_loader(
        dataset,
        batch_size=4,
        sampling=SamplingConfiguration(
            strategy=SamplingStrategy.BALANCED,
            seed=91,
        ),
    )
    assert sum(batch.batch_size for batch in balanced.batches(0)) == len(dataset)


def test_card_order_augmentation_is_deterministic_and_label_preserving() -> None:
    inputs = (
        "<HISTORY>",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "<CURRENT_HAND>",
        "<PLAYER>",
        "A",
        "6",
        "4",
        "<DEALER>",
        "10",
        "<PLAY_QUERY>",
    )
    example = _encoded(0, "<STAND>", inputs=inputs)
    dataset = DecisionDataset((example,))
    configuration = SamplingConfiguration(
        seed=91,
        card_order_augmentation=CardOrderAugmentation.PERMUTE,
    )
    first = build_decision_loader(
        dataset,
        batch_size=1,
        sampling=configuration,
    )
    second = build_decision_loader(
        dataset,
        batch_size=1,
        sampling=configuration,
    )

    def sequence(loader: DecisionLoader, epoch: int) -> tuple[str, ...]:
        batch = next(loader.batches(epoch))
        length = int(batch.prediction_positions[0].item()) + 1
        return BLACKJACK_VOCABULARY.decode(
            tuple(int(value) for value in batch.input_ids[0, :length])
        )

    epoch_zero = sequence(first, 0)
    assert epoch_zero == sequence(second, 0)
    permutations = {sequence(first, epoch) for epoch in range(4)}
    assert len(permutations) > 1
    for tokens in permutations:
        history_end = tokens.index("<CURRENT_HAND>")
        player_start = tokens.index("<PLAYER>") + 1
        player_end = tokens.index("<DEALER>")
        assert sorted(tokens[1:history_end]) == sorted(inputs[1:7])
        assert sorted(tokens[player_start:player_end]) == ["4", "6", "A"]
        assert tokens[player_end:] == ("<DEALER>", "10", "<PLAY_QUERY>")

    batch = next(first.batches(0))
    assert int(batch.target_ids[0].item()) == example.target_id
    assert batch.batch_size == len(dataset)


def test_dataset_can_select_a_nested_original_shoe_prefix() -> None:
    dataset = DecisionDataset(
        tuple(_encoded(index, "<HIT>") for index in range(8))
    )
    assert dataset.shoe_ids == (0, 1, 2, 3)
    prefix = dataset.before_shoe_id(2)
    assert prefix.shoe_ids == (0, 1)
    assert len(prefix) == 4
    with pytest.raises(ValueError, match="positive"):
        dataset.before_shoe_id(0)


def test_dataset_loads_only_model_fields_from_jsonl(tmp_path: Path) -> None:
    legal = ("<HIT>", "<STAND>")
    example = DecisionExample(
        schema_version=4,
        dataset_id="training-fixture",
        shoe_id=3,
        shoe_seed=4,
        split=DatasetSplit.TRAIN,
        round_index=0,
        decision_index=0,
        kind=DecisionKind.PLAY,
        input_tokens=("<HISTORY>", "A", "<PLAY_QUERY>"),
        target_token="<HIT>",
        behavior_token="<STAND>",
        metadata=EvaluationMetadata(
            shoe_composition=Composition.full_shoe(),
            unseen_unavailable=1,
            legal_target_tokens=legal,
            action_values=(
                ActionValue(token="<HIT>", expected_profit=Fraction(1, 10)),
                ActionValue(token="<STAND>", expected_profit=Fraction(0)),
            ),
        ),
    )
    path = tmp_path / "train.jsonl"
    path.write_text(decision_example_to_json(example), encoding="utf-8")
    dataset = DecisionDataset.from_jsonl(
        path,
        expected_split=DatasetSplit.TRAIN,
    )
    assert len(dataset) == 1
    assert dataset[0].shoe_id == 3
    assert dataset[0].input_ids == BLACKJACK_VOCABULARY.encode(
        example.input_tokens
    )
    assert dataset[0].target_id == BLACKJACK_VOCABULARY.id_for("<HIT>")

    with pytest.raises(ValueError, match="context length"):
        DecisionDataset.from_jsonl(
            path,
            expected_split=DatasetSplit.TRAIN,
            maximum_context_length=2,
        )
