from __future__ import annotations

import json
from pathlib import Path

import pytest

from blackjack.dataset import DecisionKind
from blackjack.training import (
    BLACKJACK_VOCABULARY,
    DecisionDataset,
    EncodedDecision,
)
from blackjack.training.model import TransformerConfiguration
from blackjack.training.run import (
    TrainingConfiguration,
    TrainingDevice,
    resolve_device,
    train_model,
    write_training_artifacts,
)


def _dataset() -> DecisionDataset:
    cards = ("2", "3", "4", "5", "6", "7", "8", "9")
    targets = ("<HIT>", "<STAND>")
    examples = tuple(
        EncodedDecision(
            input_ids=BLACKJACK_VOCABULARY.encode(
                ("<HISTORY>", card, "<PLAY_QUERY>")
            ),
            target_id=BLACKJACK_VOCABULARY.id_for(targets[index % 2]),
            legal_token_ids=BLACKJACK_VOCABULARY.encode(targets),
            kind=DecisionKind.PLAY,
            shoe_id=index,
            decision_index=index,
        )
        for index, card in enumerate(cards)
    )
    return DecisionDataset(examples, maximum_context_length=8)


def test_tiny_training_is_deterministic_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    model_configuration = TransformerConfiguration(
        vocabulary_size=len(dataset.vocabulary),
        context_length=8,
        embedding_dimension=8,
        head_count=2,
        layer_count=1,
        feed_forward_dimension=16,
        dropout=0,
    )
    training_configuration = TrainingConfiguration(
        epoch_count=2,
        batch_size=4,
        learning_rate=0.01,
        seed=31,
        device=TrainingDevice.CPU,
    )
    first_model, first = train_model(
        dataset,
        dataset,
        model_configuration,
        training_configuration,
        epoch_checkpoint_directory=tmp_path / "checkpoints",
        progress=False,
    )
    _, second = train_model(
        dataset,
        dataset,
        model_configuration,
        training_configuration,
        progress=False,
    )
    assert tuple(epoch.training for epoch in first.epochs) == tuple(
        epoch.training for epoch in second.epochs
    )
    assert tuple(epoch.validation for epoch in first.epochs) == tuple(
        epoch.validation for epoch in second.epochs
    )
    assert first.epochs[-1].training.mean_loss < first.epochs[0].training.mean_loss
    assert first.epochs[-1].validation.by_target
    assert (tmp_path / "checkpoints" / "epoch-01.pt").is_file()
    assert (tmp_path / "checkpoints" / "epoch-02.pt").is_file()

    write_training_artifacts(first_model, first, tmp_path)
    assert (tmp_path / "model.pt").is_file()
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["parameter_count"] == first.parameter_count
    assert len(report["epochs"]) == 2


def test_training_configuration_and_explicit_mps_are_validated() -> None:
    with pytest.raises(ValueError, match="epoch"):
        TrainingConfiguration(epoch_count=0)
    if resolve_device(TrainingDevice.AUTO).type != "mps":
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_device(TrainingDevice.MPS)
