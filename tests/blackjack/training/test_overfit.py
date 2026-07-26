from __future__ import annotations

import pytest

from blackjack.dataset import DecisionKind
from blackjack.training import (
    BLACKJACK_VOCABULARY,
    DecisionDataset,
    EncodedDecision,
)
from blackjack.training.overfit import (
    OverfitConfiguration,
    run_tiny_overfit,
)


def _overfit_dataset() -> DecisionDataset:
    targets = ("<HIT>", "<STAND>", "<DOUBLE>", "<SURRENDER>")
    cards = ("2", "3", "4", "5", "6", "7", "8", "9")
    examples = tuple(
        EncodedDecision(
            input_ids=BLACKJACK_VOCABULARY.encode(
                ("<HISTORY>", card, "<PLAY_QUERY>")
            ),
            target_id=BLACKJACK_VOCABULARY.id_for(targets[index % 4]),
            legal_token_ids=BLACKJACK_VOCABULARY.encode(targets),
            kind=DecisionKind.PLAY,
            shoe_id=index,
            decision_index=index,
        )
        for index, card in enumerate(cards)
    )
    return DecisionDataset(examples)


def test_tiny_overfit_memorizes_a_deliberately_small_dataset() -> None:
    result = run_tiny_overfit(
        _overfit_dataset(),
        OverfitConfiguration(
            example_count=8,
            update_count=100,
            seed=17,
        ),
    )
    assert result.final_loss < result.initial_loss
    assert result.final_loss < 0.01
    assert result.final_accuracy == 1


def test_tiny_overfit_rejects_more_examples_than_exist() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        run_tiny_overfit(
            _overfit_dataset(),
            OverfitConfiguration(example_count=9),
        )
