from __future__ import annotations

import torch
from torch import Tensor

from blackjack.dataset import DecisionKind
from blackjack.training import (
    BLACKJACK_VOCABULARY,
    DecisionDataset,
    EncodedDecision,
)
from blackjack.training.compare import compare_model_with_basic_strategy


class _TargetRepeatingModel:
    def __init__(self, targets: tuple[int, ...]) -> None:
        self._targets = targets

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        del attention_mask
        logits = torch.zeros(
            (
                input_ids.shape[0],
                input_ids.shape[1],
                len(BLACKJACK_VOCABULARY),
            ),
        )
        for row, target in enumerate(self._targets):
            logits[row, -1, target] = 1
        return logits


def _play_example(
    index: int,
    target: str,
) -> EncodedDecision:
    return EncodedDecision(
        input_ids=BLACKJACK_VOCABULARY.encode(
            (
                "<HISTORY>",
                "<CURRENT_HAND>",
                "<PLAYER>",
                "10",
                "6",
                "<DEALER>",
                "10",
                "<PLAY_QUERY>",
            )
        ),
        target_id=BLACKJACK_VOCABULARY.id_for(target),
        legal_token_ids=BLACKJACK_VOCABULARY.encode(
            ("<HIT>", "<STAND>", "<SURRENDER>")
        ),
        kind=DecisionKind.PLAY,
        shoe_id=0,
        decision_index=index,
    )


def test_comparison_separates_control_agreements_and_deviations() -> None:
    dataset = DecisionDataset(
        (
            _play_example(0, "<SURRENDER>"),
            _play_example(1, "<HIT>"),
        )
    )
    model = _TargetRepeatingModel(
        (
            BLACKJACK_VOCABULARY.id_for("<SURRENDER>"),
            BLACKJACK_VOCABULARY.id_for("<SURRENDER>"),
        )
    )
    comparison = compare_model_with_basic_strategy(
        model,
        dataset,
        batch_size=2,
    )
    assert comparison.baseline_agreement_total == 1
    assert comparison.baseline_agreement_model_accuracy == 1
    assert comparison.baseline_deviation_total == 1
    assert comparison.baseline_deviation_model_accuracy == 0
