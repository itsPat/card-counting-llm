import pytest
import torch
from torch import Tensor

from blackjack.dataset import DecisionKind
from blackjack.training.data import DecisionDataset, EncodedDecision
from blackjack.training.invariance import (
    evaluate_permutation_consistency,
    permutation_consistency_data,
)
from blackjack.training.vocabulary import BLACKJACK_VOCABULARY


class LastTokenModel:
    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        vocabulary_size = len(BLACKJACK_VOCABULARY)
        logits = torch.zeros(
            (batch_size, sequence_length, vocabulary_size),
        )
        logits[:, :, BLACKJACK_VOCABULARY.id_for("<HIT>")] = 1
        logits[:, :, BLACKJACK_VOCABULARY.id_for("<STAND>")] = 0
        return logits


def _dataset() -> DecisionDataset:
    vocabulary = BLACKJACK_VOCABULARY
    hit = vocabulary.id_for("<HIT>")
    stand = vocabulary.id_for("<STAND>")
    return DecisionDataset(
        (
            EncodedDecision(
                input_ids=vocabulary.encode(
                    (
                        "<HISTORY>",
                        "2",
                        "10",
                        "5",
                        "<CURRENT_HAND>",
                        "<PLAYER>",
                        "8",
                        "3",
                        "<DEALER>",
                        "6",
                        "<PLAY_QUERY>",
                    )
                ),
                target_id=hit,
                legal_token_ids=(hit, stand),
                kind=DecisionKind.PLAY,
                shoe_id=1,
                decision_index=2,
            ),
            EncodedDecision(
                input_ids=vocabulary.encode(
                    (
                        "<HISTORY>",
                        "A",
                        "4",
                        "9",
                        "<CURRENT_HAND>",
                        "<PLAYER>",
                        "10",
                        "7",
                        "<DEALER>",
                        "A",
                        "<PLAY_QUERY>",
                    )
                ),
                target_id=stand,
                legal_token_ids=(hit, stand),
                kind=DecisionKind.PLAY,
                shoe_id=2,
                decision_index=3,
            ),
        )
    )


def test_permutation_consistency_is_deterministic_and_counts_accuracy() -> None:
    result = evaluate_permutation_consistency(
        LastTokenModel(),
        _dataset(),
        permutation_count=3,
        batch_size=2,
        seed=19,
    )
    repeated = evaluate_permutation_consistency(
        LastTokenModel(),
        _dataset(),
        permutation_count=3,
        batch_size=2,
        seed=19,
    )

    assert result == repeated
    assert result.total_comparisons == 6
    assert 0 < result.changed_input_comparisons <= 6
    assert result.prediction_agreement == 1
    assert result.changed_input_prediction_agreement == 1
    assert result.original_accuracy == 0.5
    assert result.permuted_accuracy == 0.5
    assert permutation_consistency_data(result)["permutation_count"] == 3


@pytest.mark.parametrize(
    ("permutation_count", "batch_size", "seed", "message"),
    (
        (0, 2, 1, "permutation count"),
        (1, 0, 1, "batch size"),
        (1, 2, -1, "seed"),
    ),
)
def test_permutation_consistency_rejects_invalid_configuration(
    permutation_count: int,
    batch_size: int,
    seed: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_permutation_consistency(
            LastTokenModel(),
            _dataset(),
            permutation_count=permutation_count,
            batch_size=batch_size,
            seed=seed,
        )


def test_permutation_consistency_rejects_only_no_op_inputs() -> None:
    vocabulary = BLACKJACK_VOCABULARY
    minimum = vocabulary.id_for("<BET_MIN>")
    static_dataset = DecisionDataset(
        (
            EncodedDecision(
                input_ids=vocabulary.encode(
                    ("<HISTORY>", "2", "<BET_QUERY>")
                ),
                target_id=minimum,
                legal_token_ids=(minimum,),
                kind=DecisionKind.BET,
                shoe_id=1,
                decision_index=0,
            ),
        )
    )

    with pytest.raises(ValueError, match="did not change"):
        evaluate_permutation_consistency(
            LastTokenModel(),
            static_dataset,
            permutation_count=2,
        )
