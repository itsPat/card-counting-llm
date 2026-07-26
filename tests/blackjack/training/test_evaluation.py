from __future__ import annotations

from fractions import Fraction
from math import log1p
from pathlib import Path

import pytest
import torch

from blackjack.dataset import (
    ActionValue,
    DatasetSplit,
    DecisionExample,
    DecisionKind,
    EvaluationMetadata,
    ReturnDistributionRecord,
    ReturnOutcomeRecord,
    decision_example_to_json,
)
from blackjack.oracle import Composition
from blackjack.training import (
    BLACKJACK_VOCABULARY,
    DecisionCollator,
    DecisionDataset,
)
from blackjack.training.evaluation import EvaluationReferenceIndex
from blackjack.training.metrics import DecisionMetricAccumulator


def _growth(bankroll_fraction: float) -> float:
    return (
        0.5 * log1p(bankroll_fraction)
        + 0.5 * log1p(-bankroll_fraction)
    )


def _bet_example() -> DecisionExample:
    legal = ("<BET_MIN>", "<BET_HIGH>")
    return DecisionExample(
        schema_version=4,
        dataset_id="evaluation-fixture",
        shoe_id=5,
        shoe_seed=6,
        split=DatasetSplit.VALIDATION,
        round_index=0,
        decision_index=0,
        kind=DecisionKind.BET,
        input_tokens=("<HISTORY>", "<BET_QUERY>"),
        target_token="<BET_HIGH>",
        behavior_token="<BET_MIN>",
        metadata=EvaluationMetadata(
            shoe_composition=Composition.full_shoe(),
            unseen_unavailable=1,
            legal_target_tokens=legal,
            action_values=(
                ActionValue(
                    token="<BET_MIN>",
                    expected_log_growth=_growth(0.001),
                ),
                ActionValue(
                    token="<BET_HIGH>",
                    expected_log_growth=_growth(0.013),
                ),
            ),
            round_return_distribution=ReturnDistributionRecord(
                outcomes=(
                    ReturnOutcomeRecord(Fraction(1), Fraction(1, 2)),
                    ReturnOutcomeRecord(Fraction(-1), Fraction(1, 2)),
                )
            ),
            continuous_half_kelly=0.012,
            selected_bet_fraction=0.013,
        ),
    )


def test_references_measure_regret_without_entering_training_items(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.jsonl"
    path.write_text(
        decision_example_to_json(_bet_example()),
        encoding="utf-8",
    )
    dataset = DecisionDataset.from_jsonl(
        path,
        expected_split=DatasetSplit.VALIDATION,
    )
    references = EvaluationReferenceIndex.from_jsonl(
        path,
        expected_split=DatasetSplit.VALIDATION,
        vocabulary=dataset.vocabulary,
    )
    assert not hasattr(dataset[0], "action_values")

    batch = DecisionCollator()(
        tuple(dataset[index] for index in range(len(dataset)))
    )
    logits = torch.zeros(
        (1, 2, len(BLACKJACK_VOCABULARY)),
        dtype=torch.float32,
    )
    logits[
        0,
        1,
        BLACKJACK_VOCABULARY.id_for("<BET_MIN>"),
    ] = 1
    accumulator = DecisionMetricAccumulator(
        BLACKJACK_VOCABULARY,
        references,
    )
    accumulator.update(torch.tensor(1.0), logits, batch)
    metrics = accumulator.finish()
    assert metrics.accuracy == 0
    assert metrics.regret_by_kind == ()
    assert metrics.bet_policy is not None
    assert metrics.bet_policy.total == 1
    assert metrics.bet_policy.mean_absolute_fraction_error == pytest.approx(
        0.011
    )
    assert (
        metrics.bet_policy.mean_absolute_log_growth_change
        == pytest.approx(abs(_growth(0.001) - _growth(0.012)))
    )


def test_reference_rejects_a_prediction_outside_the_legal_actions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.jsonl"
    path.write_text(
        decision_example_to_json(_bet_example()),
        encoding="utf-8",
    )
    references = EvaluationReferenceIndex.from_jsonl(
        path,
        expected_split=DatasetSplit.VALIDATION,
        vocabulary=BLACKJACK_VOCABULARY,
    )
    reference = references.get(5, 0)
    assert reference.bet_policy is not None
    with pytest.raises(ValueError, match="not legal"):
        reference.bet_policy.errors(
            BLACKJACK_VOCABULARY.id_for("<HIT>")
        )
