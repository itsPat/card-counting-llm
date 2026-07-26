from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from blackjack.dataset import (
    ActionValue,
    DatasetSplit,
    DecisionExample,
    DecisionKind,
    EvaluationMetadata,
    EvaluationMethod,
    MonteCarloMetadata,
    decision_example_to_json,
)
from blackjack.dataset.quality import (
    DatasetQualityError,
    analyze_dataset,
    quality_report_json,
)
from blackjack.oracle import Composition


def _simulation() -> MonteCarloMetadata:
    return MonteCarloMetadata(
        seed=7,
        rollouts=1_000,
        expected_profit_standard_error=0.01,
        expected_profit_confidence_interval_95=(-0.02, 0.02),
    )


def _example(
    kind: DecisionKind,
    split: DatasetSplit,
    shoe_id: int,
    target: str,
    behavior: str,
) -> DecisionExample:
    if kind is DecisionKind.BET:
        legal = ("<BET_MIN>", "<BET_LOW>")
        values = (
            ActionValue(token=legal[0], expected_log_growth=0.0),
            ActionValue(token=legal[1], expected_log_growth=-0.1),
        )
        method = EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_H17
        simulation = _simulation()
    elif kind is DecisionKind.PLAY:
        legal = ("<HIT>", "<STAND>")
        values = (
            ActionValue(
                token=legal[0],
                expected_profit=Fraction(1, 10),
                monte_carlo=_simulation(),
            ),
            ActionValue(
                token=legal[1],
                expected_profit=Fraction(0),
                monte_carlo=_simulation(),
            ),
        )
        method = EvaluationMethod.SEEDED_MONTE_CARLO_FIXED_CONTINUATION_H17
        simulation = None
    else:
        legal = ("<INSURANCE>", "<NO_INSURANCE>")
        values = (
            ActionValue(token=legal[0], expected_profit=Fraction(-1, 10)),
            ActionValue(token=legal[1], expected_profit=Fraction(0)),
        )
        method = EvaluationMethod.RATIONAL_EXACT_CDP
        simulation = None
    return DecisionExample(
        schema_version=4,
        dataset_id="quality-fixture",
        shoe_id=shoe_id,
        shoe_seed=100 + shoe_id,
        split=split,
        round_index=0,
        decision_index=0,
        kind=kind,
        input_tokens=("<HISTORY>", f"<{kind.value.upper()}_QUERY>"),
        target_token=target,
        behavior_token=behavior,
        metadata=EvaluationMetadata(
            shoe_composition=Composition.full_shoe(),
            unseen_unavailable=1,
            legal_target_tokens=legal,
            action_values=values,
            evaluation_method=method,
            monte_carlo=simulation,
        ),
    )


def _write_fixture(output: Path) -> None:
    examples = {
        DatasetSplit.TRAIN: _example(
            DecisionKind.BET,
            DatasetSplit.TRAIN,
            0,
            "<BET_MIN>",
            "<BET_MIN>",
        ),
        DatasetSplit.VALIDATION: _example(
            DecisionKind.INSURANCE,
            DatasetSplit.VALIDATION,
            1,
            "<NO_INSURANCE>",
            "<INSURANCE>",
        ),
        DatasetSplit.TEST: _example(
            DecisionKind.PLAY,
            DatasetSplit.TEST,
            2,
            "<HIT>",
            "<STAND>",
        ),
    }
    for split, example in examples.items():
        (output / f"{split.value}.jsonl").write_text(
            decision_example_to_json(example),
            encoding="utf-8",
        )


def test_quality_report_validates_and_summarizes_assembled_rows(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    report = analyze_dataset(tmp_path)
    assert report.dataset_id == "quality-fixture"
    assert report.rows == report.shoes == 3
    assert report.train_rows == report.validation_rows == report.test_rows == 1
    assert report.play_action_margins is not None
    assert report.play_action_margins.mean == pytest.approx(0.1)
    assert report.play_margin_below_one_percentage_point == 0
    assert '"dataset_id": "quality-fixture"' in quality_report_json(report)


def test_quality_report_rejects_duplicate_decisions(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    train = tmp_path / "train.jsonl"
    content = train.read_text(encoding="utf-8")
    train.write_text(content + content, encoding="utf-8")
    with pytest.raises(DatasetQualityError, match="duplicate"):
        analyze_dataset(tmp_path)
