from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from blackjack.dataset import (
    ActionValue,
    CheckpointMismatchError,
    DatasetConfiguration,
    DatasetOutputMismatchError,
    DatasetSplit,
    EvaluationMetadata,
    GenerationProgress,
    LabeledDecision,
    PlayToken,
    decision_example_from_json,
    decision_example_to_json,
    generate_dataset,
    insurance_token,
    play_token,
    read_shoe_checkpoints,
    run_resumable_generation,
)
from blackjack.engine import InsuranceAction, PlayerAction
from blackjack.oracle import Composition, RoundPlayerSituation


@dataclass(slots=True)
class _CountingOracle:
    calls: int = 0

    def label_bet(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        self.calls += 1
        return _label(
            composition,
            unseen_unavailable,
            ("<BET_MIN>", "<BET_LOW>", "<BET_MEDIUM>", "<BET_HIGH>"),
            "<BET_MIN>",
        )

    def label_insurance(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        self.calls += 1
        legal = tuple(insurance_token(action).value for action in InsuranceAction)
        return _label(
            composition,
            unseen_unavailable,
            legal,
            insurance_token(InsuranceAction.DECLINE).value,
        )

    def label_play(
        self,
        situation: RoundPlayerSituation,
        legal_actions: tuple[PlayerAction, ...],
    ) -> LabeledDecision:
        self.calls += 1
        legal = tuple(play_token(action).value for action in legal_actions)
        target = PlayToken.STAND.value if PlayToken.STAND.value in legal else legal[0]
        return _label(
            situation.composition,
            situation.unseen_unavailable,
            legal,
            target,
        )


def _label(
    composition: Composition,
    unseen_unavailable: int,
    legal: tuple[str, ...],
    target: str,
) -> LabeledDecision:
    return LabeledDecision(
        target_token=target,
        metadata=EvaluationMetadata(
            shoe_composition=composition,
            unseen_unavailable=unseen_unavailable,
            legal_target_tokens=legal,
            action_values=tuple(
                ActionValue(token=token, expected_profit=Fraction(index))
                for index, token in enumerate(legal)
            ),
        ),
    )


def _configuration(*, master_seed: int = 501) -> DatasetConfiguration:
    return DatasetConfiguration(
        master_seed=master_seed,
        split_seed=502,
        exploration_seed=503,
        shoe_count=3,
        exploration_probability=Fraction(1, 4),
    )


def test_decision_json_round_trip_preserves_exact_values() -> None:
    example = generate_dataset(
        _configuration(),
        oracle=_CountingOracle(),
    ).examples[0]
    assert decision_example_from_json(decision_example_to_json(example)) == example


def test_benchmark_limit_checkpoints_and_resume_reuses_every_prefix(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    first_oracle = _CountingOracle()
    first = run_resumable_generation(
        configuration,
        tmp_path,
        maximum_new_decisions=2,
        oracle=first_oracle,
    )
    assert first.stopped_at_limit
    assert first.new_decisions == first_oracle.calls == 2
    assert len(read_shoe_checkpoints(tmp_path, shoe_id=0)) == 2

    second_oracle = _CountingOracle()
    second = run_resumable_generation(
        configuration,
        tmp_path,
        maximum_new_decisions=2,
        oracle=second_oracle,
    )
    assert second.stopped_at_limit
    assert second.new_decisions == second_oracle.calls == 2
    assert second.cached_decisions == 2
    assert len(read_shoe_checkpoints(tmp_path, shoe_id=0)) == 4

    final_oracle = _CountingOracle()
    final = run_resumable_generation(
        configuration,
        tmp_path,
        oracle=final_oracle,
    )
    assert not final.stopped_at_limit
    assert final.assembled_all_splits
    assert set(final.completed_shoe_ids) == {0, 1, 2}
    assert all((tmp_path / f"{split.value}.jsonl").exists() for split in DatasetSplit)


def test_independent_shards_equal_single_process_generation(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    expected = generate_dataset(configuration, oracle=_CountingOracle())

    first = run_resumable_generation(
        configuration,
        tmp_path,
        shard_index=0,
        shard_count=2,
        oracle=_CountingOracle(),
    )
    assert first.selected_shoe_ids == (0, 2)
    assert not first.assembled_all_splits

    second = run_resumable_generation(
        configuration,
        tmp_path,
        shard_index=1,
        shard_count=2,
        oracle=_CountingOracle(),
    )
    assert second.selected_shoe_ids == (1,)
    assert second.assembled_all_splits

    for split in DatasetSplit:
        expected_content = "".join(
            decision_example_to_json(example)
            for example in expected.examples_for(split)
        )
        assert (tmp_path / f"{split.value}.jsonl").read_text(
            encoding="utf-8"
        ) == expected_content


def test_progress_reports_new_labels_and_completed_shoes(
    tmp_path: Path,
) -> None:
    updates: list[GenerationProgress] = []
    summary = run_resumable_generation(
        _configuration(),
        tmp_path,
        oracle=_CountingOracle(),
        progress=updates.append,
    )
    assert summary.assembled_all_splits
    assert any(
        update.decision_index is not None
        and not update.cached
        and update.label_seconds >= 0
        for update in updates
    )
    completions = tuple(update for update in updates if update.shoe_completed)
    assert len(completions) == 3
    assert completions[-1].estimated_remaining_seconds == 0


def test_existing_output_rejects_a_different_configuration(
    tmp_path: Path,
) -> None:
    run_resumable_generation(
        _configuration(),
        tmp_path,
        maximum_new_decisions=1,
        oracle=_CountingOracle(),
    )
    with pytest.raises(DatasetOutputMismatchError):
        run_resumable_generation(
            _configuration(master_seed=999),
            tmp_path,
            maximum_new_decisions=1,
            oracle=_CountingOracle(),
        )


def test_truncated_completed_shard_is_rejected_during_replay(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    run_resumable_generation(
        configuration,
        tmp_path,
        oracle=_CountingOracle(),
    )
    path = tmp_path / "shards" / "shoe-000000.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointMismatchError):
        run_resumable_generation(
            configuration,
            tmp_path,
            oracle=_CountingOracle(),
        )


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    ((-1, 2), (2, 2), (0, 0)),
)
def test_invalid_shard_selection_is_rejected(
    tmp_path: Path,
    shard_index: int,
    shard_count: int,
) -> None:
    with pytest.raises(ValueError):
        run_resumable_generation(
            _configuration(),
            tmp_path,
            shard_index=shard_index,
            shard_count=shard_count,
            oracle=_CountingOracle(),
        )
