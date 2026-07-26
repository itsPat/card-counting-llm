from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from blackjack.dataset import (
    ActionValue,
    DatasetConfiguration,
    DatasetSplit,
    DecisionKind,
    EvaluationMetadata,
    LabeledDecision,
    PlayToken,
    StructureToken,
    generate_dataset,
    insurance_token,
    play_token,
    write_dataset,
)
from blackjack.engine import Card, InsuranceAction, PlayerAction, Shoe, ShoeReplay
from blackjack.oracle import CARD_VALUES, Composition, RoundPlayerSituation


class _FastOracle:
    """Deterministic labels for testing pipeline mechanics, not strategy."""

    def label_bet(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
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


def _configuration(
    *,
    exploration_probability: Fraction = Fraction(0),
) -> DatasetConfiguration:
    return DatasetConfiguration(
        master_seed=101,
        split_seed=202,
        exploration_seed=303,
        shoe_count=3,
        exploration_probability=exploration_probability,
    )


def test_complete_shoe_generation_is_reproducible() -> None:
    configuration = _configuration()
    first = generate_dataset(configuration, oracle=_FastOracle())
    second = generate_dataset(configuration, oracle=_FastOracle())
    assert first == second
    assert len(first.manifest.shoes) == 3
    assert all(len(shoe.cards) == 312 for shoe in first.manifest.shoes)
    assert all(
        any(
            example.shoe_id == shoe.shoe_id and example.kind is DecisionKind.BET
            for example in first.examples
        )
        for shoe in first.manifest.shoes
    )
    for recorded in first.manifest.shoes:
        replay = ShoeReplay(
            cards=tuple(Card(rank) for rank in recorded.cards),
            cut_card_position=recorded.cut_card_position,
        )
        assert replay == Shoe.shuffled(recorded.seed).replay


def test_whole_shoes_never_cross_dataset_splits() -> None:
    bundle = generate_dataset(_configuration(), oracle=_FastOracle())
    split_by_shoe: dict[int, set[DatasetSplit]] = {}
    for example in bundle.examples:
        split_by_shoe.setdefault(example.shoe_id, set()).add(example.split)
    assert all(len(splits) == 1 for splits in split_by_shoe.values())
    assert {shoe.split for shoe in bundle.manifest.shoes} == set(DatasetSplit)


def test_exploration_changes_behavior_without_changing_targets() -> None:
    bundle = generate_dataset(
        _configuration(exploration_probability=Fraction(1)),
        oracle=_FastOracle(),
    )
    play_examples = tuple(
        example for example in bundle.examples if example.kind is DecisionKind.PLAY
    )
    assert play_examples
    assert all(
        example.behavior_token != example.target_token
        for example in play_examples
        if len(example.metadata.legal_target_tokens) > 1
    )
    assert any(
        example.round_index > 0
        and example.input_tokens[0] == StructureToken.HISTORY
        and example.input_tokens[1] != StructureToken.CURRENT_HAND
        for example in play_examples
    )


def test_public_history_tracks_unrevealed_cards_as_unknown_unavailable() -> None:
    bundle = generate_dataset(
        _configuration(exploration_probability=Fraction(1)),
        oracle=_FastOracle(),
    )
    later_bets = tuple(
        example
        for example in bundle.examples
        if example.kind is DecisionKind.BET and example.round_index > 0
    )
    assert later_bets
    assert any(example.metadata.unseen_unavailable > 1 for example in later_bets)
    for example in later_bets:
        visible_cards = len(example.input_tokens) - 2
        assert example.metadata.shoe_composition.total == 312 - visible_cards


def test_each_visible_card_appears_exactly_once_in_model_tokens() -> None:
    bundle = generate_dataset(_configuration(), oracle=_FastOracle())
    card_tokens = {value.value for value in CARD_VALUES}
    for example in bundle.examples:
        serialized_cards = sum(token in card_tokens for token in example.input_tokens)
        assert example.metadata.shoe_composition.total == 312 - serialized_cards


def test_writer_emits_manifest_and_one_jsonl_file_per_split(
    tmp_path: Path,
) -> None:
    bundle = generate_dataset(_configuration(), oracle=_FastOracle())
    write_dataset(bundle, tmp_path)
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert bundle.manifest.dataset_id in manifest
    assert '"cards": [' in manifest
    for split in DatasetSplit:
        path = tmp_path / f"{split.value}.jsonl"
        rows = path.read_text(encoding="utf-8").splitlines()
        assert len(rows) == len(bundle.examples_for(split))
        assert all('"metadata":' in row for row in rows)


def test_too_few_shoes_are_rejected() -> None:
    with pytest.raises(ValueError):
        DatasetConfiguration(shoe_count=2)


def test_invalid_exploration_probability_is_rejected() -> None:
    with pytest.raises(ValueError):
        DatasetConfiguration(exploration_probability=Fraction(-1, 10))


def test_invalid_bet_simulation_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        DatasetConfiguration(bet_rollout_seed=-1)
    with pytest.raises(ValueError):
        DatasetConfiguration(bet_rollout_seed=2**64)
    with pytest.raises(ValueError):
        DatasetConfiguration(bet_rollouts=0)


def test_invalid_play_simulation_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        DatasetConfiguration(play_rollout_seed=-1)
    with pytest.raises(ValueError):
        DatasetConfiguration(play_rollout_seed=2**64)
    with pytest.raises(ValueError):
        DatasetConfiguration(play_rollouts=0)


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        DatasetConfiguration(
            train_fraction=Fraction(1, 2),
            validation_fraction=Fraction(1, 2),
            test_fraction=Fraction(1, 2),
        )
