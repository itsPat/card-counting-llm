"""Transparent JSON/JSONL serialization for generated decision datasets."""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import cast

from blackjack.dataset.labeling import LabeledDecision
from blackjack.dataset.records import (
    ActionValue,
    DatasetBundle,
    DatasetConfiguration,
    DatasetManifest,
    DatasetSplit,
    DecisionExample,
    DecisionKind,
    EvaluationMetadata,
    EvaluationMethod,
    MonteCarloMetadata,
    ReturnDistributionRecord,
    ReturnOutcomeRecord,
)
from blackjack.engine import CasinoRules
from blackjack.oracle import Composition

type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)


def write_dataset(bundle: DatasetBundle, output_directory: Path) -> None:
    """Write one manifest plus one JSONL file per whole-shoe split."""

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest_data(bundle.manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for split in DatasetSplit:
        lines = (
            json.dumps(_example_data(example), sort_keys=True, separators=(",", ":"))
            for example in bundle.examples_for(split)
        )
        content = "\n".join(lines)
        if content:
            content += "\n"
        (output_directory / f"{split.value}.jsonl").write_text(
            content,
            encoding="utf-8",
        )


class DatasetOutputMismatchError(RuntimeError):
    """Raised when an output directory belongs to another configuration."""


def initialize_output(
    manifest: DatasetManifest,
    output_directory: Path,
) -> None:
    """Create or validate the immutable manifest before long-running work."""

    expected = _json_text(_manifest_data(manifest), pretty=True)
    path = output_directory / "manifest.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise DatasetOutputMismatchError(
                "output manifest does not match this dataset configuration"
            )
        return
    _atomic_write(path, expected)


def read_shoe_checkpoints(
    output_directory: Path,
    shoe_id: int,
) -> tuple[DecisionExample, ...]:
    directory = _checkpoint_directory(output_directory, shoe_id)
    if not directory.exists():
        return ()
    paths = sorted(directory.glob("decision-*.json"))
    examples = tuple(
        decision_example_from_json(path.read_text(encoding="utf-8")) for path in paths
    )
    if tuple(example.decision_index for example in examples) != tuple(
        range(len(examples))
    ):
        raise DatasetOutputMismatchError(
            f"shoe {shoe_id} checkpoints are not a contiguous decision prefix"
        )
    return examples


def write_decision_checkpoint(
    output_directory: Path,
    example: DecisionExample,
) -> None:
    path = (
        _checkpoint_directory(output_directory, example.shoe_id)
        / f"decision-{example.decision_index:06d}.json"
    )
    content = decision_example_to_json(example, pretty=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise DatasetOutputMismatchError(f"checkpoint already differs at {path}")
        return
    _atomic_write(path, content)


def shoe_shard_path(output_directory: Path, shoe_id: int) -> Path:
    return output_directory / "shards" / f"shoe-{shoe_id:06d}.jsonl"


def write_shoe_shard(
    output_directory: Path,
    shoe_id: int,
    examples: tuple[DecisionExample, ...],
) -> None:
    if not examples:
        raise ValueError("a completed shoe needs at least one decision")
    if any(example.shoe_id != shoe_id for example in examples):
        raise ValueError("shoe shard contains a row from another shoe")
    content = "".join(decision_example_to_json(example) for example in examples)
    path = shoe_shard_path(output_directory, shoe_id)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    _atomic_write(path, content)


def read_shoe_shard(
    output_directory: Path,
    shoe_id: int,
) -> tuple[DecisionExample, ...]:
    path = shoe_shard_path(output_directory, shoe_id)
    if not path.exists():
        return ()
    return tuple(
        decision_example_from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def assemble_completed_splits(
    manifest: DatasetManifest,
    output_directory: Path,
) -> bool:
    """Build final split files once every shoe shard exists."""

    if any(
        not shoe_shard_path(output_directory, shoe.shoe_id).exists()
        for shoe in manifest.shoes
    ):
        return False
    for split in DatasetSplit:
        content = "".join(
            shoe_shard_path(output_directory, shoe.shoe_id).read_text(encoding="utf-8")
            for shoe in manifest.shoes
            if shoe.split is split
        )
        _atomic_write(output_directory / f"{split.value}.jsonl", content)
    return True


def decision_example_to_json(
    example: DecisionExample,
    *,
    pretty: bool = False,
) -> str:
    return _json_text(_example_data(example), pretty=pretty)


def decision_example_from_json(content: str) -> DecisionExample:
    raw: object = json.loads(content)
    data = _mapping(raw)
    metadata = _metadata_from_data(_required(data, "metadata"))
    return DecisionExample(
        schema_version=_integer(_required(data, "schema_version")),
        dataset_id=_string(_required(data, "dataset_id")),
        shoe_id=_integer(_required(data, "shoe_id")),
        shoe_seed=_integer(_required(data, "shoe_seed")),
        split=DatasetSplit(_string(_required(data, "split"))),
        round_index=_integer(_required(data, "round_index")),
        decision_index=_integer(_required(data, "decision_index")),
        kind=DecisionKind(_string(_required(data, "kind"))),
        input_tokens=tuple(
            _string(item) for item in _list(_required(data, "input_tokens"))
        ),
        target_token=_string(_required(data, "target_token")),
        behavior_token=_string(_required(data, "behavior_token")),
        metadata=metadata,
    )


def labeled_decision_to_json(label: LabeledDecision) -> str:
    return _json_text(
        {
            "target_token": label.target_token,
            "metadata": _metadata_data(label.metadata),
        },
        pretty=False,
    )


def labeled_decision_from_json(content: str) -> LabeledDecision:
    raw: object = json.loads(content)
    data = _mapping(raw)
    return LabeledDecision(
        target_token=_string(_required(data, "target_token")),
        metadata=_metadata_from_data(_required(data, "metadata")),
    )


def _checkpoint_directory(output_directory: Path, shoe_id: int) -> Path:
    return output_directory / ".checkpoints" / f"shoe-{shoe_id:06d}"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _json_text(data: JsonValue, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True) + "\n"
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def _fraction_data(value: Fraction) -> dict[str, JsonValue]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _rules_data(rules: CasinoRules) -> dict[str, JsonValue]:
    return {
        "decks": rules.decks,
        "blackjack_profit": _fraction_data(rules.blackjack_profit),
        "ordinary_win_profit": _fraction_data(rules.ordinary_win_profit),
        "dealer_hits_soft_17": rules.dealer_hits_soft_17,
        "dealer_peeks": rules.dealer_peeks,
        "double_after_split": rules.double_after_split,
        "maximum_player_hands": rules.maximum_player_hands,
        "resplit_aces": rules.resplit_aces,
        "split_aces_one_card_only": rules.split_aces_one_card_only,
        "late_surrender": rules.late_surrender,
        "insurance_fraction": _fraction_data(rules.insurance_fraction),
        "insurance_profit": _fraction_data(rules.insurance_profit),
        "minimum_penetration": _fraction_data(rules.minimum_penetration),
        "maximum_penetration": _fraction_data(rules.maximum_penetration),
        "burn_cards": rules.burn_cards,
    }


def _configuration_data(
    configuration: DatasetConfiguration,
) -> dict[str, JsonValue]:
    return {
        "master_seed": configuration.master_seed,
        "split_seed": configuration.split_seed,
        "exploration_seed": configuration.exploration_seed,
        "shoe_count": configuration.shoe_count,
        "train_fraction": _fraction_data(configuration.train_fraction),
        "validation_fraction": _fraction_data(configuration.validation_fraction),
        "test_fraction": _fraction_data(configuration.test_fraction),
        "exploration_probability": _fraction_data(
            configuration.exploration_probability
        ),
        "bet_rollout_seed": configuration.bet_rollout_seed,
        "bet_rollouts": configuration.bet_rollouts,
        "bet_evaluation_method": configuration.bet_evaluation_method.value,
        "play_rollout_seed": configuration.play_rollout_seed,
        "play_rollouts": configuration.play_rollouts,
        "play_evaluation_method": configuration.play_evaluation_method.value,
        "bet_vocabulary": [
            {
                "token": token.token.value,
                "bankroll_fraction": token.bankroll_fraction,
            }
            for token in configuration.bet_vocabulary.tokens
        ],
        "rules": _rules_data(configuration.rules),
    }


def _manifest_data(manifest: DatasetManifest) -> dict[str, JsonValue]:
    return {
        "schema_version": manifest.schema_version,
        "dataset_id": manifest.dataset_id,
        "configuration": _configuration_data(manifest.configuration),
        "shoes": [
            {
                "shoe_id": shoe.shoe_id,
                "seed": shoe.seed,
                "split": shoe.split.value,
                "cards": [rank.value for rank in shoe.cards],
                "cut_card_position": shoe.cut_card_position,
            }
            for shoe in manifest.shoes
        ],
    }


def _distribution_data(
    distribution: ReturnDistributionRecord,
) -> dict[str, JsonValue]:
    return {
        "outcomes": [
            {
                "profit": _fraction_data(outcome.profit),
                "probability": _fraction_data(outcome.probability),
            }
            for outcome in distribution.outcomes
        ]
    }


def _action_value_data(value: ActionValue) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"token": value.token}
    if value.expected_profit is not None:
        result["expected_profit"] = _fraction_data(value.expected_profit)
    if value.expected_log_growth is not None:
        result["expected_log_growth"] = value.expected_log_growth
    if value.return_distribution is not None:
        result["return_distribution"] = _distribution_data(value.return_distribution)
    if value.monte_carlo is not None:
        result["monte_carlo"] = _monte_carlo_data(value.monte_carlo)
    return result


def _metadata_data(metadata: EvaluationMetadata) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "shoe_composition": list(metadata.shoe_composition.counts),
        "unseen_unavailable": metadata.unseen_unavailable,
        "evaluation_method": metadata.evaluation_method.value,
        "legal_target_tokens": list(metadata.legal_target_tokens),
        "action_values": [
            _action_value_data(value) for value in metadata.action_values
        ],
    }
    if metadata.round_return_distribution is not None:
        result["round_return_distribution"] = _distribution_data(
            metadata.round_return_distribution
        )
    if metadata.continuous_half_kelly is not None:
        result["continuous_half_kelly"] = metadata.continuous_half_kelly
    if metadata.selected_bet_fraction is not None:
        result["selected_bet_fraction"] = metadata.selected_bet_fraction
    if metadata.monte_carlo is not None:
        result["monte_carlo"] = _monte_carlo_data(metadata.monte_carlo)
    return result


def _monte_carlo_data(metadata: MonteCarloMetadata) -> dict[str, JsonValue]:
    return {
        "seed": metadata.seed,
        "rollouts": metadata.rollouts,
        "expected_profit_standard_error": (
            metadata.expected_profit_standard_error
        ),
        "expected_profit_confidence_interval_95": list(
            metadata.expected_profit_confidence_interval_95
        ),
    }


def _example_data(example: DecisionExample) -> dict[str, JsonValue]:
    return {
        "schema_version": example.schema_version,
        "dataset_id": example.dataset_id,
        "shoe_id": example.shoe_id,
        "shoe_seed": example.shoe_seed,
        "split": example.split.value,
        "round_index": example.round_index,
        "decision_index": example.decision_index,
        "kind": example.kind.value,
        "input_tokens": list(example.input_tokens),
        "target_token": example.target_token,
        "behavior_token": example.behavior_token,
        "metadata": _metadata_data(example.metadata),
    }


def _metadata_from_data(value: object) -> EvaluationMetadata:
    data = _mapping(value)
    action_values = tuple(
        _action_value_from_data(item)
        for item in _list(_required(data, "action_values"))
    )
    round_distribution = (
        _distribution_from_data(data["round_return_distribution"])
        if "round_return_distribution" in data
        else None
    )
    return EvaluationMetadata(
        shoe_composition=Composition(
            tuple(_integer(item) for item in _list(_required(data, "shoe_composition")))
        ),
        unseen_unavailable=_integer(_required(data, "unseen_unavailable")),
        evaluation_method=EvaluationMethod(
            _string(_required(data, "evaluation_method"))
        ),
        legal_target_tokens=tuple(
            _string(item) for item in _list(_required(data, "legal_target_tokens"))
        ),
        action_values=action_values,
        round_return_distribution=round_distribution,
        continuous_half_kelly=(
            _number(data["continuous_half_kelly"])
            if "continuous_half_kelly" in data
            else None
        ),
        selected_bet_fraction=(
            _number(data["selected_bet_fraction"])
            if "selected_bet_fraction" in data
            else None
        ),
        monte_carlo=(
            _monte_carlo_from_data(data["monte_carlo"])
            if "monte_carlo" in data
            else None
        ),
    )


def _monte_carlo_from_data(value: object) -> MonteCarloMetadata:
    data = _mapping(value)
    confidence = _list(
        _required(data, "expected_profit_confidence_interval_95")
    )
    if len(confidence) != 2:
        raise ValueError("Monte Carlo confidence interval needs two endpoints")
    return MonteCarloMetadata(
        seed=_integer(_required(data, "seed")),
        rollouts=_integer(_required(data, "rollouts")),
        expected_profit_standard_error=_number(
            _required(data, "expected_profit_standard_error")
        ),
        expected_profit_confidence_interval_95=(
            _number(confidence[0]),
            _number(confidence[1]),
        ),
    )


def _action_value_from_data(value: object) -> ActionValue:
    data = _mapping(value)
    return ActionValue(
        token=_string(_required(data, "token")),
        expected_profit=(
            _fraction_from_data(data["expected_profit"])
            if "expected_profit" in data
            else None
        ),
        expected_log_growth=(
            _number(data["expected_log_growth"])
            if "expected_log_growth" in data
            else None
        ),
        return_distribution=(
            _distribution_from_data(data["return_distribution"])
            if "return_distribution" in data
            else None
        ),
        monte_carlo=(
            _monte_carlo_from_data(data["monte_carlo"])
            if "monte_carlo" in data
            else None
        ),
    )


def _distribution_from_data(value: object) -> ReturnDistributionRecord:
    data = _mapping(value)
    return ReturnDistributionRecord(
        tuple(
            ReturnOutcomeRecord(
                profit=_fraction_from_data(_required(_mapping(item), "profit")),
                probability=_fraction_from_data(
                    _required(_mapping(item), "probability")
                ),
            )
            for item in _list(_required(data, "outcomes"))
        )
    )


def _fraction_from_data(value: object) -> Fraction:
    data = _mapping(value)
    return Fraction(
        _integer(_required(data, "numerator")),
        _integer(_required(data, "denominator")),
    )


def _required(data: dict[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"missing JSON field: {key}")
    return data[key]


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object with string keys")
    untyped = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in untyped):
        raise ValueError("expected a JSON object with string keys")
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected a JSON array")
    return cast(list[object], value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a JSON string")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected a JSON integer")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a JSON number")
    return float(value)
