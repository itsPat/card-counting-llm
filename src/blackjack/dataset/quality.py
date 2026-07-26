"""Reproducible integrity and coverage summaries for generated datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from blackjack.dataset.io import decision_example_from_json
from blackjack.dataset.records import (
    DatasetSplit,
    DecisionExample,
    DecisionKind,
)


class DatasetQualityError(RuntimeError):
    """Raised when assembled dataset rows violate a structural invariant."""


@dataclass(frozen=True, slots=True)
class NumericSummary:
    count: int
    minimum: float
    mean: float
    median: float
    percentile_95: float
    maximum: float


@dataclass(frozen=True, slots=True)
class KindCoverage:
    kind: DecisionKind
    rows: int
    exploratory_rows: int
    context_lengths: NumericSummary

    @property
    def exploration_fraction(self) -> float:
        return self.exploratory_rows / self.rows


@dataclass(frozen=True, slots=True)
class TargetCoverage:
    token: str
    rows: int
    shoes: int
    train_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True, slots=True)
class DatasetQualityReport:
    dataset_id: str
    schema_version: int
    rows: int
    shoes: int
    train_rows: int
    validation_rows: int
    test_rows: int
    kinds: tuple[KindCoverage, ...]
    targets: tuple[TargetCoverage, ...]
    bet_standard_errors: NumericSummary | None
    play_action_standard_errors: NumericSummary | None
    play_action_margins: NumericSummary | None
    play_margin_below_point_one_percentage_point: int
    play_margin_below_half_percentage_point: int
    play_margin_below_one_percentage_point: int


def analyze_dataset(output_directory: Path) -> DatasetQualityReport:
    """Validate assembled split files and summarize their learning coverage."""

    examples = tuple(_read_examples(output_directory))
    if not examples:
        raise DatasetQualityError("assembled dataset contains no rows")
    dataset_ids = {example.dataset_id for example in examples}
    schema_versions = {example.schema_version for example in examples}
    if len(dataset_ids) != 1:
        raise DatasetQualityError("rows contain multiple dataset identifiers")
    if len(schema_versions) != 1:
        raise DatasetQualityError("rows contain multiple schema versions")
    _validate_unique_contiguous_decisions(examples)
    _validate_shoe_split_isolation(examples)

    split_counts = Counter(example.split for example in examples)
    kind_counts = Counter(example.kind for example in examples)
    exploratory_counts = Counter(
        example.kind
        for example in examples
        if example.behavior_token != example.target_token
    )
    context_lengths: defaultdict[DecisionKind, list[float]] = defaultdict(list)
    target_counts = Counter(example.target_token for example in examples)
    target_shoes: defaultdict[str, set[int]] = defaultdict(set)
    target_split_counts: Counter[tuple[str, DatasetSplit]] = Counter()
    method_counts: Counter[tuple[DecisionKind, str]] = Counter()
    bet_standard_errors: list[float] = []
    play_standard_errors: list[float] = []
    play_margins: list[float] = []

    for example in examples:
        context_lengths[example.kind].append(float(len(example.input_tokens)))
        target_shoes[example.target_token].add(example.shoe_id)
        target_split_counts[(example.target_token, example.split)] += 1
        method_counts[
            (example.kind, example.metadata.evaluation_method.value)
        ] += 1
        if example.kind is DecisionKind.BET:
            monte_carlo = example.metadata.monte_carlo
            if monte_carlo is None:
                raise DatasetQualityError("bet row lacks Monte Carlo metadata")
            bet_standard_errors.append(
                monte_carlo.expected_profit_standard_error
            )
        elif example.kind is DecisionKind.PLAY:
            expected_profits: list[float] = []
            for value in example.metadata.action_values:
                if value.expected_profit is None or value.monte_carlo is None:
                    raise DatasetQualityError(
                        "play action lacks value or Monte Carlo metadata"
                    )
                expected_profits.append(float(value.expected_profit))
                play_standard_errors.append(
                    value.monte_carlo.expected_profit_standard_error
                )
            if len(expected_profits) < 2:
                raise DatasetQualityError(
                    "play decision needs at least two legal actions"
                )
            expected_profits.sort(reverse=True)
            play_margins.append(expected_profits[0] - expected_profits[1])

    required_methods = {
        DecisionKind.BET: "seeded_monte_carlo_fixed_h17_basic_strategy",
        DecisionKind.INSURANCE: "rational_exact_cdp",
        DecisionKind.PLAY: "seeded_monte_carlo_fixed_h17_continuation",
    }
    for kind, expected_method in required_methods.items():
        observed = {
            method
            for observed_kind, method in method_counts
            if observed_kind is kind
        }
        if observed != {expected_method}:
            raise DatasetQualityError(
                f"{kind.value} rows use unexpected methods: {sorted(observed)!r}"
            )

    kinds = tuple(
        KindCoverage(
            kind=kind,
            rows=kind_counts[kind],
            exploratory_rows=exploratory_counts[kind],
            context_lengths=_numeric_summary(context_lengths[kind]),
        )
        for kind in DecisionKind
    )
    targets = tuple(
        TargetCoverage(
            token=token,
            rows=rows,
            shoes=len(target_shoes[token]),
            train_rows=target_split_counts[(token, DatasetSplit.TRAIN)],
            validation_rows=target_split_counts[
                (token, DatasetSplit.VALIDATION)
            ],
            test_rows=target_split_counts[(token, DatasetSplit.TEST)],
        )
        for token, rows in target_counts.most_common()
    )
    return DatasetQualityReport(
        dataset_id=next(iter(dataset_ids)),
        schema_version=next(iter(schema_versions)),
        rows=len(examples),
        shoes=len({example.shoe_id for example in examples}),
        train_rows=split_counts[DatasetSplit.TRAIN],
        validation_rows=split_counts[DatasetSplit.VALIDATION],
        test_rows=split_counts[DatasetSplit.TEST],
        kinds=kinds,
        targets=targets,
        bet_standard_errors=_optional_numeric_summary(bet_standard_errors),
        play_action_standard_errors=_optional_numeric_summary(
            play_standard_errors
        ),
        play_action_margins=_optional_numeric_summary(play_margins),
        play_margin_below_point_one_percentage_point=sum(
            margin < 0.001 for margin in play_margins
        ),
        play_margin_below_half_percentage_point=sum(
            margin < 0.005 for margin in play_margins
        ),
        play_margin_below_one_percentage_point=sum(
            margin < 0.01 for margin in play_margins
        ),
    )


def quality_report_json(report: DatasetQualityReport) -> str:
    """Serialize a report without exposing internal mutable structures."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def _read_examples(output_directory: Path) -> Iterable[DecisionExample]:
    for split in DatasetSplit:
        path = output_directory / f"{split.value}.jsonl"
        if not path.exists():
            raise DatasetQualityError(f"assembled split file is missing: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                example = decision_example_from_json(line)
                if example.split is not split:
                    raise DatasetQualityError(
                        f"{path}:{line_number} contains a {example.split.value} row"
                    )
                yield example


def _validate_unique_contiguous_decisions(
    examples: tuple[DecisionExample, ...],
) -> None:
    decisions_by_shoe: defaultdict[int, list[int]] = defaultdict(list)
    for example in examples:
        decisions_by_shoe[example.shoe_id].append(example.decision_index)
    for shoe_id, indices in decisions_by_shoe.items():
        if len(indices) != len(set(indices)):
            raise DatasetQualityError(
                f"shoe {shoe_id} contains duplicate decision indices"
            )
        if sorted(indices) != list(range(len(indices))):
            raise DatasetQualityError(
                f"shoe {shoe_id} decision indices are not contiguous"
            )


def _validate_shoe_split_isolation(
    examples: tuple[DecisionExample, ...],
) -> None:
    splits_by_shoe: defaultdict[int, set[DatasetSplit]] = defaultdict(set)
    for example in examples:
        splits_by_shoe[example.shoe_id].add(example.split)
    leaking = {
        shoe_id: splits
        for shoe_id, splits in splits_by_shoe.items()
        if len(splits) != 1
    }
    if leaking:
        raise DatasetQualityError(
            f"shoes cross dataset split boundaries: {leaking!r}"
        )


def _optional_numeric_summary(
    values: list[float],
) -> NumericSummary | None:
    return _numeric_summary(values) if values else None


def _numeric_summary(values: list[float]) -> NumericSummary:
    if not values:
        raise ValueError("numeric summary needs at least one value")
    ordered = sorted(values)
    return NumericSummary(
        count=len(ordered),
        minimum=ordered[0],
        mean=fmean(ordered),
        median=ordered[len(ordered) // 2],
        percentile_95=ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        maximum=ordered[-1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize an assembled blackjack dataset."
    )
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(quality_report_json(analyze_dataset(arguments.output)), end="")


if __name__ == "__main__":
    main()
