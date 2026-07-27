"""Pool resumable bankroll-evaluation reports without losing shard variance."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from typing import cast

from blackjack.analysis.bankroll_svg import (
    BankrollChartPoint,
    write_bankroll_series_chart,
)
from blackjack.analysis.bet_tokens import SELECTED_BET_VOCABULARY

_POLICY_NAMES = ("transformer", "hi-lo")
_MINIMUM_BET_FRACTION = min(SELECTED_BET_VOCABULARY.fractions)
_HOURLY_ROUND_SCENARIOS = (60, 100, 150)


@dataclass(frozen=True, slots=True)
class SummaryEstimate:
    mean: float
    standard_error: float
    sample_count: int

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        margin = 1.96 * self.standard_error
        return (self.mean - margin, self.mean + margin)


@dataclass(frozen=True, slots=True)
class PolicySlice:
    initial_bankroll: float
    log_growth: float
    round_count: int
    mean_bankroll_return_per_round: SummaryEstimate
    mean_profit_units_per_round: SummaryEstimate
    mean_log_growth_per_100_rounds: SummaryEstimate
    trajectory: tuple[BankrollChartPoint, ...]


@dataclass(frozen=True, slots=True)
class BreakdownSlice:
    category: str
    policy: str
    bin_name: str
    round_count: int
    estimate: SummaryEstimate


@dataclass(frozen=True, slots=True)
class ContextSlice:
    decision_count: int
    truncated_decision_count: int
    total_history_tokens_dropped: int
    maximum_original_length: int
    maximum_tokens_dropped: int


@dataclass(frozen=True, slots=True)
class EvaluationSlice:
    shoe_start: int
    shoe_count: int
    simulation_seed: int
    paired_advantage: SummaryEstimate
    transformer: PolicySlice
    hi_lo: PolicySlice
    breakdowns: tuple[BreakdownSlice, ...]
    context: ContextSlice | None


@dataclass(frozen=True, slots=True)
class AggregatePolicy:
    initial_bankroll: float
    log_growth: float
    round_count: int
    mean_bankroll_return_per_round: SummaryEstimate
    mean_profit_units_per_round: SummaryEstimate
    mean_log_growth_per_100_rounds: SummaryEstimate
    trajectory: tuple[BankrollChartPoint, ...]

    @property
    def final_bankroll(self) -> float:
        return self.initial_bankroll * exp(self.log_growth)


@dataclass(frozen=True, slots=True)
class AggregateBreakdown:
    category: str
    policy: str
    bin_name: str
    round_count: int
    estimate: SummaryEstimate


@dataclass(frozen=True, slots=True)
class AggregateContext:
    instrumented_report_count: int
    uninstrumented_report_count: int
    measured_decision_count: int
    truncated_decision_count: int
    total_history_tokens_dropped: int
    maximum_original_length: int
    maximum_tokens_dropped: int


@dataclass(frozen=True, slots=True)
class AggregateBankrollReport:
    shoe_start: int
    shoe_count: int
    simulation_seed: int
    source_report_count: int
    paired_advantage: SummaryEstimate
    transformer: AggregatePolicy
    hi_lo: AggregatePolicy
    breakdowns: tuple[AggregateBreakdown, ...]
    context: AggregateContext

    @property
    def combined_policy_round_count(self) -> int:
        return self.transformer.round_count + self.hi_lo.round_count

    @property
    def verdict(self) -> str:
        lower, upper = self.paired_advantage.confidence_interval_95
        if lower > 0:
            return "transformer_outperforms_hi_lo"
        if upper < 0:
            return "transformer_underperforms_hi_lo"
        return "inconclusive"


def read_evaluation_slice(path: Path) -> EvaluationSlice:
    """Read one atomic evaluator report into a typed summary."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    report = _mapping(raw, "evaluation report")
    methodology = _mapping(
        _required(report, "methodology"),
        "methodology",
    )
    policies = _mapping(_required(report, "policies"), "policies")
    context_raw = report.get("transformer_context_statistics")
    return EvaluationSlice(
        shoe_start=_optional_integer(methodology, "shoe_start", 0),
        shoe_count=_integer(report, "shoe_count"),
        simulation_seed=_integer(methodology, "simulation_seed"),
        paired_advantage=_estimate(
            _required(
                report,
                "paired_transformer_log_growth_advantage_per_shoe",
            ),
            "paired advantage",
        ),
        transformer=_policy(
            _required(policies, "transformer"),
            "transformer",
        ),
        hi_lo=_policy(_required(policies, "hi-lo"), "hi-lo"),
        breakdowns=_breakdowns(
            _required(report, "breakdowns"),
        ),
        context=(
            None
            if context_raw is None
            else _context(context_raw)
        ),
    )


def aggregate_evaluation_slices(
    slices: tuple[EvaluationSlice, ...],
) -> AggregateBankrollReport:
    """Pool ordered disjoint reports using exact within/between-shard variance."""

    if not slices:
        raise ValueError("aggregation needs at least one report")
    ordered = tuple(sorted(slices, key=lambda item: item.shoe_start))
    seed = ordered[0].simulation_seed
    next_start = ordered[0].shoe_start
    for item in ordered:
        if item.simulation_seed != seed:
            raise ValueError("evaluation reports use different simulation seeds")
        if item.shoe_start != next_start:
            raise ValueError("evaluation shoe ranges are not contiguous")
        if item.paired_advantage.sample_count != item.shoe_count:
            raise ValueError("paired estimate does not cover every report shoe")
        next_start += item.shoe_count

    transformer_slices = tuple(item.transformer for item in ordered)
    hi_lo_slices = tuple(item.hi_lo for item in ordered)
    return AggregateBankrollReport(
        shoe_start=ordered[0].shoe_start,
        shoe_count=sum(item.shoe_count for item in ordered),
        simulation_seed=seed,
        source_report_count=len(ordered),
        paired_advantage=pool_estimates(
            tuple(item.paired_advantage for item in ordered)
        ),
        transformer=_aggregate_policy(transformer_slices),
        hi_lo=_aggregate_policy(hi_lo_slices),
        breakdowns=_aggregate_breakdowns(ordered),
        context=_aggregate_context(ordered),
    )


def pool_estimates(
    estimates: tuple[SummaryEstimate, ...],
) -> SummaryEstimate:
    """Pool sample means and standard errors across disjoint observations."""

    if not estimates:
        raise ValueError("pooling needs at least one estimate")
    if any(estimate.sample_count <= 0 for estimate in estimates):
        raise ValueError("pooled estimates need positive sample counts")
    sample_count = sum(estimate.sample_count for estimate in estimates)
    mean = sum(
        estimate.sample_count * estimate.mean
        for estimate in estimates
    ) / sample_count
    sum_squared_deviations = 0.0
    for estimate in estimates:
        within_sample_variance = (
            estimate.standard_error**2 * estimate.sample_count
        )
        sum_squared_deviations += (
            (estimate.sample_count - 1) * within_sample_variance
            + estimate.sample_count * (estimate.mean - mean) ** 2
        )
    standard_error = (
        0.0
        if sample_count == 1
        else sqrt(
            sum_squared_deviations
            / (sample_count - 1)
            / sample_count
        )
    )
    return SummaryEstimate(
        mean=mean,
        standard_error=standard_error,
        sample_count=sample_count,
    )


def aggregate_report_data(
    report: AggregateBankrollReport,
) -> dict[str, object]:
    """Create the compact committed JSON representation."""

    return {
        "methodology": {
            "corpus": "fresh deterministic six-deck H17 simulation shoes",
            "simulation_seed": report.simulation_seed,
            "shoe_start": report.shoe_start,
            "shoe_pairing": "same replay order and cut-card position",
            "round_pairing": (
                "card allocation may diverge after policies choose different actions"
            ),
            "variance_pooling": (
                "per-shoe sample means and variances pooled with both within- "
                "and between-report sums of squares"
            ),
            "transformer_context_overflow": (
                "remove only the minimum oldest visible-history card tokens "
                "needed to fit 256; preserve the complete current decision"
            ),
        },
        "source_report_count": report.source_report_count,
        "shoe_count": report.shoe_count,
        "round_counts": {
            "transformer": report.transformer.round_count,
            "hi-lo": report.hi_lo.round_count,
            "combined_policy_rounds": report.combined_policy_round_count,
        },
        "primary_result": {
            "metric": (
                "paired transformer-minus-Hi-Lo log-bankroll growth per shoe"
            ),
            "verdict": report.verdict,
            "estimate": _estimate_data(report.paired_advantage),
        },
        "transformer_context_statistics": {
            "instrumented_report_count": (
                report.context.instrumented_report_count
            ),
            "uninstrumented_report_count": (
                report.context.uninstrumented_report_count
            ),
            "measured_decision_count": report.context.measured_decision_count,
            "truncated_decision_count": (
                report.context.truncated_decision_count
            ),
            "total_history_tokens_dropped": (
                report.context.total_history_tokens_dropped
            ),
            "maximum_original_length": (
                report.context.maximum_original_length
            ),
            "maximum_tokens_dropped": (
                report.context.maximum_tokens_dropped
            ),
            "interpretation": (
                "uninstrumented reports used the former fail-on-overflow "
                "behavior, so they completed without any overlength decision"
            ),
        },
        "policies": {
            "transformer": _aggregate_policy_data(report.transformer),
            "hi-lo": _aggregate_policy_data(report.hi_lo),
        },
        "breakdowns": _aggregate_breakdown_data(report.breakdowns),
    }


def write_aggregate_report(
    report: AggregateBankrollReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(
            aggregate_report_data(report),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def write_aggregate_chart(
    report: AggregateBankrollReport,
    output_path: Path,
) -> None:
    write_bankroll_series_chart(
        report.transformer.trajectory,
        report.hi_lo.trajectory,
        output_path,
        title=(
            f"Transformer vs Hi-Lo across "
            f"{report.combined_policy_round_count:,} policy-rounds"
        ),
    )


def _aggregate_policy(
    slices: tuple[PolicySlice, ...],
) -> AggregatePolicy:
    initial_bankroll = slices[0].initial_bankroll
    if any(item.initial_bankroll != initial_bankroll for item in slices):
        raise ValueError("policy reports use different initial bankrolls")
    return AggregatePolicy(
        initial_bankroll=initial_bankroll,
        log_growth=sum(item.log_growth for item in slices),
        round_count=sum(item.round_count for item in slices),
        mean_bankroll_return_per_round=pool_estimates(
            tuple(item.mean_bankroll_return_per_round for item in slices)
        ),
        mean_profit_units_per_round=pool_estimates(
            tuple(item.mean_profit_units_per_round for item in slices)
        ),
        mean_log_growth_per_100_rounds=pool_estimates(
            tuple(item.mean_log_growth_per_100_rounds for item in slices)
        ),
        trajectory=_stitch_trajectories(slices),
    )


def _stitch_trajectories(
    slices: tuple[PolicySlice, ...],
    *,
    maximum_points: int = 5_000,
) -> tuple[BankrollChartPoint, ...]:
    current_bankroll = slices[0].initial_bankroll
    round_offset = 0
    points = [BankrollChartPoint(0, current_bankroll)]
    for item in slices:
        for point in item.trajectory:
            points.append(
                BankrollChartPoint(
                    round_number=round_offset + point.round_number,
                    bankroll=(
                        current_bankroll
                        * point.bankroll
                        / item.initial_bankroll
                    ),
                )
            )
        current_bankroll *= exp(item.log_growth)
        round_offset += item.round_count
        if (
            points[-1].round_number != round_offset
            or abs(points[-1].bankroll - current_bankroll) > 1e-12
        ):
            points.append(
                BankrollChartPoint(round_offset, current_bankroll)
            )
    if len(points) <= maximum_points:
        return tuple(points)
    step = max(1, len(points) // (maximum_points - 1))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return tuple(sampled)


def _aggregate_breakdowns(
    slices: tuple[EvaluationSlice, ...],
) -> tuple[AggregateBreakdown, ...]:
    grouped: dict[
        tuple[str, str, str],
        list[BreakdownSlice],
    ] = {}
    for item in slices:
        for breakdown in item.breakdowns:
            key = (
                breakdown.category,
                breakdown.policy,
                breakdown.bin_name,
            )
            grouped.setdefault(key, []).append(breakdown)
    return tuple(
        AggregateBreakdown(
            category=key[0],
            policy=key[1],
            bin_name=key[2],
            round_count=sum(item.round_count for item in values),
            estimate=pool_estimates(
                tuple(item.estimate for item in values)
            ),
        )
        for key, values in sorted(grouped.items())
    )


def _aggregate_context(
    slices: tuple[EvaluationSlice, ...],
) -> AggregateContext:
    contexts = tuple(
        item.context for item in slices if item.context is not None
    )
    return AggregateContext(
        instrumented_report_count=len(contexts),
        uninstrumented_report_count=len(slices) - len(contexts),
        measured_decision_count=sum(item.decision_count for item in contexts),
        truncated_decision_count=sum(
            item.truncated_decision_count for item in contexts
        ),
        total_history_tokens_dropped=sum(
            item.total_history_tokens_dropped for item in contexts
        ),
        maximum_original_length=max(
            (item.maximum_original_length for item in contexts),
            default=0,
        ),
        maximum_tokens_dropped=max(
            (item.maximum_tokens_dropped for item in contexts),
            default=0,
        ),
    )


def _aggregate_policy_data(
    policy: AggregatePolicy,
) -> dict[str, object]:
    return_estimate = policy.mean_bankroll_return_per_round
    return {
        "initial_bankroll": policy.initial_bankroll,
        "final_bankroll": policy.final_bankroll,
        "log_growth": policy.log_growth,
        "round_count": policy.round_count,
        "mean_bankroll_return_per_round": _estimate_data(
            return_estimate
        ),
        "ev": {
            "minimum_bet_fraction_of_bankroll": _MINIMUM_BET_FRACTION,
            "bankroll_percent_per_100_rounds": _estimate_data(
                _scale_estimate(return_estimate, 10_000)
            ),
            "minimum_bet_units_per_100_rounds": _estimate_data(
                _scale_estimate(
                    return_estimate,
                    100 / _MINIMUM_BET_FRACTION,
                )
            ),
            "minimum_bet_units_per_hour": {
                str(rounds_per_hour): _estimate_data(
                    _scale_estimate(
                        return_estimate,
                        rounds_per_hour / _MINIMUM_BET_FRACTION,
                    )
                )
                for rounds_per_hour in _HOURLY_ROUND_SCENARIOS
            },
        },
        "mean_profit_units_per_round": _estimate_data(
            policy.mean_profit_units_per_round
        ),
        "mean_log_growth_per_100_rounds": _estimate_data(
            policy.mean_log_growth_per_100_rounds
        ),
        "trajectory": [
            {
                "global_round_index": point.round_number,
                "bankroll_after": point.bankroll,
            }
            for point in policy.trajectory
        ],
    }


def _aggregate_breakdown_data(
    breakdowns: tuple[AggregateBreakdown, ...],
) -> dict[str, object]:
    result: dict[str, object] = {}
    categories = sorted({item.category for item in breakdowns})
    for category in categories:
        policies: dict[str, object] = {}
        for policy in _POLICY_NAMES:
            policies[policy] = [
                {
                    "bin": item.bin_name,
                    "round_count": item.round_count,
                    "mean_log_growth_per_100_rounds": (
                        _estimate_data(item.estimate)
                    ),
                }
                for item in breakdowns
                if item.category == category and item.policy == policy
            ]
        result[category] = policies
    return result


def _estimate_data(estimate: SummaryEstimate) -> dict[str, object]:
    return {
        "mean": estimate.mean,
        "standard_error": estimate.standard_error,
        "confidence_interval_95": list(
            estimate.confidence_interval_95
        ),
        "sample_count": estimate.sample_count,
    }


def _scale_estimate(
    estimate: SummaryEstimate,
    factor: float,
) -> SummaryEstimate:
    return SummaryEstimate(
        mean=estimate.mean * factor,
        standard_error=estimate.standard_error * abs(factor),
        sample_count=estimate.sample_count,
    )


def _policy(value: object, field: str) -> PolicySlice:
    data = _mapping(value, field)
    trajectory = tuple(
        BankrollChartPoint(
            round_number=_integer(
                _mapping(item, "trajectory point"),
                "global_round_index",
            ),
            bankroll=_number(
                _mapping(item, "trajectory point"),
                "bankroll_after",
            ),
        )
        for item in _list(_required(data, "trajectory"), "trajectory")
    )
    return PolicySlice(
        initial_bankroll=_number(data, "initial_bankroll"),
        log_growth=_number(data, "log_growth"),
        round_count=_integer(data, "round_count"),
        mean_bankroll_return_per_round=_estimate(
            _required(data, "mean_bankroll_return_per_round"),
            f"{field} mean bankroll return",
        ),
        mean_profit_units_per_round=_estimate(
            _required(data, "mean_profit_units_per_round"),
            f"{field} mean profit",
        ),
        mean_log_growth_per_100_rounds=_estimate(
            _required(data, "mean_log_growth_per_100_rounds"),
            f"{field} mean log growth",
        ),
        trajectory=trajectory,
    )


def _breakdowns(value: object) -> tuple[BreakdownSlice, ...]:
    categories = _mapping(value, "breakdowns")
    result: list[BreakdownSlice] = []
    for category, raw_policies in categories.items():
        policies = _mapping(raw_policies, f"{category} policies")
        for policy in _POLICY_NAMES:
            entries = _list(
                _required(policies, policy),
                f"{category} {policy} entries",
            )
            for raw_entry in entries:
                entry = _mapping(raw_entry, "breakdown entry")
                result.append(
                    BreakdownSlice(
                        category=category,
                        policy=policy,
                        bin_name=_string(entry, "bin"),
                        round_count=_integer(entry, "round_count"),
                        estimate=_estimate(
                            _required(
                                entry,
                                "mean_log_growth_per_100_rounds",
                            ),
                            "breakdown estimate",
                        ),
                    )
                )
    return tuple(result)


def _context(value: object) -> ContextSlice:
    data = _mapping(value, "transformer context statistics")
    return ContextSlice(
        decision_count=_integer(data, "decision_count"),
        truncated_decision_count=_integer(
            data,
            "truncated_decision_count",
        ),
        total_history_tokens_dropped=_integer(
            data,
            "total_history_tokens_dropped",
        ),
        maximum_original_length=_integer(
            data,
            "maximum_original_length",
        ),
        maximum_tokens_dropped=_integer(
            data,
            "maximum_tokens_dropped",
        ),
    )


def _estimate(value: object, field: str) -> SummaryEstimate:
    data = _mapping(value, field)
    return SummaryEstimate(
        mean=_number(data, "mean"),
        standard_error=_number(data, "standard_error"),
        sample_count=_integer(data, "sample_count"),
    )


def _required(data: dict[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"missing JSON field: {key}")
    return data[key]


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    untyped = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in untyped):
        raise ValueError(f"{field} keys must be strings")
    return {str(key): item for key, item in untyped.items()}


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _integer(data: dict[str, object], field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_integer(
    data: dict[str, object],
    field: str,
    default: int,
) -> int:
    if field not in data:
        return default
    return _integer(data, field)


def _number(data: dict[str, object], field: str) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _string(data: dict[str, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pool contiguous atomic bankroll reports.",
    )
    parser.add_argument("output_path", type=Path)
    parser.add_argument("input_paths", nargs="+", type=Path)
    parser.add_argument("--chart-path", type=Path)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    report = aggregate_evaluation_slices(
        tuple(read_evaluation_slice(path) for path in arguments.input_paths)
    )
    write_aggregate_report(report, arguments.output_path)
    if arguments.chart_path is not None:
        write_aggregate_chart(report, arguments.chart_path)
    print(
        json.dumps(
            {
                "output_path": str(arguments.output_path),
                "shoe_count": report.shoe_count,
                "transformer_rounds": report.transformer.round_count,
                "hi_lo_rounds": report.hi_lo.round_count,
                "combined_policy_rounds": (
                    report.combined_policy_round_count
                ),
                "paired_advantage": _estimate_data(
                    report.paired_advantage
                ),
                "verdict": report.verdict,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
