from __future__ import annotations

from math import sqrt

import pytest

from blackjack.analysis.bankroll_svg import BankrollChartPoint
from blackjack.training.bankroll_aggregate import (
    ContextSlice,
    EvaluationSlice,
    PolicySlice,
    SummaryEstimate,
    aggregate_evaluation_slices,
    pool_estimates,
)


def test_pool_estimates_preserves_within_and_between_group_variance() -> None:
    pooled = pool_estimates(
        (
            SummaryEstimate(mean=2, standard_error=1, sample_count=2),
            SummaryEstimate(mean=6, standard_error=1, sample_count=2),
        )
    )

    assert pooled.mean == 4
    assert pooled.sample_count == 4
    assert pooled.standard_error == pytest.approx(sqrt((20 / 3) / 4))


def test_aggregate_requires_contiguous_shoe_ranges() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        aggregate_evaluation_slices(
            (
                _slice(shoe_start=0, mean=1),
                _slice(shoe_start=2, mean=2),
            )
        )


def test_aggregate_reports_exact_round_totals_and_context_overflow() -> None:
    report = aggregate_evaluation_slices(
        (
            _slice(shoe_start=0, mean=1),
            _slice(shoe_start=1, mean=3, with_context=True),
        )
    )

    assert report.shoe_count == 2
    assert report.transformer.round_count == 20
    assert report.hi_lo.round_count == 22
    assert report.combined_policy_round_count == 42
    assert report.paired_advantage.mean == 2
    assert report.context.instrumented_report_count == 1
    assert report.context.uninstrumented_report_count == 1
    assert report.context.truncated_decision_count == 1
    assert report.context.maximum_original_length == 257


def _slice(
    *,
    shoe_start: int,
    mean: float,
    with_context: bool = False,
) -> EvaluationSlice:
    estimate = SummaryEstimate(
        mean=mean,
        standard_error=0,
        sample_count=1,
    )
    transformer = _policy(round_count=10, estimate=estimate)
    hi_lo = _policy(round_count=11, estimate=estimate)
    return EvaluationSlice(
        shoe_start=shoe_start,
        shoe_count=1,
        simulation_seed=7,
        paired_advantage=estimate,
        transformer=transformer,
        hi_lo=hi_lo,
        breakdowns=(),
        context=(
            ContextSlice(
                decision_count=20,
                truncated_decision_count=1,
                total_history_tokens_dropped=1,
                maximum_original_length=257,
                maximum_tokens_dropped=1,
            )
            if with_context
            else None
        ),
    )


def _policy(
    *,
    round_count: int,
    estimate: SummaryEstimate,
) -> PolicySlice:
    return PolicySlice(
        initial_bankroll=100,
        log_growth=0.1,
        round_count=round_count,
        mean_bankroll_return_per_round=estimate,
        mean_profit_units_per_round=estimate,
        mean_log_growth_per_100_rounds=estimate,
        trajectory=(
            BankrollChartPoint(
                round_number=round_count,
                bankroll=110,
            ),
        ),
    )
