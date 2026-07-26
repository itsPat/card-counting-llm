from __future__ import annotations

import argparse

import pytest

from blackjack.dataset.benchmark import (
    GenerationBenchmarkResult,
    parse_worker_counts,
)


def test_benchmark_result_calculates_observed_throughput() -> None:
    result = GenerationBenchmarkResult(
        worker_count=4,
        shoe_count=16,
        decision_count=1_600,
        computed_label_count=1_500,
        cache_hit_count=100,
        elapsed_seconds=40,
    )
    assert result.shoes_per_second == pytest.approx(0.4)
    assert result.decisions_per_second == pytest.approx(40)


@pytest.mark.parametrize("value", ["", "0,1", "1,-2", "1,1"])
def test_worker_count_parser_rejects_invalid_lists(value: str) -> None:
    with pytest.raises((ValueError, argparse.ArgumentTypeError)):
        parse_worker_counts(value)
