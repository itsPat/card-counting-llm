"""Reproducible complete-shoe generation concurrency benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from blackjack.dataset.cache import SQLiteLabelCache
from blackjack.dataset.records import DatasetConfiguration
from blackjack.dataset.runner import (
    GenerationRunSummary,
    run_resumable_generation,
)


@dataclass(frozen=True, slots=True)
class GenerationBenchmarkResult:
    worker_count: int
    shoe_count: int
    decision_count: int
    computed_label_count: int
    cache_hit_count: int
    elapsed_seconds: float

    @property
    def shoes_per_second(self) -> float:
        return self.shoe_count / self.elapsed_seconds

    @property
    def decisions_per_second(self) -> float:
        return self.decision_count / self.elapsed_seconds


def _run_shard(
    configuration: DatasetConfiguration,
    output_directory: Path,
    shard_index: int,
    shard_count: int,
    label_cache_path: Path,
) -> GenerationRunSummary:
    return run_resumable_generation(
        configuration,
        output_directory,
        shard_index=shard_index,
        shard_count=shard_count,
        label_cache_path=label_cache_path,
    )


def run_generation_benchmark(
    configuration: DatasetConfiguration,
    output_directory: Path,
    *,
    worker_count: int,
) -> GenerationBenchmarkResult:
    """Time one fresh generation of the same complete shoes."""

    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    if worker_count > configuration.shoe_count:
        raise ValueError("worker count cannot exceed shoe count")
    if (output_directory / "manifest.json").exists():
        raise ValueError(
            "benchmark output already contains generated labels; "
            "use a fresh directory for an uncached measurement"
        )

    label_cache_path = output_directory / "oracle-labels.sqlite3"
    SQLiteLabelCache(label_cache_path)
    started = perf_counter()
    if worker_count == 1:
        summaries = (
            _run_shard(
                configuration,
                output_directory,
                0,
                1,
                label_cache_path,
            ),
        )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = tuple(
                executor.submit(
                    _run_shard,
                    configuration,
                    output_directory,
                    shard_index,
                    worker_count,
                    label_cache_path,
                )
                for shard_index in range(worker_count)
            )
            summaries = tuple(future.result() for future in futures)
    elapsed_seconds = perf_counter() - started

    completed_shoes = sum(
        len(summary.completed_shoe_ids) for summary in summaries
    )
    if completed_shoes != configuration.shoe_count:
        raise RuntimeError(
            f"benchmark completed {completed_shoes} of "
            f"{configuration.shoe_count} shoes"
        )
    decision_count = sum(
        summary.new_decisions + summary.cached_decisions
        for summary in summaries
    )
    computed_labels = sum(
        summary.label_cache.misses for summary in summaries
    )
    cache_hits = sum(summary.label_cache.hits for summary in summaries)
    return GenerationBenchmarkResult(
        worker_count=worker_count,
        shoe_count=configuration.shoe_count,
        decision_count=decision_count,
        computed_label_count=computed_labels,
        cache_hit_count=cache_hits,
        elapsed_seconds=elapsed_seconds,
    )


def _result_data(
    result: GenerationBenchmarkResult,
) -> dict[str, int | float]:
    return {
        "worker_count": result.worker_count,
        "shoe_count": result.shoe_count,
        "decision_count": result.decision_count,
        "computed_label_count": result.computed_label_count,
        "cache_hit_count": result.cache_hit_count,
        "elapsed_seconds": result.elapsed_seconds,
        "shoes_per_second": result.shoes_per_second,
        "decisions_per_second": result.decisions_per_second,
    }


def _write_report(
    path: Path,
    configuration: DatasetConfiguration,
    results: tuple[GenerationBenchmarkResult, ...],
) -> None:
    report: dict[str, object] = {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "configuration": {
            "shoe_count": configuration.shoe_count,
            "master_seed": configuration.master_seed,
            "bet_rollouts": configuration.bet_rollouts,
            "play_rollouts": configuration.play_rollouts,
        },
        "results": [_result_data(result) for result in results],
    }
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_worker_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in value.split(","))
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "worker counts must be comma-separated positive integers"
        )
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("worker counts must be unique")
    return counts


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fresh complete-shoe dataset generation with several "
            "process counts."
        ),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--shoe-count", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=parse_worker_counts,
        default=(1, 2, 4, 8),
        help="comma-separated process counts; defaults to 1,2,4,8",
    )
    parser.add_argument("--bet-rollouts", type=int, default=1_000_000)
    parser.add_argument("--play-rollouts", type=int, default=1_000_000)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    configuration = DatasetConfiguration(
        shoe_count=arguments.shoe_count,
        bet_rollouts=arguments.bet_rollouts,
        play_rollouts=arguments.play_rollouts,
    )
    if any(count > configuration.shoe_count for count in arguments.workers):
        raise SystemExit("a worker count cannot exceed --shoe-count")
    arguments.output.mkdir(parents=True, exist_ok=True)
    results: list[GenerationBenchmarkResult] = []
    for worker_count in arguments.workers:
        print(
            f"Benchmarking {worker_count} worker(s) across "
            f"{configuration.shoe_count} fresh shoes...",
            flush=True,
        )
        result = run_generation_benchmark(
            configuration,
            arguments.output / f"workers-{worker_count:02d}",
            worker_count=worker_count,
        )
        results.append(result)
        print(
            f"{worker_count} worker(s): {result.elapsed_seconds:.1f}s, "
            f"{result.shoes_per_second:.3f} shoes/s, "
            f"{result.decisions_per_second:.1f} decisions/s",
            flush=True,
        )
        _write_report(
            arguments.output / "benchmark.json",
            configuration,
            tuple(results),
        )


if __name__ == "__main__":
    main()
