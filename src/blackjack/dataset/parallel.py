"""Coordinate resumable complete-shoe generator processes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from blackjack.dataset.cache import SQLiteLabelCache
from blackjack.dataset.records import DatasetConfiguration
from blackjack.dataset.runner import (
    GenerationProgress,
    GenerationRunSummary,
    run_resumable_generation,
)


@dataclass(frozen=True, slots=True)
class ParallelGenerationSummary:
    worker_count: int
    shoe_count: int
    completed_shoe_count: int
    new_decisions: int
    cached_decisions: int
    elapsed_seconds: float
    assembled_all_splits: bool


def _progress(update: GenerationProgress) -> None:
    if update.shoe_completed:
        print(
            f"[shard shoe {update.shoe_id:04d}] complete "
            f"({update.selected_shoe_number}/"
            f"{update.selected_shoe_count})",
            flush=True,
        )


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
        progress=_progress,
    )


def run_parallel_generation(
    configuration: DatasetConfiguration,
    output_directory: Path,
    *,
    worker_count: int,
    label_cache_path: Path | None = None,
) -> ParallelGenerationSummary:
    """Generate every modulo-sharded shoe with independent processes."""

    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    if worker_count > configuration.shoe_count:
        raise ValueError("worker count cannot exceed shoe count")
    cache_path = (
        output_directory / "oracle-labels.sqlite3"
        if label_cache_path is None
        else label_cache_path
    )
    SQLiteLabelCache(cache_path)
    started = perf_counter()
    summaries: list[GenerationRunSummary] = []
    if worker_count == 1:
        summaries.append(
            _run_shard(
                configuration,
                output_directory,
                0,
                1,
                cache_path,
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_shards = {
                executor.submit(
                    _run_shard,
                    configuration,
                    output_directory,
                    shard_index,
                    worker_count,
                    cache_path,
                ): shard_index
                for shard_index in range(worker_count)
            }
            for future in as_completed(future_shards):
                shard_index = future_shards[future]
                summary = future.result()
                summaries.append(summary)
                print(
                    f"worker shard {shard_index} complete: "
                    f"{len(summary.completed_shoe_ids)} shoes, "
                    f"{summary.new_decisions} new decisions, "
                    f"{summary.cached_decisions} checkpointed decisions reused",
                    flush=True,
                )

    completed = sum(len(summary.completed_shoe_ids) for summary in summaries)
    return ParallelGenerationSummary(
        worker_count=worker_count,
        shoe_count=configuration.shoe_count,
        completed_shoe_count=completed,
        new_decisions=sum(summary.new_decisions for summary in summaries),
        cached_decisions=sum(
            summary.cached_decisions for summary in summaries
        ),
        elapsed_seconds=perf_counter() - started,
        assembled_all_splits=any(
            summary.assembled_all_splits for summary in summaries
        ),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a complete dataset with process-level shoe shards.",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--shoe-count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--master-seed", type=int, default=20250725)
    parser.add_argument("--split-seed", type=int, default=20250726)
    parser.add_argument("--exploration-seed", type=int, default=20250727)
    parser.add_argument(
        "--exploration-probability",
        type=Fraction,
        default=Fraction(1, 5),
    )
    parser.add_argument("--bet-rollout-seed", type=int, default=20250728)
    parser.add_argument("--bet-rollouts", type=int, default=1_000_000)
    parser.add_argument("--play-rollout-seed", type=int, default=20250730)
    parser.add_argument("--play-rollouts", type=int, default=1_000_000)
    parser.add_argument("--label-cache", type=Path)
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    configuration = DatasetConfiguration(
        master_seed=arguments.master_seed,
        split_seed=arguments.split_seed,
        exploration_seed=arguments.exploration_seed,
        shoe_count=arguments.shoe_count,
        exploration_probability=arguments.exploration_probability,
        bet_rollout_seed=arguments.bet_rollout_seed,
        bet_rollouts=arguments.bet_rollouts,
        play_rollout_seed=arguments.play_rollout_seed,
        play_rollouts=arguments.play_rollouts,
    )
    try:
        summary = run_parallel_generation(
            configuration,
            arguments.output,
            worker_count=arguments.workers,
            label_cache_path=arguments.label_cache,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed decisions remain checkpointed; "
            "re-run the same command to resume.",
            flush=True,
        )
        raise SystemExit(130) from None
    print(
        f"complete: {summary.completed_shoe_count}/"
        f"{summary.shoe_count} shoes, {summary.new_decisions} new decisions, "
        f"{summary.cached_decisions} checkpointed decisions reused, "
        f"{summary.elapsed_seconds:.1f}s",
        flush=True,
    )
    if not summary.assembled_all_splits:
        raise SystemExit("all shoes completed but final splits were not assembled")


if __name__ == "__main__":
    main()
