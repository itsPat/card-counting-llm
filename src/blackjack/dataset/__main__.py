"""Command-line entry point for reproducible dataset generation."""

from __future__ import annotations

import argparse
import os
from fractions import Fraction
from pathlib import Path

from blackjack.dataset.records import DatasetConfiguration
from blackjack.dataset.runner import (
    GenerationProgress,
    run_resumable_generation,
)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _progress(update: GenerationProgress) -> None:
    prefix = (
        f"[shoe {update.shoe_id:03d} "
        f"{update.selected_shoe_number}/{update.selected_shoe_count}]"
    )
    if update.shoe_completed:
        source = "already complete" if update.cached else "complete"
        print(
            f"{prefix} {source}; shard ETA "
            f"{_duration(update.estimated_remaining_seconds)}",
            flush=True,
        )
    elif not update.cached:
        print(
            f"{prefix} decision {update.decision_index} "
            f"{update.decision_kind} labeled in "
            f"{_duration(update.label_seconds)}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate blackjack labels with atomic checkpoints and "
            "complete-shoe shards."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--shoe-count", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=20250725)
    parser.add_argument("--split-seed", type=int, default=20250726)
    parser.add_argument("--exploration-seed", type=int, default=20250727)
    parser.add_argument(
        "--exploration-probability",
        type=Fraction,
        default=Fraction(1, 5),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based worker index",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="number of workers sharing the shoe IDs",
    )
    parser.add_argument(
        "--benchmark",
        nargs="?",
        const=2,
        type=int,
        metavar="NEW_DECISIONS",
        help=(
            "stop after this many newly computed labels; defaults to 2 "
            "and preserves checkpoints"
        ),
    )
    parser.add_argument(
        "--bet-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=(
            "deprecated compatibility option; native fixed-policy rollouts "
            "do not spawn nested workers"
        ),
    )
    parser.add_argument(
        "--bet-rollout-seed",
        type=int,
        default=20250728,
        help="master seed used to derive a replay seed for each bet state",
    )
    parser.add_argument(
        "--bet-rollouts",
        type=int,
        default=1_000_000,
        help="seeded fixed-policy rounds simulated for each bet label",
    )
    parser.add_argument(
        "--play-rollout-seed",
        type=int,
        default=20250730,
        help="master seed used to derive a replay seed for each play state",
    )
    parser.add_argument(
        "--play-rollouts",
        type=int,
        default=1_000_000,
        help="fixed-continuation rollouts simulated per legal play action",
    )
    parser.add_argument(
        "--label-cache",
        type=Path,
        help=("shared SQLite label cache; defaults to OUTPUT/oracle-labels.sqlite3"),
    )
    arguments = parser.parse_args()
    label_cache = (
        arguments.label_cache
        if arguments.label_cache is not None
        else arguments.output / "oracle-labels.sqlite3"
    )

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
        summary = run_resumable_generation(
            configuration,
            arguments.output,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
            maximum_new_decisions=arguments.benchmark,
            bet_worker_count=arguments.bet_workers,
            label_cache_path=label_cache,
            progress=_progress,
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Every previously completed decision remains "
            "checkpointed; re-run the same command to resume.",
            flush=True,
        )
        raise SystemExit(130) from None
    print(
        f"{summary.dataset_id}: {summary.new_decisions} new labels, "
        f"{summary.cached_decisions} reused, "
        f"{_duration(summary.elapsed_seconds)} wall time.",
        flush=True,
    )
    cache = summary.label_cache
    print(
        "SQLite label cache: "
        f"{cache.hits} hits, {cache.misses} misses, "
        f"{cache.waits} waits, {cache.writes} writes.",
        flush=True,
    )
    if summary.new_decisions:
        mean = summary.label_seconds / summary.new_decisions
        print(
            f"Observed label mean: {_duration(mean)}. "
            "A whole-shoe ETA appears after the first completed shoe.",
            flush=True,
        )
    if summary.stopped_at_limit:
        print(
            "Benchmark limit reached. Re-run the same command to resume, or "
            "remove --benchmark to continue through the selected shoes.",
            flush=True,
        )
    elif summary.assembled_all_splits:
        print("All shoe shards are complete; final split JSONL files assembled.")
    else:
        print("Selected shard complete; other shoe shards remain.")


if __name__ == "__main__":
    main()
