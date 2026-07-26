"""Resumable, shardable execution for expensive dataset generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from blackjack.dataset.cache import (
    CachedDatasetOracle,
    LabelCacheStatistics,
    SQLiteLabelCache,
)
from blackjack.dataset.generation import (
    SCHEMA_VERSION,
    DatasetOracle,
    dataset_id,
    generate_shoe_examples,
    prepare_shoes,
)
from blackjack.dataset.io import (
    assemble_completed_splits,
    initialize_output,
    read_shoe_checkpoints,
    read_shoe_shard,
    write_decision_checkpoint,
    write_shoe_shard,
)
from blackjack.dataset.labeling import ProductionDatasetOracle
from blackjack.dataset.records import (
    DatasetConfiguration,
    DatasetManifest,
    DecisionExample,
)


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    shoe_id: int
    selected_shoe_number: int
    selected_shoe_count: int
    decision_index: int | None
    decision_kind: str | None
    cached: bool
    label_seconds: float
    shoe_completed: bool
    estimated_remaining_seconds: float | None


@dataclass(frozen=True, slots=True)
class GenerationRunSummary:
    dataset_id: str
    selected_shoe_ids: tuple[int, ...]
    completed_shoe_ids: tuple[int, ...]
    new_decisions: int
    cached_decisions: int
    label_seconds: float
    elapsed_seconds: float
    stopped_at_limit: bool
    assembled_all_splits: bool
    label_cache: LabelCacheStatistics


type ProgressCallback = Callable[[GenerationProgress], None]


def run_resumable_generation(
    configuration: DatasetConfiguration,
    output_directory: Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    maximum_new_decisions: int | None = None,
    bet_worker_count: int = 1,
    label_cache_path: Path | None = None,
    oracle: DatasetOracle | None = None,
    progress: ProgressCallback | None = None,
) -> GenerationRunSummary:
    """Generate selected complete-shoe shards with per-decision checkpoints."""

    output = output_directory
    if shard_count <= 0:
        raise ValueError("shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index must lie in [0, shard_count)")
    if maximum_new_decisions is not None and maximum_new_decisions <= 0:
        raise ValueError("maximum new decisions must be positive")
    if bet_worker_count <= 0:
        raise ValueError("bet worker count must be positive")

    prepared = prepare_shoes(configuration)
    identifier = dataset_id(configuration, prepared)
    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        dataset_id=identifier,
        configuration=configuration,
        shoes=tuple(item.manifest for item in prepared),
    )
    initialize_output(manifest, output)
    selected = tuple(
        item for item in prepared if item.manifest.shoe_id % shard_count == shard_index
    )
    if oracle is None:
        base_oracle: DatasetOracle = ProductionDatasetOracle(
            bet_rollout_seed=configuration.bet_rollout_seed,
            bet_rollouts=configuration.bet_rollouts,
            play_rollout_seed=configuration.play_rollout_seed,
            play_rollouts=configuration.play_rollouts,
        )
        cache_namespace = (
            "blackjack.dataset.ProductionDatasetOracle:"
            f"bet_method={configuration.bet_evaluation_method.value}:"
            f"seed={configuration.bet_rollout_seed}:"
            f"rollouts={configuration.bet_rollouts}:"
            f"play_method={configuration.play_evaluation_method.value}:"
            f"play_seed={configuration.play_rollout_seed}:"
            f"play_rollouts={configuration.play_rollouts}"
        )
    else:
        base_oracle = oracle
        cache_namespace = None
    cache = SQLiteLabelCache(label_cache_path) if label_cache_path is not None else None
    labeler: DatasetOracle = (
        CachedDatasetOracle(
            base_oracle,
            cache,
            namespace=cache_namespace,
        )
        if cache is not None
        else base_oracle
    )
    started = perf_counter()
    completed_ids: list[int] = []
    new_decisions = 0
    cached_decisions = 0
    label_seconds_total = 0.0
    stopped_at_limit = False
    timed_shoes_completed = 0

    for selected_index, item in enumerate(selected, start=1):
        shoe_id = item.manifest.shoe_id
        existing_shard = read_shoe_shard(output, shoe_id)
        if existing_shard:
            replayed = generate_shoe_examples(
                item,
                identifier,
                configuration,
                labeler,
                cached_examples=existing_shard,
                cache_must_complete=True,
            )
            if not replayed.completed or replayed.new_decisions:
                raise AssertionError("completed shard validation must stay cached")
            completed_ids.append(shoe_id)
            cached_decisions += replayed.cached_decisions
            _notify_shoe_complete(
                progress,
                shoe_id,
                selected_index,
                len(selected),
                cached=True,
                estimated_remaining_seconds=None,
            )
            continue

        cached = read_shoe_checkpoints(output, shoe_id)
        remaining_limit = (
            None
            if maximum_new_decisions is None
            else maximum_new_decisions - new_decisions
        )
        if remaining_limit is not None and remaining_limit <= 0:
            stopped_at_limit = True
            break

        def on_decision(
            example: DecisionExample,
            was_cached: bool,
            label_seconds: float,
            current_shoe_id: int = shoe_id,
            current_selected_index: int = selected_index,
        ) -> None:
            nonlocal label_seconds_total
            if not was_cached:
                write_decision_checkpoint(output, example)
                label_seconds_total += label_seconds
            if progress is not None:
                progress(
                    GenerationProgress(
                        shoe_id=current_shoe_id,
                        selected_shoe_number=current_selected_index,
                        selected_shoe_count=len(selected),
                        decision_index=example.decision_index,
                        decision_kind=example.kind.value,
                        cached=was_cached,
                        label_seconds=label_seconds,
                        shoe_completed=False,
                        estimated_remaining_seconds=None,
                    )
                )

        result = generate_shoe_examples(
            item,
            identifier,
            configuration,
            labeler,
            cached_examples=cached,
            maximum_new_decisions=remaining_limit,
            callback=on_decision,
        )
        new_decisions += result.new_decisions
        cached_decisions += result.cached_decisions
        if not result.completed:
            stopped_at_limit = True
            break
        write_shoe_shard(output, shoe_id, result.examples)
        completed_ids.append(shoe_id)
        timed_shoes_completed += 1
        average_shoe_seconds = (perf_counter() - started) / timed_shoes_completed
        _notify_shoe_complete(
            progress,
            shoe_id,
            selected_index,
            len(selected),
            cached=False,
            estimated_remaining_seconds=average_shoe_seconds
            * (len(selected) - len(completed_ids)),
        )

    elapsed = perf_counter() - started
    assembled = assemble_completed_splits(manifest, output)
    return GenerationRunSummary(
        dataset_id=identifier,
        selected_shoe_ids=tuple(item.manifest.shoe_id for item in selected),
        completed_shoe_ids=tuple(completed_ids),
        new_decisions=new_decisions,
        cached_decisions=cached_decisions,
        label_seconds=label_seconds_total,
        elapsed_seconds=elapsed,
        stopped_at_limit=stopped_at_limit,
        assembled_all_splits=assembled,
        label_cache=(cache.statistics if cache is not None else LabelCacheStatistics()),
    )


def _notify_shoe_complete(
    progress: ProgressCallback | None,
    shoe_id: int,
    selected_index: int,
    selected_count: int,
    *,
    cached: bool,
    estimated_remaining_seconds: float | None,
) -> None:
    if progress is None:
        return
    progress(
        GenerationProgress(
            shoe_id=shoe_id,
            selected_shoe_number=selected_index,
            selected_shoe_count=selected_count,
            decision_index=None,
            decision_kind=None,
            cached=cached,
            label_seconds=0.0,
            shoe_completed=True,
            estimated_remaining_seconds=estimated_remaining_seconds,
        )
    )
