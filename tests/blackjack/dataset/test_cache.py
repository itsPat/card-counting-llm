from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from time import sleep

from blackjack.dataset import (
    ActionValue,
    CachedDatasetOracle,
    EvaluationMetadata,
    LabeledDecision,
    SQLiteLabelCache,
)
from blackjack.engine import PlayerAction
from blackjack.oracle import Composition, RoundPlayerSituation


class _BetOnlyOracle:
    def __init__(self) -> None:
        self.calls = 0

    def label_bet(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        self.calls += 1
        return _label(composition, unseen_unavailable)

    def label_insurance(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        raise AssertionError("insurance is not used by this cache test")

    def label_play(
        self,
        situation: RoundPlayerSituation,
        legal_actions: tuple[PlayerAction, ...],
    ) -> LabeledDecision:
        raise AssertionError("play is not used by this cache test")


def _label(
    composition: Composition,
    unseen_unavailable: int,
) -> LabeledDecision:
    return LabeledDecision(
        target_token="<BET_MIN>",
        metadata=EvaluationMetadata(
            shoe_composition=composition,
            unseen_unavailable=unseen_unavailable,
            legal_target_tokens=("<BET_MIN>",),
            action_values=(
                ActionValue(
                    token="<BET_MIN>",
                    expected_log_growth=0.0,
                ),
            ),
        ),
    )


def test_sqlite_cache_reuses_labels_across_oracle_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.sqlite3"
    composition = Composition.full_shoe()
    first_oracle = _BetOnlyOracle()
    first = CachedDatasetOracle(first_oracle, SQLiteLabelCache(path))
    expected = first.label_bet(composition, unseen_unavailable=1)
    assert first_oracle.calls == 1

    second_oracle = _BetOnlyOracle()
    second_cache = SQLiteLabelCache(path)
    second = CachedDatasetOracle(second_oracle, second_cache)
    assert second.label_bet(composition, unseen_unavailable=1) == expected
    assert second_oracle.calls == 0
    assert second_cache.statistics.hits == 1


def test_sqlite_claim_prevents_duplicate_concurrent_computation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.sqlite3"
    caches = (
        SQLiteLabelCache(
            path,
            poll_seconds=0.01,
            wait_timeout_seconds=2,
            claim_stale_seconds=4,
        ),
        SQLiteLabelCache(
            path,
            poll_seconds=0.01,
            wait_timeout_seconds=2,
            claim_stale_seconds=4,
        ),
    )
    barrier = Barrier(2)
    lock = Lock()
    calls = 0
    label = _label(Composition.full_shoe(), unseen_unavailable=1)

    def request(cache: SQLiteLabelCache) -> LabeledDecision:
        nonlocal calls
        barrier.wait()

        def compute() -> LabeledDecision:
            nonlocal calls
            with lock:
                calls += 1
            sleep(0.1)
            return label

        return cache.get_or_compute(
            kind="bet",
            state='{"same":"state"}',
            compute=compute,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(request, caches))

    assert results == (label, label)
    assert calls == 1
    statistics = tuple(cache.statistics for cache in caches)
    assert sum(item.misses for item in statistics) == 1
    assert sum(item.hits for item in statistics) == 1
    assert sum(item.waits for item in statistics) == 1
