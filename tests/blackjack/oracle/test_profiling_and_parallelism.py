from __future__ import annotations

import pytest

from blackjack.dataset import ExactDatasetOracle
from blackjack.oracle import (
    CardValue,
    Composition,
    clear_oracle_caches,
    dealer_distribution,
    oracle_cache_profile,
    round_return_distribution,
)


def test_oracle_cache_profile_reports_hits_misses_and_clear() -> None:
    clear_oracle_caches()
    composition = Composition.from_values(
        (
            CardValue.ACE,
            CardValue.FIVE,
            CardValue.SIX,
            CardValue.TEN,
            CardValue.TEN,
        )
    )
    first = dealer_distribution(composition, CardValue.SIX)
    second = dealer_distribution(composition, CardValue.SIX)
    assert first == second
    profile = oracle_cache_profile()
    public = next(
        counter
        for counter in profile.counters
        if counter.name == "dealer_distribution_public"
    )
    assert public.hits == 1
    assert public.misses == 1
    assert public.current_size == 1

    clear_oracle_caches()
    cleared = oracle_cache_profile()
    assert cleared.total_hits == 0
    assert cleared.total_misses == 0
    assert cleared.total_current_size == 0


def test_bounded_processes_preserve_the_exact_round_distribution() -> None:
    composition = Composition.from_values((CardValue.TEN,) * 20)
    serial = round_return_distribution(
        composition,
        unseen_unavailable=0,
        worker_count=1,
    )
    parallel = round_return_distribution(
        composition,
        unseen_unavailable=0,
        worker_count=2,
    )
    assert parallel == serial


def test_bet_worker_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ExactDatasetOracle(bet_worker_count=0)
