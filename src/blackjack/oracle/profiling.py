"""Inspectable counters for the exact oracle's in-process memoization."""

from __future__ import annotations

from dataclasses import dataclass

from blackjack.oracle.dealer import clear_dealer_caches, dealer_cache_counts
from blackjack.oracle.player import clear_player_caches, player_cache_counts
from blackjack.oracle.round_returns import (
    clear_round_return_caches,
    round_return_cache_counts,
)


@dataclass(frozen=True, slots=True)
class OracleCacheCounter:
    name: str
    hits: int
    misses: int
    current_size: int

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0


@dataclass(frozen=True, slots=True)
class OracleCacheProfile:
    counters: tuple[OracleCacheCounter, ...]

    @property
    def total_hits(self) -> int:
        return sum(counter.hits for counter in self.counters)

    @property
    def total_misses(self) -> int:
        return sum(counter.misses for counter in self.counters)

    @property
    def total_current_size(self) -> int:
        return sum(counter.current_size for counter in self.counters)


def oracle_cache_profile() -> OracleCacheProfile:
    raw = (
        *dealer_cache_counts(),
        *player_cache_counts(),
        *round_return_cache_counts(),
    )
    return OracleCacheProfile(
        tuple(
            OracleCacheCounter(
                name=name,
                hits=hits,
                misses=misses,
                current_size=current_size,
            )
            for name, hits, misses, current_size in raw
        )
    )


def clear_oracle_caches() -> None:
    clear_dealer_caches()
    clear_player_caches()
    clear_round_return_caches()
