"""Continuous Kelly sizing from a complete exact return distribution."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from blackjack.oracle.distributions import ReturnDistribution


@dataclass(frozen=True, slots=True)
class KellyRecommendation:
    full_kelly: float
    half_kelly: float
    expected_log_growth: float


def expected_log_growth(
    distribution: ReturnDistribution,
    bankroll_fraction: float,
) -> float:
    if bankroll_fraction < 0:
        raise ValueError("bankroll fraction cannot be negative")
    growth = 0.0
    for outcome in distribution.outcomes:
        wealth_multiplier = 1.0 + bankroll_fraction * float(outcome.profit)
        if wealth_multiplier <= 0:
            return float("-inf")
        growth += float(outcome.probability) * log(wealth_multiplier)
    return growth


def _growth_derivative(
    distribution: ReturnDistribution,
    bankroll_fraction: float,
) -> float:
    return sum(
        float(outcome.probability)
        * float(outcome.profit)
        / (1.0 + bankroll_fraction * float(outcome.profit))
        for outcome in distribution.outcomes
    )


def kelly_recommendation(
    distribution: ReturnDistribution,
    *,
    maximum_fraction: float = 1.0,
    iterations: int = 100,
) -> KellyRecommendation:
    if not 0 < maximum_fraction <= 1:
        raise ValueError("maximum fraction must lie in (0, 1]")
    if iterations <= 0:
        raise ValueError("iteration count must be positive")
    if distribution.expected_profit <= 0:
        return KellyRecommendation(0.0, 0.0, 0.0)

    lower = 0.0
    upper = maximum_fraction
    if distribution.minimum_profit < 0:
        solvency_limit = -1.0 / float(distribution.minimum_profit)
        upper = min(upper, solvency_limit * (1.0 - 1e-12))
    if _growth_derivative(distribution, upper) >= 0:
        full_kelly = upper
    else:
        for _ in range(iterations):
            midpoint = (lower + upper) / 2
            if _growth_derivative(distribution, midpoint) > 0:
                lower = midpoint
            else:
                upper = midpoint
        full_kelly = (lower + upper) / 2
    return KellyRecommendation(
        full_kelly=full_kelly,
        half_kelly=full_kelly / 2,
        expected_log_growth=expected_log_growth(distribution, full_kelly),
    )
