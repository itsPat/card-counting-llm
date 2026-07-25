"""Exact finite discrete distributions used by the oracle."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class ReturnOutcome:
    profit: Fraction
    probability: Fraction


@dataclass(frozen=True, slots=True)
class ReturnDistribution:
    outcomes: tuple[ReturnOutcome, ...]

    def __post_init__(self) -> None:
        if any(item.probability <= 0 for item in self.outcomes):
            raise ValueError("return probabilities must be positive")
        if sum(
            (item.probability for item in self.outcomes),
            start=Fraction(0),
        ) != Fraction(1):
            raise ValueError("return probabilities must sum to one")

    @classmethod
    def constant(cls, profit: Fraction | int) -> ReturnDistribution:
        return cls((ReturnOutcome(Fraction(profit), Fraction(1)),))

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[Fraction, Fraction]],
    ) -> ReturnDistribution:
        merged: defaultdict[Fraction, Fraction] = defaultdict(Fraction)
        for profit, probability in pairs:
            if probability < 0:
                raise ValueError("probabilities cannot be negative")
            merged[profit] += probability
        outcomes = tuple(
            ReturnOutcome(profit, probability)
            for profit, probability in sorted(merged.items())
            if probability > 0
        )
        return cls(outcomes)

    @classmethod
    def mixture(
        cls,
        branches: Iterable[tuple[Fraction, ReturnDistribution]],
    ) -> ReturnDistribution:
        return cls.from_pairs(
            (
                outcome.profit,
                branch_probability * outcome.probability,
            )
            for branch_probability, distribution in branches
            for outcome in distribution.outcomes
        )

    @property
    def expected_profit(self) -> Fraction:
        return sum(
            (outcome.profit * outcome.probability for outcome in self.outcomes),
            start=Fraction(0),
        )

    @property
    def minimum_profit(self) -> Fraction:
        return min(item.profit for item in self.outcomes)

    def probability(self, profit: Fraction | int) -> Fraction:
        target = Fraction(profit)
        return next(
            (
                outcome.probability
                for outcome in self.outcomes
                if outcome.profit == target
            ),
            Fraction(0),
        )

    def shifted(self, amount: Fraction | int) -> ReturnDistribution:
        shift = Fraction(amount)
        return ReturnDistribution(
            tuple(
                ReturnOutcome(outcome.profit + shift, outcome.probability)
                for outcome in self.outcomes
            )
        )
