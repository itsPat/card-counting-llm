"""Reproducible Monte Carlo pilot for selecting discrete bet tokens."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from math import floor, isclose, log, sqrt
from random import Random

from blackjack.engine.actions import PlayerAction
from blackjack.engine.rules import FIXED_RULES, CasinoRules
from blackjack.engine.shoe import Shoe
from blackjack.oracle.composition import CARD_VALUES, CardValue, Composition
from blackjack.oracle.kelly import KellyRecommendation
from blackjack.oracle.player import oracle_hand_value

_PROBABILITY_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class EmpiricalReturnOutcome:
    profit: float
    probability: float


@dataclass(frozen=True, slots=True)
class EmpiricalReturnDistribution:
    """A complete finite return distribution estimated from seeded rollouts."""

    outcomes: tuple[EmpiricalReturnOutcome, ...]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("a return distribution needs at least one outcome")
        if any(outcome.probability <= 0 for outcome in self.outcomes):
            raise ValueError("return probabilities must be positive")
        total = sum(outcome.probability for outcome in self.outcomes)
        if not isclose(total, 1.0, abs_tol=_PROBABILITY_TOLERANCE):
            raise ValueError("return probabilities must sum to one")

    @property
    def expected_profit(self) -> float:
        return sum(outcome.profit * outcome.probability for outcome in self.outcomes)

    @property
    def minimum_profit(self) -> float:
        return min(outcome.profit for outcome in self.outcomes)

    @property
    def variance(self) -> float:
        mean = self.expected_profit
        return sum(
            outcome.probability * (outcome.profit - mean) ** 2
            for outcome in self.outcomes
        )

    def probability(self, profit: float) -> float:
        return next(
            (
                outcome.probability
                for outcome in self.outcomes
                if outcome.profit == profit
            ),
            0.0,
        )


@dataclass(frozen=True, slots=True)
class PilotConfiguration:
    """Every input needed to reproduce the pilot sample and rollouts."""

    seed: int = 20250725
    shoe_count: int = 24
    samples_per_shoe: int = 4
    rollouts_per_composition: int = 25_000

    def __post_init__(self) -> None:
        if self.shoe_count <= 0:
            raise ValueError("shoe count must be positive")
        if self.samples_per_shoe <= 0:
            raise ValueError("samples per shoe must be positive")
        if self.rollouts_per_composition <= 0:
            raise ValueError("rollout count must be positive")


@dataclass(frozen=True, slots=True)
class SampledComposition:
    sample_index: int
    shoe_seed: int
    visible_cards: int
    penetration: float
    unseen_unavailable: int
    composition: Composition


@dataclass(frozen=True, slots=True)
class PilotObservation:
    sample: SampledComposition
    distribution: EmpiricalReturnDistribution
    expected_profit: float
    expected_profit_standard_error: float
    half_kelly: float


class BetAction(StrEnum):
    MINIMUM = "<BET_MIN>"
    LOW = "<BET_LOW>"
    MEDIUM = "<BET_MEDIUM>"
    HIGH = "<BET_HIGH>"


@dataclass(frozen=True, slots=True)
class BetToken:
    token: BetAction
    bankroll_fraction: float


@dataclass(frozen=True, slots=True)
class BetVocabulary:
    name: str
    fractions: tuple[float, ...]
    actions: tuple[BetAction, ...] = ()

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValueError("a bet vocabulary needs at least one fraction")
        if any(fraction <= 0 or fraction > 1 for fraction in self.fractions):
            raise ValueError("bet fractions must lie in (0, 1]")
        if self.fractions != tuple(sorted(set(self.fractions))):
            raise ValueError("bet fractions must be unique and increasing")
        if self.actions and len(self.actions) != len(self.fractions):
            raise ValueError("bet actions and fractions must have equal lengths")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("bet actions must be unique")

    @property
    def tokens(self) -> tuple[BetToken, ...]:
        if not self.actions:
            raise ValueError("candidate vocabulary has no serialized bet actions")
        return tuple(
            BetToken(
                token=action,
                bankroll_fraction=fraction,
            )
            for action, fraction in zip(
                self.actions,
                self.fractions,
                strict=True,
            )
        )

    def nearest_fraction(self, target: float) -> float:
        return min(
            self.fractions,
            key=lambda fraction: (abs(fraction - target), fraction),
        )


SELECTED_BET_VOCABULARY = BetVocabulary(
    "selected-four-token",
    (0.001, 0.005, 0.009, 0.013),
    (
        BetAction.MINIMUM,
        BetAction.LOW,
        BetAction.MEDIUM,
        BetAction.HIGH,
    ),
)


@dataclass(frozen=True, slots=True)
class PilotMetrics:
    vocabulary: BetVocabulary
    mean_absolute_rounding_error: float
    p95_absolute_rounding_error: float
    mean_log_growth_regret: float
    p95_absolute_log_growth_change: float
    class_counts: tuple[int, ...]

    @property
    def occupied_classes(self) -> int:
        return sum(count > 0 for count in self.class_counts)


@dataclass(slots=True)
class _SimulatedHand:
    cards: list[CardValue]
    wager_half_units: int = 2
    from_split: bool = False
    split_aces: bool = False
    surrendered: bool = False
    finished: bool = False

    @property
    def total(self) -> int:
        return oracle_hand_value(tuple(self.cards)).total

    @property
    def is_soft(self) -> bool:
        return oracle_hand_value(tuple(self.cards)).is_soft

    @property
    def is_bust(self) -> bool:
        return oracle_hand_value(tuple(self.cards)).is_bust

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0] is self.cards[1]

    @property
    def is_natural_blackjack(self) -> bool:
        return not self.from_split and len(self.cards) == 2 and self.total == 21


class _MutableComposition:
    __slots__ = ("counts", "total")

    def __init__(self, composition: Composition) -> None:
        self.counts = list(composition.counts)
        self.total = composition.total

    def draw(self, rng: Random) -> CardValue:
        if self.total == 0:
            raise ValueError("simulation exhausted the composition")
        selected = rng.randrange(self.total)
        cumulative = 0
        for index, count in enumerate(self.counts):
            cumulative += count
            if selected < cumulative:
                self.counts[index] -= 1
                self.total -= 1
                return CARD_VALUES[index]
        raise AssertionError("draw index must select a card")

    def count(self, value: CardValue) -> int:
        return self.counts[CARD_VALUES.index(value)]


def sample_representative_compositions(
    configuration: PilotConfiguration,
) -> tuple[SampledComposition, ...]:
    """Stratify public-history samples across randomized six-deck shoes."""

    samples: list[SampledComposition] = []
    sample_index = 0
    for shoe_offset in range(configuration.shoe_count):
        shoe_seed = configuration.seed + shoe_offset
        shoe = Shoe.shuffled(shoe_seed)
        maximum_visible = max(0, shoe.replay.cut_card_position - 5)
        jitter = Random(configuration.seed * 1_000_003 + shoe_seed)
        for stratum in range(configuration.samples_per_shoe):
            lower = floor(maximum_visible * stratum / configuration.samples_per_shoe)
            upper = floor(
                maximum_visible * (stratum + 1) / configuration.samples_per_shoe
            )
            visible_cards = jitter.randint(lower, max(lower, upper))
            visible_history = shoe.replay.deal_order[:visible_cards]
            composition = Composition.full_shoe().remove_cards(visible_history)
            samples.append(
                SampledComposition(
                    sample_index=sample_index,
                    shoe_seed=shoe_seed,
                    visible_cards=visible_cards,
                    penetration=visible_cards / Composition.full_shoe().total,
                    unseen_unavailable=1,
                    composition=composition,
                )
            )
            sample_index += 1
    return tuple(samples)


def _dealer_value(value: CardValue) -> int:
    return 11 if value is CardValue.ACE else value.hard_value


def _legal_actions(
    hand: _SimulatedHand,
    hands_in_round: int,
    rules: CasinoRules,
) -> tuple[PlayerAction, ...]:
    if hand.is_bust or hand.total >= 21 or hand.split_aces:
        return ()
    actions = [PlayerAction.HIT, PlayerAction.STAND]
    if len(hand.cards) == 2 and (not hand.from_split or rules.double_after_split):
        actions.append(PlayerAction.DOUBLE)
    if (
        hand.is_pair
        and hands_in_round < rules.maximum_player_hands
        and not (
            hand.from_split
            and hand.cards[0] is CardValue.ACE
            and not rules.resplit_aces
        )
    ):
        actions.append(PlayerAction.SPLIT)
    if hand.from_split is False and len(hand.cards) == 2 and rules.late_surrender:
        actions.append(PlayerAction.SURRENDER)
    return tuple(actions)


def _basic_strategy_action(
    hand: _SimulatedHand,
    dealer_upcard: CardValue,
    legal: tuple[PlayerAction, ...],
) -> PlayerAction:
    """Six-deck H17, DAS, late-surrender baseline used only by the pilot.

    Reference chart:
    https://www.blackjackapprenticeship.com/wp-content/uploads/2024/09/H17-Basic-Strategy.pdf
    """

    dealer = _dealer_value(dealer_upcard)
    if PlayerAction.SURRENDER in legal and not hand.is_soft:
        surrender = (
            (hand.total == 17 and dealer == 11)
            or (hand.total == 16 and dealer in (9, 10, 11))
            or (hand.total == 15 and dealer in (10, 11))
        )
        if surrender:
            return PlayerAction.SURRENDER

    if PlayerAction.SPLIT in legal:
        pair = hand.cards[0]
        should_split = (
            pair in (CardValue.ACE, CardValue.EIGHT)
            or (pair is CardValue.NINE and dealer in (2, 3, 4, 5, 6, 8, 9))
            or (pair is CardValue.SEVEN and dealer in (2, 3, 4, 5, 6, 7))
            or (pair is CardValue.SIX and dealer in (2, 3, 4, 5, 6))
            or (pair is CardValue.FOUR and dealer in (5, 6))
            or (
                pair in (CardValue.TWO, CardValue.THREE)
                and dealer in (2, 3, 4, 5, 6, 7)
            )
        )
        if should_split:
            return PlayerAction.SPLIT

    can_double = PlayerAction.DOUBLE in legal
    if hand.is_soft:
        if hand.total >= 20:
            return PlayerAction.STAND
        if hand.total == 19:
            return (
                PlayerAction.DOUBLE
                if can_double and dealer == 6
                else PlayerAction.STAND
            )
        if hand.total == 18:
            if can_double and dealer in (2, 3, 4, 5, 6):
                return PlayerAction.DOUBLE
            return (
                PlayerAction.STAND
                if dealer in (2, 3, 4, 5, 6, 7, 8)
                else PlayerAction.HIT
            )
        double_ranges = {
            17: (3, 4, 5, 6),
            16: (4, 5, 6),
            15: (4, 5, 6),
            14: (5, 6),
            13: (5, 6),
        }
        if can_double and dealer in double_ranges.get(hand.total, ()):
            return PlayerAction.DOUBLE
        return PlayerAction.HIT

    if hand.total >= 17:
        return PlayerAction.STAND
    if 13 <= hand.total <= 16:
        return PlayerAction.STAND if dealer in (2, 3, 4, 5, 6) else PlayerAction.HIT
    if hand.total == 12:
        return PlayerAction.STAND if dealer in (4, 5, 6) else PlayerAction.HIT
    if can_double and hand.total == 11:
        return PlayerAction.DOUBLE
    if can_double and hand.total == 10 and dealer in range(2, 10):
        return PlayerAction.DOUBLE
    if can_double and hand.total == 9 and dealer in (3, 4, 5, 6):
        return PlayerAction.DOUBLE
    return PlayerAction.HIT


def _dealer_should_hit(cards: list[CardValue], rules: CasinoRules) -> bool:
    value = oracle_hand_value(tuple(cards))
    return value.total < 17 or (
        value.total == 17 and value.is_soft and rules.dealer_hits_soft_17
    )


def _profit_against_dealer(
    hand: _SimulatedHand,
    dealer_cards: list[CardValue],
) -> int:
    if hand.surrendered:
        return -hand.wager_half_units // 2
    if hand.is_bust:
        return -hand.wager_half_units
    dealer = oracle_hand_value(tuple(dealer_cards))
    if dealer.is_bust or hand.total > dealer.total:
        return hand.wager_half_units
    if hand.total == dealer.total:
        return 0
    return -hand.wager_half_units


def _simulate_round(
    composition: Composition,
    rng: Random,
    rules: CasinoRules,
) -> int:
    shoe = _MutableComposition(composition)
    first_player = shoe.draw(rng)
    dealer_upcard = shoe.draw(rng)
    second_player = shoe.draw(rng)
    public_hole_pool = shoe.total
    take_insurance = (
        dealer_upcard is CardValue.ACE
        and shoe.count(CardValue.TEN) / public_hole_pool > 1 / 3
    )
    dealer_hole = shoe.draw(rng)
    dealer_cards = [dealer_upcard, dealer_hole]
    dealer_blackjack = oracle_hand_value(tuple(dealer_cards)).total == 21
    player = _SimulatedHand([first_player, second_player])

    insurance_profit = 2 if take_insurance and dealer_blackjack else 0
    if take_insurance and not dealer_blackjack:
        insurance_profit = -1
    if dealer_blackjack:
        player_profit = 0 if player.is_natural_blackjack else -2
        return player_profit + insurance_profit
    if player.is_natural_blackjack:
        return 3 + insurance_profit

    hands = [player]
    active_index = 0
    while active_index < len(hands):
        hand = hands[active_index]
        if hand.finished or hand.is_bust or hand.total >= 21 or hand.split_aces:
            hand.finished = True
            active_index += 1
            continue
        legal = _legal_actions(hand, len(hands), rules)
        action = _basic_strategy_action(hand, dealer_upcard, legal)
        if action is PlayerAction.HIT:
            hand.cards.append(shoe.draw(rng))
        elif action is PlayerAction.STAND:
            hand.finished = True
        elif action is PlayerAction.DOUBLE:
            hand.wager_half_units *= 2
            hand.cards.append(shoe.draw(rng))
            hand.finished = True
        elif action is PlayerAction.SURRENDER:
            hand.surrendered = True
            hand.finished = True
        else:
            pair = hand.cards[0]
            split_aces = pair is CardValue.ACE
            left = _SimulatedHand(
                [pair, shoe.draw(rng)],
                wager_half_units=hand.wager_half_units,
                from_split=True,
                split_aces=split_aces,
            )
            right = _SimulatedHand(
                [pair, shoe.draw(rng)],
                wager_half_units=hand.wager_half_units,
                from_split=True,
                split_aces=split_aces,
            )
            if split_aces and rules.split_aces_one_card_only:
                left.finished = True
                right.finished = True
            hands[active_index : active_index + 1] = [left, right]

    if all(hand.surrendered or hand.is_bust for hand in hands):
        return (
            sum(
                -hand.wager_half_units // 2
                if hand.surrendered
                else -hand.wager_half_units
                for hand in hands
            )
            + insurance_profit
        )

    while _dealer_should_hit(dealer_cards, rules):
        dealer_cards.append(shoe.draw(rng))
    return (
        sum(_profit_against_dealer(hand, dealer_cards) for hand in hands)
        + insurance_profit
    )


def empirical_round_return_distribution(
    composition: Composition,
    *,
    seed: int,
    rollouts: int,
    rules: CasinoRules = FIXED_RULES,
) -> EmpiricalReturnDistribution:
    """Estimate the complete return distribution under the pilot policy."""

    if rollouts <= 0:
        raise ValueError("rollout count must be positive")
    if composition.total <= 4:
        raise ValueError("composition is too small to simulate a round")
    rng = Random(seed)
    outcomes: Counter[int] = Counter(
        _simulate_round(composition, rng, rules) for _ in range(rollouts)
    )
    return EmpiricalReturnDistribution(
        tuple(
            EmpiricalReturnOutcome(
                profit=profit_half_units / 2,
                probability=count / rollouts,
            )
            for profit_half_units, count in sorted(outcomes.items())
        )
    )


def run_bet_token_pilot(
    configuration: PilotConfiguration,
) -> tuple[PilotObservation, ...]:
    samples = sample_representative_compositions(configuration)
    observations: list[PilotObservation] = []
    for sample in samples:
        rollout_seed = configuration.seed * 1_000_000_007 + sample.sample_index * 97_409
        distribution = empirical_round_return_distribution(
            sample.composition,
            seed=rollout_seed,
            rollouts=configuration.rollouts_per_composition,
        )
        half_kelly = _empirical_kelly_recommendation(distribution).half_kelly
        observations.append(
            PilotObservation(
                sample=sample,
                distribution=distribution,
                expected_profit=distribution.expected_profit,
                expected_profit_standard_error=sqrt(
                    distribution.variance / configuration.rollouts_per_composition
                ),
                half_kelly=half_kelly,
            )
        )
    return tuple(observations)


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, floor(percentile * len(ordered)))
    return ordered[index]


def _log_growth(
    distribution: EmpiricalReturnDistribution,
    fraction: float,
) -> float:
    return sum(
        outcome.probability * log(1 + fraction * outcome.profit)
        for outcome in distribution.outcomes
    )


def _empirical_kelly_recommendation(
    distribution: EmpiricalReturnDistribution,
    *,
    maximum_fraction: float = 1.0,
    iterations: int = 100,
) -> KellyRecommendation:
    if not 0 <= maximum_fraction <= 1:
        raise ValueError("maximum fraction must lie in [0, 1]")
    if iterations <= 0:
        raise ValueError("iteration count must be positive")
    if distribution.expected_profit <= 0:
        return KellyRecommendation(0.0, 0.0, 0.0)

    def derivative(fraction: float) -> float:
        return sum(
            outcome.probability * outcome.profit / (1 + fraction * outcome.profit)
            for outcome in distribution.outcomes
        )

    lower = 0.0
    upper = maximum_fraction
    if distribution.minimum_profit < 0:
        solvency_limit = -1 / distribution.minimum_profit
        upper = min(upper, solvency_limit * (1 - 1e-12))
    if derivative(upper) >= 0:
        full_kelly = upper
    else:
        for _ in range(iterations):
            midpoint = (lower + upper) / 2
            if derivative(midpoint) > 0:
                lower = midpoint
            else:
                upper = midpoint
        full_kelly = (lower + upper) / 2
    return KellyRecommendation(
        full_kelly=full_kelly,
        half_kelly=full_kelly / 2,
        expected_log_growth=_log_growth(distribution, full_kelly),
    )


def analyze_vocabulary(
    observations: tuple[PilotObservation, ...],
    vocabulary: BetVocabulary,
) -> PilotMetrics:
    if not observations:
        raise ValueError("vocabulary analysis needs pilot observations")
    counts = [0] * len(vocabulary.fractions)
    rounding_errors: list[float] = []
    regrets: list[float] = []
    absolute_growth_changes: list[float] = []
    for observation in observations:
        rounded = vocabulary.nearest_fraction(observation.half_kelly)
        class_index = vocabulary.fractions.index(rounded)
        counts[class_index] += 1
        rounding_errors.append(abs(rounded - observation.half_kelly))
        continuous_growth = _log_growth(
            observation.distribution,
            observation.half_kelly,
        )
        rounded_growth = _log_growth(observation.distribution, rounded)
        growth_change = continuous_growth - rounded_growth
        regrets.append(max(0.0, growth_change))
        absolute_growth_changes.append(abs(growth_change))
    return PilotMetrics(
        vocabulary=vocabulary,
        mean_absolute_rounding_error=sum(rounding_errors) / len(rounding_errors),
        p95_absolute_rounding_error=_percentile(tuple(rounding_errors), 0.95),
        mean_log_growth_regret=sum(regrets) / len(regrets),
        p95_absolute_log_growth_change=_percentile(
            tuple(absolute_growth_changes),
            0.95,
        ),
        class_counts=tuple(counts),
    )


def _linear_vocabulary(
    name: str,
    minimum: float,
    maximum: float,
    spacing: float,
) -> BetVocabulary:
    steps = floor((maximum - minimum) / spacing)
    fractions = tuple(
        round(minimum + index * spacing, 10) for index in range(steps + 1)
    )
    if fractions[-1] < maximum:
        fractions = (*fractions, maximum)
    return BetVocabulary(name, fractions)


def candidate_vocabularies() -> tuple[BetVocabulary, ...]:
    """Candidate floors, caps, and linear spacings for the pilot comparison."""

    return (
        SELECTED_BET_VOCABULARY,
        BetVocabulary("five-token", (0.001, 0.003, 0.006, 0.009, 0.013)),
        BetVocabulary(
            "six-token",
            (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.013),
        ),
        _linear_vocabulary("compact", 0.001, 0.02, 0.005),
        _linear_vocabulary("balanced", 0.001, 0.025, 0.0025),
        _linear_vocabulary("fine", 0.001, 0.03, 0.001),
        _linear_vocabulary("higher-floor", 0.0025, 0.025, 0.0025),
        _linear_vocabulary("wider-cap", 0.001, 0.04, 0.005),
        BetVocabulary(
            "hybrid",
            (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03),
        ),
    )
