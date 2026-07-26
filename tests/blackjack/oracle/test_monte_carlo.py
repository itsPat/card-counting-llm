from __future__ import annotations

from fractions import Fraction

import pytest

from blackjack.analysis import empirical_round_return_distribution
from blackjack.engine import PlayerAction
from blackjack.oracle import (
    CardValue,
    Composition,
    OracleHand,
    PeekCondition,
    ResolvedHand,
    RoundPlayerSituation,
    evaluate_round_actions,
    fixed_policy_play_action_estimates,
    fixed_policy_play_rollout_seed,
    fixed_policy_rollout_seed,
    fixed_policy_round_return_estimate,
)


def test_fixed_policy_rollouts_are_exactly_replayable() -> None:
    composition = Composition.full_shoe()
    first = fixed_policy_round_return_estimate(
        composition,
        unseen_unavailable=1,
        seed=917,
        rollouts=25_000,
    )
    second = fixed_policy_round_return_estimate(
        composition,
        unseen_unavailable=1,
        seed=917,
        rollouts=25_000,
    )
    assert first == second
    assert sum(
        outcome.probability for outcome in first.distribution.outcomes
    ) == 1


def test_different_rollout_seeds_produce_different_empirical_distributions() -> None:
    composition = Composition.full_shoe()
    first = fixed_policy_round_return_estimate(
        composition,
        unseen_unavailable=1,
        seed=10,
        rollouts=10_000,
    )
    second = fixed_policy_round_return_estimate(
        composition,
        unseen_unavailable=1,
        seed=11,
        rollouts=10_000,
    )
    assert first.distribution != second.distribution


def test_rollout_seed_depends_on_every_public_simulation_input() -> None:
    full = Composition.full_shoe()
    depleted = full.remove(CardValue.TEN)
    seed = fixed_policy_rollout_seed(full, 1, 20250728)
    assert seed == fixed_policy_rollout_seed(full, 1, 20250728)
    assert seed != fixed_policy_rollout_seed(depleted, 1, 20250728)
    assert seed != fixed_policy_rollout_seed(full, 2, 20250728)
    assert seed != fixed_policy_rollout_seed(full, 1, 20250729)


def test_unknown_unavailable_cards_are_removed_before_each_round() -> None:
    composition = Composition.full_shoe()
    available = fixed_policy_round_return_estimate(
        composition,
        unseen_unavailable=0,
        seed=81,
        rollouts=20_000,
    )
    unavailable = fixed_policy_round_return_estimate(
        composition,
        unseen_unavailable=3,
        seed=81,
        rollouts=20_000,
    )
    assert available.distribution != unavailable.distribution


def test_native_policy_agrees_with_the_python_reference_simulator() -> None:
    composition = Composition.full_shoe()
    reference = empirical_round_return_distribution(
        composition,
        seed=123,
        rollouts=100_000,
    )
    native = fixed_policy_round_return_estimate(
        composition,
        unseen_unavailable=0,
        seed=123,
        rollouts=100_000,
    )
    assert float(native.distribution.expected_profit) == pytest.approx(
        reference.expected_profit,
        abs=0.02,
    )
    for profit in (
        Fraction(-2),
        Fraction(-1),
        Fraction(0),
        Fraction(1),
        Fraction(3, 2),
        Fraction(2),
    ):
        assert float(native.distribution.probability(profit)) == pytest.approx(
            reference.probability(float(profit)),
            abs=0.015,
        )


def test_all_ten_rounds_are_deterministic_pushes() -> None:
    estimate = fixed_policy_round_return_estimate(
        Composition.from_values((CardValue.TEN,) * 24),
        unseen_unavailable=2,
        seed=7,
        rollouts=1_000,
    )
    assert estimate.distribution.probability(0) == 1
    assert estimate.expected_profit_standard_error == 0
    assert estimate.expected_profit_confidence_interval_95 == (0, 0)


@pytest.mark.parametrize(
    ("unseen_unavailable", "seed", "rollouts"),
    ((-1, 1, 1), (0, -1, 1), (0, 2**64, 1), (0, 1, 0)),
)
def test_invalid_simulation_inputs_are_rejected(
    unseen_unavailable: int,
    seed: int,
    rollouts: int,
) -> None:
    with pytest.raises(ValueError):
        fixed_policy_round_return_estimate(
            Composition.full_shoe(),
            unseen_unavailable=unseen_unavailable,
            seed=seed,
            rollouts=rollouts,
        )


def test_play_action_rollouts_are_replayable_and_close_to_exact_values() -> None:
    composition = Composition.full_shoe()
    for card in (CardValue.TEN, CardValue.SIX, CardValue.TEN):
        composition = composition.remove(card)
    situation = RoundPlayerSituation(
        composition=composition,
        active_hand=OracleHand((CardValue.TEN, CardValue.SIX)),
        pending_hands=(),
        finished_hands=(),
        dealer_upcard=CardValue.TEN,
        peek_condition=PeekCondition.NO_BLACKJACK,
    )
    exact = evaluate_round_actions(situation)
    actions = tuple(evaluation.action for evaluation in exact)
    seed = fixed_policy_play_rollout_seed(situation, 919)
    first = fixed_policy_play_action_estimates(
        situation,
        actions,
        seed=seed,
        rollouts=250_000,
    )
    second = fixed_policy_play_action_estimates(
        situation,
        actions,
        seed=seed,
        rollouts=250_000,
    )
    assert first == second
    for estimate, reference in zip(first, exact, strict=True):
        assert estimate.action is reference.action
        assert float(estimate.expected_profit) == pytest.approx(
            float(reference.expected_profit),
            abs=0.015,
        )


def test_play_rollouts_retain_finished_pending_and_split_hand_outcomes() -> None:
    composition = Composition.full_shoe()
    for card in (
        CardValue.EIGHT,
        CardValue.EIGHT,
        CardValue.TEN,
        CardValue.SIX,
        CardValue.FIVE,
    ):
        composition = composition.remove(card)
    situation = RoundPlayerSituation(
        composition=composition,
        active_hand=OracleHand(
            (CardValue.EIGHT, CardValue.EIGHT),
            from_split=True,
            can_surrender=False,
        ),
        pending_hands=(
            OracleHand(
                (CardValue.TEN, CardValue.SIX),
                from_split=True,
                can_surrender=False,
            ),
        ),
        finished_hands=(
            ResolvedHand(
                total=20,
                wager=Fraction(1),
                is_natural_blackjack=False,
                is_bust=False,
            ),
        ),
        dealer_upcard=CardValue.FIVE,
        peek_condition=PeekCondition.NONE,
    )
    estimate = fixed_policy_play_action_estimates(
        situation,
        (PlayerAction.SPLIT,),
        seed=41,
        rollouts=20_000,
    )[0]
    assert estimate.action is PlayerAction.SPLIT
    assert estimate.distribution.minimum_profit == -6
    assert estimate.distribution.probability(Fraction(6)) > 0


def test_play_rollout_seed_changes_with_hidden_card_count_and_round_state() -> None:
    situation = RoundPlayerSituation(
        composition=Composition.full_shoe(),
        active_hand=OracleHand((CardValue.FIVE, CardValue.SIX)),
        pending_hands=(),
        finished_hands=(),
        dealer_upcard=CardValue.SIX,
        peek_condition=PeekCondition.NONE,
    )
    with_unavailable = RoundPlayerSituation(
        composition=situation.composition,
        active_hand=situation.active_hand,
        pending_hands=(),
        finished_hands=(),
        dealer_upcard=situation.dealer_upcard,
        peek_condition=situation.peek_condition,
        unseen_unavailable=2,
    )
    seed = fixed_policy_play_rollout_seed(situation, 2025)
    assert seed == fixed_policy_play_rollout_seed(situation, 2025)
    assert seed != fixed_policy_play_rollout_seed(with_unavailable, 2025)
    assert seed != fixed_policy_play_rollout_seed(situation, 2026)


def test_play_rollouts_reject_an_illegal_first_action() -> None:
    situation = RoundPlayerSituation(
        composition=Composition.full_shoe(),
        active_hand=OracleHand((CardValue.FIVE, CardValue.SIX)),
        pending_hands=(),
        finished_hands=(),
        dealer_upcard=CardValue.SIX,
        peek_condition=PeekCondition.NONE,
    )
    with pytest.raises(ValueError, match="illegal action"):
        fixed_policy_play_action_estimates(
            situation,
            (PlayerAction.SPLIT,),
            seed=1,
            rollouts=10,
        )
