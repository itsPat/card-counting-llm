from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from blackjack.analysis import BetAction
from blackjack.dataset import StructureToken
from blackjack.engine import (
    Card,
    DecisionType,
    InsuranceAction,
    ModelContext,
    PlayerAction,
    Rank,
    Shoe,
)
from blackjack.training.bankroll import (
    EvaluationPolicyName,
    EvaluationShoe,
    HiLoRuntimePolicy,
    fit_runtime_context,
    generate_evaluation_shoes,
    load_evaluation_shoes,
    simulate_policy,
)


def test_context_fitting_is_a_noop_within_model_window() -> None:
    tokens = (
        StructureToken.HISTORY.value,
        "2",
        "3",
        StructureToken.BET_QUERY.value,
    )

    fitted, dropped = fit_runtime_context(tokens, maximum_length=4)

    assert fitted == tokens
    assert dropped == 0


def test_context_fitting_drops_only_oldest_visible_history() -> None:
    tokens = (
        StructureToken.HISTORY.value,
        "2",
        "3",
        "4",
        StructureToken.CURRENT_HAND.value,
        StructureToken.PLAYER.value,
        "A",
        "10",
        StructureToken.DEALER.value,
        "6",
        StructureToken.PLAY_QUERY.value,
    )

    fitted, dropped = fit_runtime_context(tokens, maximum_length=9)

    assert fitted == (
        StructureToken.HISTORY.value,
        "4",
        StructureToken.CURRENT_HAND.value,
        StructureToken.PLAYER.value,
        "A",
        "10",
        StructureToken.DEALER.value,
        "6",
        StructureToken.PLAY_QUERY.value,
    )
    assert dropped == 2


def test_context_fitting_rejects_a_window_too_short_for_current_state() -> None:
    tokens = (
        StructureToken.HISTORY.value,
        StructureToken.CURRENT_HAND.value,
        StructureToken.PLAYER.value,
        "A",
        "10",
        StructureToken.DEALER.value,
        "6",
        StructureToken.PLAY_QUERY.value,
    )

    with pytest.raises(ValueError, match="current state"):
        fit_runtime_context(tokens, maximum_length=7)


@dataclass(frozen=True, slots=True)
class _StandPolicy:
    @property
    def name(self) -> EvaluationPolicyName:
        return EvaluationPolicyName.HI_LO

    def bet(self, prior_history: tuple[Card, ...]) -> BetAction:
        del prior_history
        return BetAction.MINIMUM

    def insurance(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> InsuranceAction:
        del prior_history, context
        return InsuranceAction.DECLINE

    def play(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> PlayerAction:
        del prior_history, context
        return PlayerAction.STAND


def test_manifest_loader_selects_only_validation_replays(
    tmp_path: Path,
) -> None:
    manifest = {
        "shoes": [
            {
                "shoe_id": 1,
                "split": "train",
                "cards": ["2", "3"],
                "cut_card_position": 2,
            },
            {
                "shoe_id": 2,
                "split": "validation",
                "cards": ["A", "10", "9", "8", "7"],
                "cut_card_position": 5,
            },
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    shoes = load_evaluation_shoes(path)

    assert len(shoes) == 1
    assert shoes[0].shoe_id == 2
    assert shoes[0].replay.burn_card.rank is Rank.ACE


def test_policy_simulation_compounds_exact_engine_returns() -> None:
    shoe = Shoe.arranged(
        (
            Card(Rank.TEN),
            Card(Rank.SIX),
            Card(Rank.KING),
            Card(Rank.TEN),
            Card(Rank.TWO),
        ),
        cut_card_position=5,
    )
    trajectory = simulate_policy(
        (EvaluationShoe(7, shoe.replay),),
        _StandPolicy(),
        initial_bankroll=100,
    )

    assert len(trajectory.rounds) == 1
    record = trajectory.rounds[0]
    assert record.profit_units == 1
    assert record.bet_fraction == pytest.approx(0.001)
    assert record.bankroll_after == pytest.approx(100.1)
    assert record.bankroll_return == pytest.approx(0.001)


def test_hi_lo_runtime_policy_uses_count_deviations_and_insurance() -> None:
    policy = HiLoRuntimePolicy()
    positive_history = (Card(Rank.TWO),) * 36
    context = ModelContext(
        decision_type=DecisionType.INSURANCE,
        history=(),
        current_hand=(Card(Rank.TEN), Card(Rank.NINE)),
        dealer_upcard=Card(Rank.ACE),
        legal_player_actions=(),
        legal_insurance_actions=(
            InsuranceAction.TAKE,
            InsuranceAction.DECLINE,
        ),
    )

    assert policy.bet(positive_history) is BetAction.HIGH
    assert (
        policy.insurance(positive_history, context)
        is InsuranceAction.TAKE
    )


def test_fresh_evaluation_shoes_are_seeded_and_distinct() -> None:
    first = generate_evaluation_shoes(3, seed=91)
    second = generate_evaluation_shoes(3, seed=91)
    different = generate_evaluation_shoes(3, seed=92)

    assert first == second
    assert first != different
    assert len({shoe.replay for shoe in first}) == 3


def test_fresh_evaluation_shoe_ranges_are_nonoverlapping() -> None:
    first = generate_evaluation_shoes(3, seed=91, shoe_start=0)
    second = generate_evaluation_shoes(2, seed=91, shoe_start=3)
    combined = generate_evaluation_shoes(5, seed=91)

    assert (*first, *second) == combined
    assert tuple(shoe.shoe_id for shoe in second) == (3, 4)


def test_simulation_rejects_nonpositive_bankroll() -> None:
    with pytest.raises(ValueError, match="positive"):
        simulate_policy((), _StandPolicy(), initial_bankroll=0)
