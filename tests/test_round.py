from __future__ import annotations

from fractions import Fraction

import pytest

from blackjack import (
    BlackjackRound,
    EventType,
    IllegalActionError,
    InsuranceAction,
    InvalidPhaseError,
    PlayerAction,
    Rank,
    RoundPhase,
    Shoe,
    cards,
)
from blackjack.settlement import HandOutcome, InsuranceOutcome


def arranged_round(
    *deal_order: str,
    wager: int | Fraction = 10,
) -> BlackjackRound:
    return BlackjackRound(Shoe.arranged(cards(*deal_order)), Fraction(wager))


def test_initial_deal_order_and_hole_card_redaction() -> None:
    game = arranged_round("5", "9", "6", "7")
    public = game.public_state
    internal = game.internal_state
    assert public.player_hands[0].cards == cards("5", "6")
    assert public.dealer_cards == cards("9")
    assert internal.dealer_cards == cards("9", "7")
    assert public.visible_card_history == cards("5", "9", "6")
    assert cards("7")[0] not in public.visible_card_history
    assert all(
        event.event_type is not EventType.DEALER_HOLE_CARD_DEALT
        for event in public.events
    )


def test_ten_upcard_peek_reveals_natural_and_settles() -> None:
    game = arranged_round("9", "K", "7", "A")
    assert game.public_state.phase is RoundPhase.SETTLED
    assert game.public_state.dealer_cards == cards("K", "A")
    assert game.public_state.settlement is not None
    assert game.public_state.settlement.hands[0].outcome is HandOutcome.LOSS


def test_negative_ten_upcard_peek_continues_without_reveal() -> None:
    game = arranged_round("9", "K", "7", "8")
    assert game.public_state.phase is RoundPhase.PLAYER_ACTIONS
    assert game.public_state.dealer_cards == cards("K")
    assert game.internal_state.dealer_cards == cards("K", "8")


def test_ace_upcard_offers_insurance_before_peek() -> None:
    game = arranged_round("10", "A", "7", "K")
    assert game.public_state.phase is RoundPhase.INSURANCE
    assert game.public_state.insurance_available
    assert game.public_state.dealer_cards == cards("A")
    game.decide_insurance(InsuranceAction.DECLINE)
    assert game.public_state.phase is RoundPhase.SETTLED
    assert game.public_state.dealer_hole_revealed


def test_insurance_loss_is_settled_separately() -> None:
    game = arranged_round("10", "A", "8", "9")
    game.decide_insurance(InsuranceAction.TAKE)
    game.act(PlayerAction.STAND)
    settlement = game.public_state.settlement
    assert settlement is not None and settlement.insurance is not None
    assert settlement.insurance.outcome is InsuranceOutcome.LOST
    assert settlement.insurance.profit == Fraction(-5)
    assert settlement.total_profit == Fraction(-15)


def test_player_natural_with_insurance_reproduces_even_money() -> None:
    game = arranged_round("A", "A", "K", "10")
    game.decide_insurance(InsuranceAction.TAKE)
    settlement = game.public_state.settlement
    assert settlement is not None and settlement.insurance is not None
    assert settlement.hands[0].outcome is HandOutcome.PUSH
    assert settlement.insurance.profit == Fraction(10)
    assert settlement.total_profit == Fraction(10)


def test_hit_then_automatic_twenty_one_and_dealer_bust() -> None:
    game = arranged_round("10", "9", "7", "7", "4", "10")
    assert PlayerAction.HIT in game.legal_actions
    game.act(PlayerAction.HIT)
    settlement = game.public_state.settlement
    assert settlement is not None
    assert game.public_state.player_hands[0].value.total == 21
    assert settlement.hands[0].outcome is HandOutcome.WIN


def test_stand_and_push() -> None:
    game = arranged_round("10", "10", "8", "8")
    game.act(PlayerAction.STAND)
    assert game.public_state.settlement is not None
    assert game.public_state.settlement.hands[0].outcome is HandOutcome.PUSH


def test_double_draws_once_and_doubles_wager() -> None:
    game = arranged_round("5", "6", "6", "10", "10", "10")
    game.act(PlayerAction.DOUBLE)
    hand = game.public_state.player_hands[0]
    assert hand.cards == cards("5", "6", "10")
    assert hand.wager == Fraction(20)
    assert hand.doubled
    assert game.public_state.settlement is not None
    assert game.public_state.settlement.hands[0].profit == Fraction(20)


def test_late_surrender_is_only_initial_first_action() -> None:
    game = arranged_round("10", "9", "6", "8", "2")
    assert PlayerAction.SURRENDER in game.legal_actions
    game.act(PlayerAction.HIT)
    assert PlayerAction.SURRENDER not in game.legal_actions

    surrendered = arranged_round("10", "9", "6", "8")
    surrendered.act(PlayerAction.SURRENDER)
    settlement = surrendered.public_state.settlement
    assert settlement is not None
    assert settlement.hands[0].outcome is HandOutcome.SURRENDER
    assert settlement.hands[0].profit == Fraction(-5)


def test_double_after_split() -> None:
    game = arranged_round("8", "6", "8", "10", "3", "2", "10", "10", "10")
    game.act(PlayerAction.SPLIT)
    assert PlayerAction.DOUBLE in game.legal_actions
    game.act(PlayerAction.DOUBLE)
    assert game.public_state.player_hands[0].wager == Fraction(20)


def test_split_limit_is_four_hands() -> None:
    game = arranged_round(
        "8",
        "6",
        "8",
        "10",
        "8",
        "8",
        "2",
        "3",
        "4",
        "5",
        "10",
    )
    game.act(PlayerAction.SPLIT)
    game.act(PlayerAction.SPLIT)
    game.act(PlayerAction.STAND)
    game.act(PlayerAction.STAND)
    game.act(PlayerAction.SPLIT)
    assert len(game.public_state.player_hands) == 4
    assert PlayerAction.SPLIT not in game.legal_actions


def test_split_aces_get_one_card_each_and_cannot_be_resplit() -> None:
    game = arranged_round("A", "6", "A", "10", "A", "K", "10")
    game.act(PlayerAction.SPLIT)
    assert game.public_state.phase is RoundPhase.SETTLED
    assert len(game.public_state.player_hands) == 2
    assert all(len(hand.cards) == 2 for hand in game.public_state.player_hands)
    assert all(
        hand.split_aces and hand.finished for hand in game.public_state.player_hands
    )
    assert all(
        hand.outcome is HandOutcome.WIN for hand in game.public_state.player_hands
    )
    assert all(hand.profit == Fraction(10) for hand in game.public_state.player_hands)


def test_mixed_ten_valued_cards_can_split() -> None:
    game = arranged_round("10", "6", "K", "10", "2", "3")
    assert PlayerAction.SPLIT in game.legal_actions
    game.act(PlayerAction.SPLIT)
    assert len(game.public_state.player_hands) == 2


def test_dealer_stands_on_hard_seventeen() -> None:
    game = arranged_round("10", "10", "7", "7")
    game.act(PlayerAction.STAND)
    assert game.public_state.dealer_cards == cards("10", "7")


def test_dealer_hits_soft_seventeen() -> None:
    game = arranged_round("10", "A", "8", "6", "10")
    game.decide_insurance(InsuranceAction.DECLINE)
    game.act(PlayerAction.STAND)
    assert game.public_state.dealer_cards == cards("A", "6", "10")


def test_dealer_draws_through_multiple_soft_totals() -> None:
    game = arranged_round("10", "A", "8", "2", "A", "3", "10")
    game.decide_insurance(InsuranceAction.DECLINE)
    game.act(PlayerAction.STAND)
    assert game.public_state.dealer_cards == cards("A", "2", "A", "3", "10")
    assert game.public_state.dealer_hole_revealed


def test_dealer_bust_settles_live_hands_as_wins() -> None:
    game = arranged_round("10", "9", "8", "6", "10")
    game.act(PlayerAction.STAND)
    settlement = game.public_state.settlement
    assert settlement is not None
    assert settlement.hands[0].outcome is HandOutcome.WIN


def test_natural_pays_three_to_two() -> None:
    game = arranged_round("A", "6", "K", "10")
    settlement = game.public_state.settlement
    assert settlement is not None
    assert settlement.hands[0].outcome is HandOutcome.BLACKJACK
    assert settlement.hands[0].profit == Fraction(15)


@pytest.mark.parametrize("action", [PlayerAction.HIT, PlayerAction.SURRENDER])
def test_all_busted_or_surrendered_keeps_hole_hidden(
    action: PlayerAction,
) -> None:
    game = arranged_round("10", "9", "6", "7", "10")
    game.act(action)
    assert game.public_state.phase is RoundPhase.SETTLED
    assert not game.public_state.dealer_hole_revealed
    assert game.public_state.dealer_cards == cards("9")
    assert game.public_state.visible_card_history.count(cards("7")[0]) == 0


def test_every_exposed_card_appears_once_in_public_history() -> None:
    game = arranged_round("10", "9", "7", "7", "4", "10")
    game.act(PlayerAction.HIT)
    assert game.public_state.visible_card_history == cards(
        "10", "9", "7", "4", "7", "10"
    )
    reveal_events = [
        event
        for event in game.public_state.events
        if event.event_type is EventType.DEALER_HOLE_REVEALED
    ]
    assert len(reveal_events) == 1


def test_illegal_actions_and_invalid_phases_are_rejected() -> None:
    insurance = arranged_round("10", "A", "7", "9")
    with pytest.raises(InvalidPhaseError):
        insurance.act(PlayerAction.HIT)

    ordinary = arranged_round("10", "9", "7", "8")
    with pytest.raises(InvalidPhaseError):
        ordinary.decide_insurance(InsuranceAction.TAKE)
    with pytest.raises(IllegalActionError):
        ordinary.act(PlayerAction.SPLIT)
    ordinary.act(PlayerAction.STAND)
    with pytest.raises(InvalidPhaseError):
        ordinary.act(PlayerAction.STAND)


def test_public_rank_values_remain_typed() -> None:
    game = arranged_round("5", "9", "6", "7")
    assert game.public_state.dealer_cards[0].rank is Rank.NINE
