from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

from blackjack import (
    Card,
    CasinoRules,
    InvalidReplayError,
    Rank,
    Shoe,
    ShoeExhaustedError,
    ShoeReplay,
    cards,
    six_deck_rank_counts,
)


def test_six_deck_shoe_has_exact_rank_counts() -> None:
    shoe = Shoe.shuffled(1234)
    observed = Counter(card.rank for card in shoe.replay.cards)
    assert observed == six_deck_rank_counts()
    assert len(shoe.replay.cards) == 312
    assert all(count == 24 for count in observed.values())


def test_burn_card_is_unseen_and_not_dealt() -> None:
    shoe = Shoe.arranged(cards("A", "2", "3"), burn_card=Card(Rank.KING))
    assert shoe.burn_card == Card(Rank.KING)
    assert shoe.dealt_count == 0
    assert shoe.deal() == Card(Rank.ACE)
    assert shoe.dealt_count == 1


@pytest.mark.parametrize("seed", [0, 1, 7, 99, 2**31])
def test_penetration_stays_within_fixed_boundaries(seed: int) -> None:
    replay = Shoe.shuffled(seed).replay
    assert 312 * Fraction(7, 10) <= replay.cut_card_position
    assert replay.cut_card_position <= 312 * Fraction(4, 5)


def test_identical_seeds_produce_identical_replays() -> None:
    assert Shoe.shuffled(2026).replay == Shoe.shuffled(2026).replay


def test_different_seeds_produce_different_shoes() -> None:
    assert Shoe.shuffled(1).replay != Shoe.shuffled(2).replay


def test_explicit_replay_reproduces_order_and_cut_position() -> None:
    original = Shoe.shuffled(88)
    replayed = Shoe.from_replay(original.replay)
    assert replayed.replay.cut_card_position == original.replay.cut_card_position
    assert tuple(replayed.deal() for _ in range(12)) == tuple(
        original.deal() for _ in range(12)
    )


def test_arranged_replay_tracks_cut_card() -> None:
    shoe = Shoe.arranged(cards("2", "3", "4"), cut_card_position=3)
    assert not shoe.reached_cut_card
    assert shoe.deal().rank is Rank.TWO
    assert shoe.deal().rank is Rank.THREE
    assert shoe.reached_cut_card


def test_replay_validation_and_exhaustion() -> None:
    with pytest.raises(InvalidReplayError):
        ShoeReplay(cards=cards("A"), cut_card_position=1)
    shoe = Shoe.arranged(cards("A"))
    assert shoe.deal().rank is Rank.ACE
    with pytest.raises(ShoeExhaustedError):
        shoe.deal()


def test_custom_rule_deck_count_is_respected() -> None:
    one_deck = CasinoRules(decks=1)
    assert len(Shoe.shuffled(3, one_deck).replay.cards) == 52
