from __future__ import annotations

import pytest

from blackjack import Hand, Rank, calculate_hand_value, cards


@pytest.mark.parametrize(
    ("labels", "total"),
    [
        (("2", "3"), 5),
        (("4", "5"), 9),
        (("6", "7"), 13),
        (("8", "9"), 17),
        (("10", "J"), 20),
        (("Q", "K", "2"), 22),
    ],
)
def test_hard_hand_boundaries(labels: tuple[str, ...], total: int) -> None:
    value = calculate_hand_value(cards(*labels))
    assert value.total == total
    assert not value.is_soft
    assert value.is_bust is (total > 21)


@pytest.mark.parametrize(
    ("labels", "total", "soft"),
    [
        (("A",), 11, True),
        (("A", "2"), 13, True),
        (("A", "6"), 17, True),
        (("A", "9"), 20, True),
        (("A", "10"), 21, True),
        (("A", "A"), 12, True),
        (("A", "A", "8"), 20, True),
        (("A", "A", "9"), 21, True),
        (("A", "A", "10"), 12, False),
        (("A", "A", "A", "8"), 21, True),
        (("A", "9", "A", "K"), 21, False),
    ],
)
def test_soft_and_multiple_ace_boundaries(
    labels: tuple[str, ...],
    total: int,
    soft: bool,
) -> None:
    value = calculate_hand_value(cards(*labels))
    assert value.total == total
    assert value.is_soft is soft
    assert not value.is_bust


@pytest.mark.parametrize(
    "labels",
    [
        ("10", "Q", "2"),
        ("A", "K", "A", "K"),
        ("A", "9", "5", "7"),
    ],
)
def test_busts(labels: tuple[str, ...]) -> None:
    assert calculate_hand_value(cards(*labels)).is_bust


def test_rank_values_and_ten_valued_faces() -> None:
    assert Rank.ACE.hard_value == 1
    assert Rank.NINE.hard_value == 9
    assert all(rank.hard_value == 10 for rank in Rank if rank.is_ten_valued)


def test_only_an_original_two_card_twenty_one_is_natural() -> None:
    assert Hand(cards("A", "K")).is_natural_blackjack
    assert not Hand(cards("7", "7", "7")).is_natural_blackjack
    assert not Hand(cards("A", "K"), from_split=True).is_natural_blackjack


def test_any_ten_valued_pair_can_split() -> None:
    assert Hand(cards("10", "K")).is_pair
    assert Hand(cards("J", "Q")).is_pair
    assert not Hand(cards("9", "K")).is_pair
