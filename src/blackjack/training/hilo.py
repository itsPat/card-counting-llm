"""Visibility-matched six-deck H17 Hi-Lo evaluation control."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import torch
from torch import Tensor

from blackjack.analysis import BetAction
from blackjack.dataset import DecisionKind, InsuranceToken, PlayToken, play_token
from blackjack.engine import PlayerAction
from blackjack.oracle import CardValue, basic_strategy_action, oracle_hand_value
from blackjack.training.data import DecisionBatch, DecisionDataset
from blackjack.training.vocabulary import BlackjackVocabulary

_INITIAL_CARD_COUNT = 6 * 52
_CARDS_PER_DECK = 52
_CARD_TOKENS = frozenset(card.value for card in CardValue)

_PLAY_ACTIONS: dict[PlayToken, PlayerAction] = {
    PlayToken.HIT: PlayerAction.HIT,
    PlayToken.STAND: PlayerAction.STAND,
    PlayToken.DOUBLE: PlayerAction.DOUBLE,
    PlayToken.SPLIT: PlayerAction.SPLIT,
    PlayToken.SURRENDER: PlayerAction.SURRENDER,
}


@dataclass(frozen=True, slots=True)
class HiLoBetRamp:
    """Map floored true counts onto the fixed bankroll-fraction tokens."""

    low_index: int = 2
    medium_index: int = 4
    high_index: int = 6

    def __post_init__(self) -> None:
        if not self.low_index < self.medium_index < self.high_index:
            raise ValueError("Hi-Lo bet-ramp indices must increase")

    def action(self, true_count: int) -> BetAction:
        if true_count >= self.high_index:
            return BetAction.HIGH
        if true_count >= self.medium_index:
            return BetAction.MEDIUM
        if true_count >= self.low_index:
            return BetAction.LOW
        return BetAction.MINIMUM


_DEFAULT_BET_RAMP = HiLoBetRamp()


def hi_lo_running_count(cards: tuple[CardValue, ...]) -> int:
    """Return the balanced Hi-Lo running count for visible cards."""

    return sum(
        1
        if card.hard_value in range(2, 7)
        else -1
        if card in (CardValue.TEN, CardValue.ACE)
        else 0
        for card in cards
    )


def floored_true_count(cards: tuple[CardValue, ...]) -> int:
    """Convert visible cards to the standard floored six-deck true count."""

    remaining_cards = _INITIAL_CARD_COUNT - len(cards)
    if remaining_cards <= 0:
        raise ValueError("true count needs cards remaining in the shoe")
    decks_remaining = remaining_cards / _CARDS_PER_DECK
    return floor(hi_lo_running_count(cards) / decks_remaining)


def hi_lo_play_action(
    cards: tuple[CardValue, ...],
    dealer_upcard: CardValue,
    legal_actions: tuple[PlayerAction, ...],
    true_count: int,
) -> PlayerAction:
    """Apply H17 Illustrious 18 and Fab 4 indices over basic strategy."""

    if not cards:
        raise ValueError("Hi-Lo play needs a player hand")
    if not legal_actions:
        raise ValueError("Hi-Lo play needs legal actions")
    value = oracle_hand_value(cards)
    dealer = 11 if dealer_upcard is CardValue.ACE else dealer_upcard.hard_value

    if PlayerAction.SURRENDER in legal_actions and not value.is_soft:
        surrender_indices = {
            (14, 10): 3,
            (15, 9): 2,
            (15, 10): 0,
            (15, 11): -1,
        }
        surrender_index = surrender_indices.get((value.total, dealer))
        if surrender_index is not None:
            return (
                PlayerAction.SURRENDER
                if true_count >= surrender_index
                else PlayerAction.HIT
            )

    basic = basic_strategy_action(cards, dealer_upcard, legal_actions)
    can_surrender = PlayerAction.SURRENDER in legal_actions

    if (
        PlayerAction.SPLIT in legal_actions
        and len(cards) == 2
        and cards[0] is CardValue.TEN
        and cards[1] is CardValue.TEN
    ):
        split_index = {5: 5, 6: 4}.get(dealer)
        if split_index is not None and true_count >= split_index:
            return PlayerAction.SPLIT

    if value.is_soft:
        return basic

    index_plays: dict[
        tuple[int, int],
        tuple[int, PlayerAction],
    ] = {
        (16, 10): (0, PlayerAction.STAND),
        (15, 10): (4, PlayerAction.STAND),
        (10, 10): (4, PlayerAction.DOUBLE),
        (12, 3): (2, PlayerAction.STAND),
        (12, 2): (3, PlayerAction.STAND),
        (11, 11): (-1, PlayerAction.DOUBLE),
        (9, 2): (1, PlayerAction.DOUBLE),
        (10, 11): (3, PlayerAction.DOUBLE),
        (9, 7): (3, PlayerAction.DOUBLE),
        (16, 9): (5, PlayerAction.STAND),
        (13, 2): (-1, PlayerAction.STAND),
        (12, 4): (0, PlayerAction.STAND),
        (12, 5): (-2, PlayerAction.STAND),
        (12, 6): (-3, PlayerAction.STAND),
        (13, 3): (-2, PlayerAction.STAND),
    }
    index_play = index_plays.get((value.total, dealer))
    if index_play is None:
        return basic
    index, action = index_play
    if can_surrender and (value.total, dealer) in ((15, 10), (16, 9), (16, 10)):
        return basic
    if true_count >= index and action in legal_actions:
        return action
    if true_count < index and PlayerAction.HIT in legal_actions:
        return PlayerAction.HIT
    return basic


class HiLoBaseline:
    """Use only the visible-card Hi-Lo count and current decision state."""

    def __init__(
        self,
        dataset: DecisionDataset,
        *,
        bet_ramp: HiLoBetRamp = _DEFAULT_BET_RAMP,
    ) -> None:
        self._vocabulary = dataset.vocabulary
        self._bet_ramp = bet_ramp

    def logits(self, batch: DecisionBatch) -> Tensor:
        logits = torch.zeros(
            (
                batch.batch_size,
                batch.input_ids.shape[1],
                len(self._vocabulary),
            ),
            dtype=torch.float32,
            device=batch.input_ids.device,
        )
        for row, kind in enumerate(batch.kinds):
            tokens = self._tokens(batch, row)
            visible_cards = tuple(
                CardValue(token) for token in tokens if token in _CARD_TOKENS
            )
            true_count = floored_true_count(visible_cards)
            if kind is DecisionKind.BET:
                token = self._bet_ramp.action(true_count).value
            elif kind is DecisionKind.INSURANCE:
                token = (
                    InsuranceToken.TAKE.value
                    if true_count >= 3
                    else InsuranceToken.DECLINE.value
                )
            else:
                token = play_token(
                    self._play_action(batch, row, tokens, true_count)
                ).value
            position = int(batch.prediction_positions[row].item())
            logits[row, position, self._vocabulary.id_for(token)] = 1
        return logits

    def _tokens(
        self,
        batch: DecisionBatch,
        row: int,
    ) -> tuple[str, ...]:
        length = int(batch.prediction_positions[row].item()) + 1
        token_ids = tuple(
            int(token_id) for token_id in batch.input_ids[row, :length]
        )
        return self._vocabulary.decode(token_ids)

    def _play_action(
        self,
        batch: DecisionBatch,
        row: int,
        tokens: tuple[str, ...],
        true_count: int,
    ) -> PlayerAction:
        cards, dealer = _current_play_state(tokens)
        return hi_lo_play_action(
            cards,
            dealer,
            _legal_player_actions(batch, row, self._vocabulary),
            true_count,
        )


def _current_play_state(
    tokens: tuple[str, ...],
) -> tuple[tuple[CardValue, ...], CardValue]:
    try:
        player_start = tokens.index("<PLAYER>") + 1
        dealer_index = tokens.index("<DEALER>", player_start)
        dealer_token = tokens[dealer_index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("play input lacks player/dealer structure") from error
    return (
        tuple(CardValue(token) for token in tokens[player_start:dealer_index]),
        CardValue(dealer_token),
    )


def _legal_player_actions(
    batch: DecisionBatch,
    row: int,
    vocabulary: BlackjackVocabulary,
) -> tuple[PlayerAction, ...]:
    return tuple(
        action
        for token, action in _PLAY_ACTIONS.items()
        if bool(
            batch.legal_token_mask[
                row,
                vocabulary.id_for(token.value),
            ].item()
        )
    )
