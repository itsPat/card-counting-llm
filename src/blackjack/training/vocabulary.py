"""Stable token-to-index mapping for the blackjack decision model."""

from __future__ import annotations

from dataclasses import dataclass

from blackjack.analysis import BetAction
from blackjack.dataset.tokens import (
    InsuranceToken,
    PlayToken,
    StructureToken,
)
from blackjack.oracle import CARD_VALUES

PAD_TOKEN = "<PAD>"

_CARD_TOKENS = tuple(value.value for value in CARD_VALUES)
_STRUCTURE_TOKENS = tuple(token.value for token in StructureToken)
_DECISION_TOKENS = (
    *(token.value for token in BetAction),
    *(token.value for token in PlayToken),
    *(token.value for token in InsuranceToken),
)
_MODEL_TOKENS = (
    PAD_TOKEN,
    *_CARD_TOKENS,
    *_STRUCTURE_TOKENS,
    *_DECISION_TOKENS,
)


@dataclass(frozen=True, slots=True)
class BlackjackVocabulary:
    """One immutable and explicitly ordered model vocabulary."""

    tokens: tuple[str, ...] = _MODEL_TOKENS

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("model vocabulary cannot be empty")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("model vocabulary tokens must be unique")
        if self.tokens[0] != PAD_TOKEN:
            raise ValueError("padding must have the stable zero index")
        missing = set(_MODEL_TOKENS).difference(self.tokens)
        if missing:
            raise ValueError(f"model vocabulary is missing tokens: {missing!r}")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def decision_token_ids(self) -> frozenset[int]:
        return frozenset(self.id_for(token) for token in _DECISION_TOKENS)

    def id_for(self, token: str) -> int:
        try:
            return self.tokens.index(token)
        except ValueError as error:
            raise ValueError(f"unknown model token: {token!r}") from error

    def token_for(self, token_id: int) -> str:
        if not 0 <= token_id < len(self):
            raise ValueError(f"model token index is out of range: {token_id}")
        return self.tokens[token_id]

    def encode(self, tokens: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(self.id_for(token) for token in tokens)

    def decode(self, token_ids: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(self.token_for(token_id) for token_id in token_ids)


BLACKJACK_VOCABULARY = BlackjackVocabulary()
