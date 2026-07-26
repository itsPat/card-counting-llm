"""Process-safe persistent cache for completed exact oracle labels."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import sleep, time
from typing import Protocol, cast
from uuid import uuid4

from blackjack.analysis import SELECTED_BET_VOCABULARY
from blackjack.dataset.io import (
    labeled_decision_from_json,
    labeled_decision_to_json,
)
from blackjack.dataset.labeling import LabeledDecision
from blackjack.engine import FIXED_RULES, PlayerAction
from blackjack.oracle import (
    Composition,
    OracleHand,
    ResolvedHand,
    RoundPlayerSituation,
)

CACHE_SCHEMA_VERSION = 4


class LabelOracle(Protocol):
    def label_bet(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision: ...

    def label_insurance(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision: ...

    def label_play(
        self,
        situation: RoundPlayerSituation,
        legal_actions: tuple[PlayerAction, ...],
    ) -> LabeledDecision: ...


class CacheClaimTimeoutError(TimeoutError):
    """Raised when another process holds a label claim beyond the wait limit."""


class CachedLabelMismatchError(RuntimeError):
    """Raised if two computations disagree for one canonical state key."""


@dataclass(slots=True)
class LabelCacheStatistics:
    hits: int = 0
    misses: int = 0
    waits: int = 0
    writes: int = 0


class SQLiteLabelCache:
    """A coarse label cache; recursive hot states remain in process memory."""

    __slots__ = (
        "_claim_stale_seconds",
        "_path",
        "_poll_seconds",
        "_statistics",
        "_wait_timeout_seconds",
    )

    def __init__(
        self,
        path: Path,
        *,
        poll_seconds: float = 0.25,
        wait_timeout_seconds: float = 7200,
        claim_stale_seconds: float = 14400,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("cache poll interval must be positive")
        if wait_timeout_seconds <= 0:
            raise ValueError("cache wait timeout must be positive")
        if claim_stale_seconds <= wait_timeout_seconds:
            raise ValueError("stale claim interval must exceed wait timeout")
        self._path = path
        self._poll_seconds = poll_seconds
        self._wait_timeout_seconds = wait_timeout_seconds
        self._claim_stale_seconds = claim_stale_seconds
        self._statistics = LabelCacheStatistics()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    cache_key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    cache_key TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    claimed_at REAL NOT NULL
                )
                """
            )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def statistics(self) -> LabelCacheStatistics:
        return LabelCacheStatistics(
            hits=self._statistics.hits,
            misses=self._statistics.misses,
            waits=self._statistics.waits,
            writes=self._statistics.writes,
        )

    def get_or_compute(
        self,
        *,
        kind: str,
        state: str,
        compute: ComputeLabel,
    ) -> LabeledDecision:
        key = _cache_key(kind, state)
        cached = self._read(key)
        if cached is not None:
            self._statistics.hits += 1
            return labeled_decision_from_json(cached)

        owner = uuid4().hex
        waited = False
        wait_started = time()
        while True:
            if self._claim(key, owner):
                cached = self._read(key)
                if cached is None:
                    break
                self._release(key, owner)
                self._statistics.hits += 1
                return labeled_decision_from_json(cached)
            if not waited:
                self._statistics.waits += 1
                waited = True
            cached = self._read(key)
            if cached is not None:
                self._statistics.hits += 1
                return labeled_decision_from_json(cached)
            if time() - wait_started >= self._wait_timeout_seconds:
                raise CacheClaimTimeoutError(
                    f"timed out waiting for cached {kind} label"
                )
            sleep(self._poll_seconds)

        self._statistics.misses += 1
        try:
            label = compute()
            payload = labeled_decision_to_json(label)
            winner = self._store(key, kind, state, payload)
            if winner != payload:
                raise CachedLabelMismatchError(
                    "concurrent computations produced different labels"
                )
            self._statistics.writes += 1
            return label
        finally:
            self._release(key, owner)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _read(self, key: str) -> str | None:
        with self._connect() as connection:
            row: object = connection.execute(
                "SELECT payload FROM labels WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        values = cast(tuple[object, ...], row)
        if len(values) != 1 or not isinstance(values[0], str):
            raise RuntimeError("label cache returned an invalid payload row")
        return values[0]

    def _claim(self, key: str, owner: str) -> bool:
        now = time()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM claims WHERE claimed_at < ?",
                (now - self._claim_stale_seconds,),
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO claims(cache_key, owner, claimed_at)
                VALUES (?, ?, ?)
                """,
                (key, owner, now),
            )
            return cursor.rowcount == 1

    def _store(
        self,
        key: str,
        kind: str,
        state: str,
        payload: str,
    ) -> str:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO labels(
                    cache_key, kind, state, payload, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (key, kind, state, payload, time()),
            )
        winner = self._read(key)
        if winner is None:
            raise RuntimeError("stored cache label could not be read back")
        return winner

    def _release(self, key: str, owner: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM claims WHERE cache_key = ? AND owner = ?",
                (key, owner),
            )


type ComputeLabel = Callable[[], LabeledDecision]


class CachedDatasetOracle:
    """Cache labels by canonical public oracle state and methodology."""

    __slots__ = ("_cache", "_namespace", "_oracle")

    def __init__(
        self,
        oracle: LabelOracle,
        cache: SQLiteLabelCache,
        *,
        namespace: str | None = None,
    ) -> None:
        self._oracle = oracle
        self._cache = cache
        oracle_type = type(oracle)
        self._namespace = (
            namespace
            if namespace is not None
            else f"{oracle_type.__module__}.{oracle_type.__qualname__}"
        )

    @property
    def cache(self) -> SQLiteLabelCache:
        return self._cache

    def label_bet(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        state = _state_json(
            self._namespace,
            {
                "composition": list(composition.counts),
                "unseen_unavailable": unseen_unavailable,
            },
        )
        return self._cache.get_or_compute(
            kind="bet",
            state=state,
            compute=lambda: self._oracle.label_bet(
                composition,
                unseen_unavailable,
            ),
        )

    def label_insurance(
        self,
        composition: Composition,
        unseen_unavailable: int,
    ) -> LabeledDecision:
        state = _state_json(
            self._namespace,
            {
                "composition": list(composition.counts),
                "unseen_unavailable": unseen_unavailable,
            },
        )
        return self._cache.get_or_compute(
            kind="insurance",
            state=state,
            compute=lambda: self._oracle.label_insurance(
                composition,
                unseen_unavailable,
            ),
        )

    def label_play(
        self,
        situation: RoundPlayerSituation,
        legal_actions: tuple[PlayerAction, ...],
    ) -> LabeledDecision:
        state = _state_json(
            self._namespace,
            {
                "composition": list(situation.composition.counts),
                "unseen_unavailable": situation.unseen_unavailable,
                "dealer_upcard": situation.dealer_upcard.value,
                "peek_condition": situation.peek_condition.value,
                "active_hand": _hand_data(situation.active_hand),
                "pending_hands": [_hand_data(hand) for hand in situation.pending_hands],
                "finished_hands": [
                    _resolved_hand_data(hand) for hand in situation.finished_hands
                ],
                "legal_actions": [action.value for action in legal_actions],
            },
        )
        return self._cache.get_or_compute(
            kind="play",
            state=state,
            compute=lambda: self._oracle.label_play(
                situation,
                legal_actions,
            ),
        )


def _cache_key(kind: str, state: str) -> str:
    return f"{kind}:{sha256(state.encode()).hexdigest()}"


def _state_json(namespace: str, state: dict[str, object]) -> str:
    envelope = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "oracle_namespace": namespace,
        "rules": repr(FIXED_RULES),
        "bet_vocabulary": [
            (token.token.value, token.bankroll_fraction)
            for token in SELECTED_BET_VOCABULARY.tokens
        ],
        "state": state,
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"))


def _hand_data(hand: OracleHand) -> dict[str, object]:
    return {
        "cards": [card.value for card in hand.cards],
        "wager": (hand.wager.numerator, hand.wager.denominator),
        "from_split": hand.from_split,
        "split_aces": hand.split_aces,
        "can_double": hand.can_double,
        "can_surrender": hand.can_surrender,
    }


def _resolved_hand_data(hand: ResolvedHand) -> dict[str, object]:
    return {
        "total": hand.total,
        "wager": (hand.wager.numerator, hand.wager.denominator),
        "is_natural_blackjack": hand.is_natural_blackjack,
        "is_bust": hand.is_bust,
        "surrendered": hand.surrendered,
    }
