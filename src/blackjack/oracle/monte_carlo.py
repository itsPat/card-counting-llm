"""Deterministic native rollouts under the documented fixed playing policy."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

from blackjack.engine.actions import PlayerAction
from blackjack.engine.rules import FIXED_RULES, CasinoRules
from blackjack.oracle.composition import CARD_VALUES, Composition
from blackjack.oracle.dealer import PeekCondition
from blackjack.oracle.distributions import ReturnDistribution
from blackjack.oracle.player import (
    OracleHand,
    PlayerSituation,
    RoundPlayerSituation,
)
from blackjack.oracle.player import (
    legal_actions as oracle_legal_actions,
)

_MINIMUM_PROFIT_HALF_UNITS = -17
_MAXIMUM_PROFIT_HALF_UNITS = 18
_OUTCOME_COUNT = _MAXIMUM_PROFIT_HALF_UNITS - _MINIMUM_PROFIT_HALF_UNITS + 1
_SOURCE = Path(__file__).with_name("native") / "fixed_policy.cpp"
_BUILD_LOCK = Lock()
_function: _NativeSimulationFunction | None = None
_play_function: _NativePlaySimulationFunction | None = None

_ACTION_CODE: dict[PlayerAction, int] = {
    PlayerAction.HIT: 0,
    PlayerAction.STAND: 1,
    PlayerAction.DOUBLE: 2,
    PlayerAction.SPLIT: 3,
    PlayerAction.SURRENDER: 4,
}
_FROM_SPLIT = 1 << 0
_SPLIT_ACES = 1 << 1
_CAN_DOUBLE = 1 << 2
_CAN_SURRENDER = 1 << 3
_SURRENDERED = 1 << 4
_FINISHED = 1 << 5


class _NativeSimulationFunction(Protocol):
    argtypes: tuple[object, ...]
    restype: object

    def __call__(
        self,
        counts: object,
        unseen_unavailable: int,
        seed: int,
        rollouts: int,
        output: object,
    ) -> int: ...


class _NativePlaySimulationFunction(Protocol):
    argtypes: tuple[object, ...]
    restype: object

    def __call__(
        self,
        counts: object,
        unseen_unavailable: int,
        dealer_upcard: int,
        negative_peek: int,
        hand_counts: object,
        hand_totals: object,
        wagers: object,
        hand_flags: object,
        hand_count: int,
        active_hand: int,
        action: int,
        seed: int,
        rollouts: int,
        output: object,
    ) -> int: ...


class NativeSimulationBuildError(RuntimeError):
    """Raised when the local fixed-policy rollout kernel cannot be compiled."""


@dataclass(frozen=True, slots=True)
class FixedPolicyRoundEstimate:
    """An empirical complete return distribution and its sampling uncertainty."""

    distribution: ReturnDistribution
    seed: int
    rollouts: int
    expected_profit_standard_error: float

    @property
    def expected_profit_confidence_interval_95(self) -> tuple[float, float]:
        mean = float(self.distribution.expected_profit)
        radius = 1.96 * self.expected_profit_standard_error
        return (mean - radius, mean + radius)


@dataclass(frozen=True, slots=True)
class FixedPolicyActionEstimate:
    """One forced first action followed by the fixed rollout policy."""

    action: PlayerAction
    distribution: ReturnDistribution
    seed: int
    rollouts: int
    expected_profit_standard_error: float

    @property
    def expected_profit(self) -> Fraction:
        return self.distribution.expected_profit

    @property
    def expected_profit_confidence_interval_95(self) -> tuple[float, float]:
        mean = float(self.expected_profit)
        radius = 1.96 * self.expected_profit_standard_error
        return (mean - radius, mean + radius)


def fixed_policy_rollout_seed(
    composition: Composition,
    unseen_unavailable: int,
    master_seed: int,
) -> int:
    """Derive one stable 64-bit rollout seed from the complete public state."""

    if unseen_unavailable < 0:
        raise ValueError("unseen unavailable count cannot be negative")
    payload = (
        f"{master_seed}|{unseen_unavailable}|"
        + ",".join(str(count) for count in composition.counts)
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode()).digest()[:8],
        byteorder="big",
        signed=False,
    )


def fixed_policy_round_return_estimate(
    composition: Composition,
    *,
    unseen_unavailable: int,
    seed: int,
    rollouts: int,
    rules: CasinoRules = FIXED_RULES,
) -> FixedPolicyRoundEstimate:
    """Simulate complete H17 basic-strategy rounds with exact insurance.

    SplitMix64 and rejection-sampled bounded draws define the replay contract,
    rather than relying on a standard-library random distribution.
    """

    if rules != FIXED_RULES:
        raise ValueError("native fixed-policy simulation uses the fixed rules")
    if unseen_unavailable < 0:
        raise ValueError("unseen unavailable count cannot be negative")
    if rollouts <= 0:
        raise ValueError("rollout count must be positive")
    if not 0 <= seed < 2**64:
        raise ValueError("rollout seed must fit in an unsigned 64-bit integer")

    function = _load_function()
    count_array = (ctypes.c_int * 10)(*composition.counts)
    output_array = (ctypes.c_uint64 * _OUTCOME_COUNT)()
    status = function(
        count_array,
        unseen_unavailable,
        seed,
        rollouts,
        output_array,
    )
    if status != 0:
        raise ValueError(f"native fixed-policy simulation rejected state ({status})")

    distribution, standard_error = _distribution_and_standard_error(
        output_array,
        rollouts,
    )
    return FixedPolicyRoundEstimate(
        distribution=distribution,
        seed=seed,
        rollouts=rollouts,
        expected_profit_standard_error=standard_error,
    )


def fixed_policy_play_rollout_seed(
    situation: RoundPlayerSituation,
    master_seed: int,
) -> int:
    """Derive a stable seed from the complete player-visible round state."""

    if not 0 <= master_seed < 2**64:
        raise ValueError("master rollout seed must fit in unsigned 64 bits")
    hand_payload = tuple(
        _oracle_hand_seed_payload(hand)
        for hand in (
            situation.active_hand,
            *situation.pending_hands,
        )
    )
    finished_payload = tuple(
        (
            hand.total,
            hand.wager.numerator,
            hand.wager.denominator,
            hand.is_natural_blackjack,
            hand.is_bust,
            hand.surrendered,
        )
        for hand in situation.finished_hands
    )
    payload = repr(
        (
            master_seed,
            situation.composition.counts,
            situation.unseen_unavailable,
            situation.dealer_upcard.value,
            situation.peek_condition.value,
            hand_payload,
            finished_payload,
        )
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode()).digest()[:8],
        byteorder="big",
        signed=False,
    )


def fixed_policy_play_action_estimates(
    situation: RoundPlayerSituation,
    legal_actions: tuple[PlayerAction, ...],
    *,
    seed: int,
    rollouts: int,
) -> tuple[FixedPolicyActionEstimate, ...]:
    """Estimate every legal first action, then follow fixed H17 basic strategy.

    Every action uses matched per-rollout random streams. The estimates therefore
    measure action value under the documented continuation policy, not exact
    rational continuation.
    """

    if situation.rules != FIXED_RULES:
        raise ValueError("native play simulation uses the fixed rules")
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise ValueError("play simulation needs unique legal actions")
    if rollouts <= 0:
        raise ValueError("rollout count must be positive")
    if not 0 <= seed < 2**64:
        raise ValueError("rollout seed must fit in an unsigned 64-bit integer")
    permitted_actions = oracle_legal_actions(
        PlayerSituation(
            composition=situation.composition,
            hand=situation.active_hand,
            dealer_upcard=situation.dealer_upcard,
            peek_condition=situation.peek_condition,
            rules=situation.rules,
            unseen_unavailable=situation.unseen_unavailable,
        ),
        hands_in_round=(
            1
            + len(situation.pending_hands)
            + len(situation.finished_hands)
        ),
    )
    if any(action not in permitted_actions for action in legal_actions):
        raise ValueError(
            "play simulation received an illegal action: "
            f"requested={legal_actions!r}, permitted={permitted_actions!r}"
        )

    function = _load_play_function()
    packed = _pack_round_situation(situation)
    count_array = (ctypes.c_int * 10)(*situation.composition.counts)
    hand_counts = (ctypes.c_int * 40)(*packed.hand_counts)
    hand_totals = (ctypes.c_int * 4)(*packed.hand_totals)
    wagers = (ctypes.c_int * 4)(*packed.wagers)
    hand_flags = (ctypes.c_uint8 * 4)(*packed.hand_flags)
    estimates: list[FixedPolicyActionEstimate] = []
    for action in legal_actions:
        output_array = (ctypes.c_uint64 * _OUTCOME_COUNT)()
        status = function(
            count_array,
            situation.unseen_unavailable,
            CARD_VALUES.index(situation.dealer_upcard),
            int(situation.peek_condition is PeekCondition.NO_BLACKJACK),
            hand_counts,
            hand_totals,
            wagers,
            hand_flags,
            packed.hand_count,
            packed.active_hand,
            _ACTION_CODE[action],
            seed,
            rollouts,
            output_array,
        )
        if status != 0:
            raise ValueError(
                "native fixed-policy play simulation rejected state "
                f"for {action.value} ({status})"
            )
        distribution, standard_error = _distribution_and_standard_error(
            output_array,
            rollouts,
        )
        estimates.append(
            FixedPolicyActionEstimate(
                action=action,
                distribution=distribution,
                seed=seed,
                rollouts=rollouts,
                expected_profit_standard_error=standard_error,
            )
        )
    return tuple(estimates)


def native_simulation_library_path() -> Path:
    source_digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return _SOURCE.parent / ".build" / f"fixed-policy-{source_digest}{suffix}"


def ensure_native_simulation_kernel() -> Path:
    """Compile and load the deterministic rollout entry point."""

    _load_function()
    _load_play_function()
    return native_simulation_library_path()


def _load_function() -> _NativeSimulationFunction:
    global _function
    if _function is not None:
        return _function
    with _BUILD_LOCK:
        if _function is not None:
            return _function
        library_path = native_simulation_library_path()
        if not library_path.exists():
            _compile(library_path)
        library = ctypes.CDLL(str(library_path))
        function = cast(
            _NativeSimulationFunction,
            library.blackjack_fixed_policy_simulation,
        )
        function.argtypes = (
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        )
        function.restype = ctypes.c_int
        _function = function
        return function


def _load_play_function() -> _NativePlaySimulationFunction:
    global _play_function
    if _play_function is not None:
        return _play_function
    with _BUILD_LOCK:
        if _play_function is not None:
            return _play_function
        library_path = native_simulation_library_path()
        if not library_path.exists():
            _compile(library_path)
        library = ctypes.CDLL(str(library_path))
        function = cast(
            _NativePlaySimulationFunction,
            library.blackjack_play_action_simulation,
        )
        function.argtypes = (
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        )
        function.restype = ctypes.c_int
        _play_function = function
        return function


@dataclass(frozen=True, slots=True)
class _PackedRoundSituation:
    hand_counts: tuple[int, ...]
    hand_totals: tuple[int, ...]
    wagers: tuple[int, ...]
    hand_flags: tuple[int, ...]
    hand_count: int
    active_hand: int


def _pack_round_situation(
    situation: RoundPlayerSituation,
) -> _PackedRoundSituation:
    hand_count = (
        len(situation.finished_hands)
        + 1
        + len(situation.pending_hands)
    )
    counts: list[int] = []
    totals: list[int] = []
    wagers: list[int] = []
    flags: list[int] = []
    for hand in situation.finished_hands:
        counts.extend((0,) * 10)
        totals.append(hand.total)
        wagers.append(_wager_half_units(hand.wager))
        flags.append(
            _FINISHED
            | (_SURRENDERED if hand.surrendered else 0)
        )
    for hand in (situation.active_hand, *situation.pending_hands):
        counts.extend(hand.cards.count(value) for value in CARD_VALUES)
        totals.append(0)
        wagers.append(_wager_half_units(hand.wager))
        flags.append(
            (_FROM_SPLIT if hand.from_split else 0)
            | (_SPLIT_ACES if hand.split_aces else 0)
            | (_CAN_DOUBLE if hand.can_double else 0)
            | (_CAN_SURRENDER if hand.can_surrender else 0)
        )
    padding = 4 - hand_count
    counts.extend((0,) * (padding * 10))
    totals.extend((0,) * padding)
    wagers.extend((0,) * padding)
    flags.extend((0,) * padding)
    return _PackedRoundSituation(
        hand_counts=tuple(counts),
        hand_totals=tuple(totals),
        wagers=tuple(wagers),
        hand_flags=tuple(flags),
        hand_count=hand_count,
        active_hand=len(situation.finished_hands),
    )


def _wager_half_units(wager: Fraction) -> int:
    half_units = wager * 2
    if half_units.denominator != 1:
        raise ValueError("native simulation needs wagers in half-unit increments")
    return half_units.numerator


def _oracle_hand_seed_payload(hand: OracleHand) -> tuple[object, ...]:
    return (
        tuple(card.value for card in hand.cards),
        hand.wager.numerator,
        hand.wager.denominator,
        hand.from_split,
        hand.split_aces,
        hand.can_double,
        hand.can_surrender,
    )


def _distribution_and_standard_error(
    output: ctypes.Array[ctypes.c_uint64],
    rollouts: int,
) -> tuple[ReturnDistribution, float]:
    distribution = ReturnDistribution.from_pairs(
        (
            Fraction(profit_half_units, 2),
            Fraction(count, rollouts),
        )
        for profit_half_units, count in zip(
            range(_MINIMUM_PROFIT_HALF_UNITS, _MAXIMUM_PROFIT_HALF_UNITS + 1),
            output,
            strict=True,
        )
        if count > 0
    )
    mean = float(distribution.expected_profit)
    variance = sum(
        float(outcome.probability) * (float(outcome.profit) - mean) ** 2
        for outcome in distribution.outcomes
    )
    return distribution, sqrt(variance / rollouts)


def _compile(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    command = [
        "c++",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(_SOURCE),
        "-o",
        str(temporary),
    ]
    if platform.system() == "Darwin":
        command.insert(1, "-dynamiclib")
    else:
        command[1:1] = ["-shared", "-fPIC"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise NativeSimulationBuildError(
            "a C++20 compiler is required for production bet labels"
        ) from error
    if completed.returncode != 0:
        raise NativeSimulationBuildError(
            "failed to compile the native fixed-policy simulation kernel:\n"
            f"{completed.stderr.strip()}"
        )
    temporary.replace(output)
