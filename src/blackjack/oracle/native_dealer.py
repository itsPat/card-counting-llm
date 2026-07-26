"""Typed loader for the small native dealer-probability kernel."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Final, Protocol, cast

from blackjack.oracle.dealer import PeekCondition

type NativeDealerDistribution = tuple[tuple[int, float], ...]

_DEALER_BUST: Final = 22
_DEALER_BLACKJACK: Final = 23
_SOURCE = Path(__file__).with_name("native") / "dealer.cpp"
_BUILD_LOCK = Lock()
_function: _NativeFunction | None = None
_split_function: _NativeSplitFunction | None = None


class _NativeFunction(Protocol):
    argtypes: tuple[object, ...]
    restype: object

    def __call__(
        self,
        counts: object,
        upcard_index: int,
        no_blackjack: int,
        hit_soft_17: int,
        output: object,
    ) -> int: ...


class _NativeSplitFunction(Protocol):
    argtypes: tuple[object, ...]
    restype: object

    def __call__(
        self,
        counts: object,
        pair_card: int,
        upcard_index: int,
        no_blackjack: int,
        hit_soft_17: int,
        endpoint_count: int,
        endpoint_cards: object,
        endpoint_wagers: object,
        endpoint_multiplicities: object,
        output: object,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class NativeSplitEndpoint:
    cards: tuple[int, ...]
    wager_half_units: int
    multiplicity: int


class NativeDealerBuildError(RuntimeError):
    """Raised when the local C++ dealer kernel cannot be compiled."""


def native_dealer_distribution(
    counts: tuple[int, ...],
    upcard_index: int,
    condition: PeekCondition,
    *,
    hit_soft_17: bool,
) -> NativeDealerDistribution:
    if len(counts) != 10:
        raise ValueError("native dealer composition needs ten counts")
    function = _load_function()
    count_array = (ctypes.c_int * 10)(*counts)
    output_array = (ctypes.c_double * 7)()
    status = function(
        count_array,
        upcard_index,
        int(condition is PeekCondition.NO_BLACKJACK),
        int(hit_soft_17),
        output_array,
    )
    if status != 0:
        raise ValueError(f"native dealer calculation rejected state ({status})")
    outcomes = (17, 18, 19, 20, 21, _DEALER_BUST, _DEALER_BLACKJACK)
    return tuple(
        (outcome, probability)
        for outcome, probability in zip(outcomes, output_array, strict=True)
        if probability > 0
    )


def native_library_path() -> Path:
    source_digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return _SOURCE.parent / ".build" / f"dealer-{source_digest}{suffix}"


def ensure_native_dealer_kernel() -> Path:
    """Compile and load the native entry points before workers are spawned."""

    _load_function()
    _load_split_function()
    return native_library_path()


def native_split_distribution(
    counts: tuple[int, ...],
    pair_card: int,
    upcard_index: int,
    condition: PeekCondition,
    endpoints: tuple[NativeSplitEndpoint, ...],
    *,
    hit_soft_17: bool,
) -> tuple[tuple[int, float], ...]:
    if len(counts) != 10:
        raise ValueError("native split composition needs ten counts")
    if not endpoints:
        raise ValueError("native split calculation needs endpoints")
    if any(len(endpoint.cards) != 10 for endpoint in endpoints):
        raise ValueError("native split endpoint needs ten card counts")
    function = _load_split_function()
    count_array = (ctypes.c_int * 10)(*counts)
    flat_cards = tuple(card for endpoint in endpoints for card in endpoint.cards)
    cards_array = (ctypes.c_int16 * len(flat_cards))(*flat_cards)
    wagers_array = (ctypes.c_int16 * len(endpoints))(
        *(endpoint.wager_half_units for endpoint in endpoints)
    )
    multiplicities_array = (ctypes.c_uint64 * len(endpoints))(
        *(endpoint.multiplicity for endpoint in endpoints)
    )
    output_array = (ctypes.c_double * 17)()
    status = function(
        count_array,
        pair_card,
        upcard_index,
        int(condition is PeekCondition.NO_BLACKJACK),
        int(hit_soft_17),
        len(endpoints),
        cards_array,
        wagers_array,
        multiplicities_array,
        output_array,
    )
    if status != 0:
        raise ValueError(f"native split calculation rejected state ({status})")
    return tuple(
        (profit_half_units, probability)
        for profit_half_units, probability in zip(
            range(-8, 9),
            output_array,
            strict=True,
        )
        if probability > 0
    )


def _load_function() -> _NativeFunction:
    global _function
    if _function is not None:
        return _function
    with _BUILD_LOCK:
        if _function is not None:
            return _function
        library_path = native_library_path()
        if not library_path.exists():
            _compile(library_path)
        library = ctypes.CDLL(str(library_path))
        function = cast(
            _NativeFunction,
            library.blackjack_dealer_distribution,
        )
        function.argtypes = (
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        )
        function.restype = ctypes.c_int
        _function = function
        return function


def _load_split_function() -> _NativeSplitFunction:
    global _split_function
    if _split_function is not None:
        return _split_function
    with _BUILD_LOCK:
        if _split_function is not None:
            return _split_function
        library_path = native_library_path()
        if not library_path.exists():
            _compile(library_path)
        library = ctypes.CDLL(str(library_path))
        function = cast(
            _NativeSplitFunction,
            library.blackjack_split_distribution,
        )
        function.argtypes = (
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
        )
        function.restype = ctypes.c_int
        _split_function = function
        return function


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
        raise NativeDealerBuildError(
            "a C++20 compiler is required for production bet labels"
        ) from error
    if completed.returncode != 0:
        raise NativeDealerBuildError(
            f"failed to compile the native dealer kernel:\n{completed.stderr.strip()}"
        )
    temporary.replace(output)
