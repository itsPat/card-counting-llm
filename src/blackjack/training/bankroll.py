"""Paired bankroll evaluation over deterministic held-out shoe replays."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from math import log, sqrt
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Protocol, cast

import torch

from blackjack.analysis import SELECTED_BET_VOCABULARY, BetAction
from blackjack.dataset import (
    InsuranceToken,
    PlayToken,
    StructureToken,
    encode_bet_input,
    encode_decision_input,
)
from blackjack.engine import (
    FIXED_RULES,
    BlackjackRound,
    Card,
    InsuranceAction,
    ModelContext,
    PlayerAction,
    Rank,
    RoundPhase,
    Shoe,
    ShoeReplay,
)
from blackjack.oracle import CardValue, Composition
from blackjack.training.compare import load_model_artifact
from blackjack.training.hilo import (
    HiLoBetRamp,
    floored_true_count,
    hi_lo_play_action,
)
from blackjack.training.run import TrainingDevice, resolve_device
from blackjack.training.vocabulary import (
    BLACKJACK_VOCABULARY,
    BlackjackVocabulary,
)

_INITIAL_CARD_COUNT = FIXED_RULES.decks * 52
_BET_FRACTIONS = {
    token.token: token.bankroll_fraction
    for token in SELECTED_BET_VOCABULARY.tokens
}
_PLAY_ACTIONS = {
    PlayToken.HIT.value: PlayerAction.HIT,
    PlayToken.STAND.value: PlayerAction.STAND,
    PlayToken.DOUBLE.value: PlayerAction.DOUBLE,
    PlayToken.SPLIT.value: PlayerAction.SPLIT,
    PlayToken.SURRENDER.value: PlayerAction.SURRENDER,
}
_INSURANCE_ACTIONS = {
    InsuranceToken.TAKE.value: InsuranceAction.TAKE,
    InsuranceToken.DECLINE.value: InsuranceAction.DECLINE,
}

type ProgressCallback = Callable[
    [EvaluationPolicyName, int, int, int, float, float],
    None,
]


class EvaluationPolicyName(StrEnum):
    TRANSFORMER = "transformer"
    HI_LO = "hi-lo"


class RuntimePolicy(Protocol):
    """A policy that sees only the same public information as a player."""

    @property
    def name(self) -> EvaluationPolicyName: ...

    def bet(self, prior_history: tuple[Card, ...]) -> BetAction: ...

    def insurance(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> InsuranceAction: ...

    def play(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> PlayerAction: ...


@dataclass(frozen=True, slots=True)
class EvaluationShoe:
    shoe_id: int
    replay: ShoeReplay


@dataclass(frozen=True, slots=True)
class RoundBankrollRecord:
    shoe_id: int
    round_index: int
    global_round_index: int
    visible_cards_before: int
    penetration: float
    running_count: int
    true_count: int
    public_remaining_counts: tuple[int, ...]
    bet_action: BetAction
    bet_fraction: float
    profit_units: float
    bankroll_before: float
    bankroll_after: float

    @property
    def bankroll_return(self) -> float:
        return self.bankroll_after / self.bankroll_before - 1


@dataclass(frozen=True, slots=True)
class PolicyTrajectory:
    policy: EvaluationPolicyName
    initial_bankroll: float
    rounds: tuple[RoundBankrollRecord, ...]

    @property
    def final_bankroll(self) -> float:
        if not self.rounds:
            return self.initial_bankroll
        return self.rounds[-1].bankroll_after

    @property
    def log_growth(self) -> float:
        return log(self.final_bankroll / self.initial_bankroll)


@dataclass(frozen=True, slots=True)
class InferenceContextStatistics:
    context_length: int
    decision_count: int
    truncated_decision_count: int
    total_history_tokens_dropped: int
    maximum_original_length: int
    maximum_tokens_dropped: int


@dataclass(frozen=True, slots=True)
class PairedBankrollEvaluation:
    corpus: str
    shoe_count: int
    shoe_start: int
    simulation_seed: int | None
    transformer: PolicyTrajectory
    hi_lo: PolicyTrajectory
    transformer_context: InferenceContextStatistics | None = None

    def __post_init__(self) -> None:
        if not self.corpus:
            raise ValueError("bankroll evaluation needs a corpus description")
        if self.shoe_count <= 0:
            raise ValueError("bankroll evaluation needs at least one shoe")
        if self.shoe_start < 0:
            raise ValueError("bankroll evaluation shoe start cannot be negative")


@dataclass(frozen=True, slots=True)
class HiLoRuntimePolicy:
    bet_ramp: HiLoBetRamp = field(default_factory=HiLoBetRamp)

    @property
    def name(self) -> EvaluationPolicyName:
        return EvaluationPolicyName.HI_LO

    def bet(self, prior_history: tuple[Card, ...]) -> BetAction:
        return self.bet_ramp.action(_true_count(prior_history))

    def insurance(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> InsuranceAction:
        visible = (*prior_history, *context.history, *context.current_hand)
        visible = (*visible, context.dealer_upcard)
        return (
            InsuranceAction.TAKE
            if _true_count(visible) >= 3
            else InsuranceAction.DECLINE
        )

    def play(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> PlayerAction:
        visible = (*prior_history, *context.history, *context.current_hand)
        visible = (*visible, context.dealer_upcard)
        return hi_lo_play_action(
            _card_values(context.current_hand),
            CardValue.from_card(context.dealer_upcard),
            context.legal_player_actions,
            _true_count(visible),
        )


class TransformerRuntimePolicy:
    """Run the retained transformer against live public round contexts."""

    def __init__(
        self,
        artifact_directory: Path,
        *,
        device_selection: TrainingDevice = TrainingDevice.AUTO,
        vocabulary: BlackjackVocabulary = BLACKJACK_VOCABULARY,
        batch_size: int = 128,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("runtime inference batch size must be positive")
        self._vocabulary = vocabulary
        self._device = resolve_device(device_selection)
        self._batch_size = batch_size
        self._model = load_model_artifact(
            artifact_directory,
            vocabulary,
        ).to(self._device)
        self._decision_count = 0
        self._truncated_decision_count = 0
        self._total_history_tokens_dropped = 0
        self._maximum_original_length = 0
        self._maximum_tokens_dropped = 0

    @property
    def name(self) -> EvaluationPolicyName:
        return EvaluationPolicyName.TRANSFORMER

    @property
    def context_statistics(self) -> InferenceContextStatistics:
        return InferenceContextStatistics(
            context_length=self._model.configuration.context_length,
            decision_count=self._decision_count,
            truncated_decision_count=self._truncated_decision_count,
            total_history_tokens_dropped=self._total_history_tokens_dropped,
            maximum_original_length=self._maximum_original_length,
            maximum_tokens_dropped=self._maximum_tokens_dropped,
        )

    def bet(self, prior_history: tuple[Card, ...]) -> BetAction:
        legal = tuple(action.value for action in BetAction)
        return BetAction(self._predict(encode_bet_input(prior_history), legal))

    def insurance(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> InsuranceAction:
        legal = tuple(token.value for token in InsuranceToken)
        prediction = self._predict(
            encode_decision_input(prior_history, context),
            legal,
        )
        return _INSURANCE_ACTIONS[prediction]

    def play(
        self,
        prior_history: tuple[Card, ...],
        context: ModelContext,
    ) -> PlayerAction:
        legal = tuple(
            _play_token(action) for action in context.legal_player_actions
        )
        prediction = self._predict(
            encode_decision_input(prior_history, context),
            legal,
        )
        return _PLAY_ACTIONS[prediction]

    def _predict(
        self,
        tokens: tuple[str, ...],
        legal_tokens: tuple[str, ...],
    ) -> str:
        if not legal_tokens:
            raise ValueError("a runtime decision needs a legal token")
        fitted_tokens = self._fit_tokens(tokens)
        token_ids = self._vocabulary.encode(fitted_tokens)
        input_ids = torch.tensor(
            (token_ids,),
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            logits = self._model(input_ids, attention_mask)[0, -1]
        legal_ids = torch.tensor(
            self._vocabulary.encode(legal_tokens),
            dtype=torch.long,
            device=self._device,
        )
        selected = int(legal_ids[logits[legal_ids].argmax()].item())
        return self._vocabulary.token_for(selected)

    def predict_batch(
        self,
        requests: tuple[_RuntimeDecisionRequest, ...],
    ) -> tuple[str, ...]:
        """Predict independent live decisions in efficient padded batches."""

        predictions: list[str] = []
        for start in range(0, len(requests), self._batch_size):
            chunk = requests[start : start + self._batch_size]
            fitted_tokens = tuple(
                self._fit_tokens(request.tokens) for request in chunk
            )
            maximum_length = max(len(tokens) for tokens in fitted_tokens)
            input_ids = torch.full(
                (len(chunk), maximum_length),
                self._vocabulary.pad_id,
                dtype=torch.long,
                device=self._device,
            )
            attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            positions = torch.empty(
                len(chunk),
                dtype=torch.long,
                device=self._device,
            )
            for row, tokens in enumerate(fitted_tokens):
                encoded = self._vocabulary.encode(tokens)
                length = len(encoded)
                input_ids[row, :length] = torch.tensor(
                    encoded,
                    dtype=torch.long,
                    device=self._device,
                )
                attention_mask[row, :length] = True
                positions[row] = length - 1
            with torch.no_grad():
                all_logits = self._model(input_ids, attention_mask)
            row_indices = torch.arange(len(chunk), device=self._device)
            decision_logits = all_logits[row_indices, positions]
            for row, request in enumerate(chunk):
                legal_ids = torch.tensor(
                    self._vocabulary.encode(request.legal_tokens),
                    dtype=torch.long,
                    device=self._device,
                )
                selected = int(
                    legal_ids[
                        decision_logits[row, legal_ids].argmax()
                    ].item()
                )
                predictions.append(self._vocabulary.token_for(selected))
        return tuple(predictions)

    def _fit_tokens(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        fitted, dropped = fit_runtime_context(
            tokens,
            self._model.configuration.context_length,
        )
        original_length = len(tokens)
        self._decision_count += 1
        self._maximum_original_length = max(
            self._maximum_original_length,
            original_length,
        )
        if dropped:
            self._truncated_decision_count += 1
            self._total_history_tokens_dropped += dropped
            self._maximum_tokens_dropped = max(
                self._maximum_tokens_dropped,
                dropped,
            )
        return fitted


class _RuntimeDecisionKind(StrEnum):
    BET = "bet"
    INSURANCE = "insurance"
    PLAY = "play"


@dataclass(frozen=True, slots=True)
class _RuntimeDecisionRequest:
    kind: _RuntimeDecisionKind
    tokens: tuple[str, ...]
    legal_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("runtime decision input cannot be empty")
        if not self.legal_tokens:
            raise ValueError("runtime decision needs legal tokens")


def fit_runtime_context(
    tokens: tuple[str, ...],
    maximum_length: int,
) -> tuple[tuple[str, ...], int]:
    """Fit a live decision by removing only its oldest history cards."""

    if maximum_length <= 0:
        raise ValueError("maximum context length must be positive")
    if not tokens or tokens[0] != StructureToken.HISTORY.value:
        raise ValueError("decision input must begin with the history marker")
    if tokens[-1] not in {
        StructureToken.BET_QUERY.value,
        StructureToken.PLAY_QUERY.value,
        StructureToken.INSURANCE_QUERY.value,
    }:
        raise ValueError("decision input must end with a query marker")
    try:
        history_end = tokens.index(StructureToken.CURRENT_HAND.value, 1)
    except ValueError:
        history_end = len(tokens) - 1
    history_length = history_end - 1
    fixed_length = len(tokens) - history_length
    if fixed_length > maximum_length:
        raise ValueError(
            "decision structure and current state exceed the context length"
        )
    dropped = max(0, len(tokens) - maximum_length)
    return (tokens[:1] + tokens[1 + dropped :], dropped)


@dataclass(frozen=True, slots=True)
class _ActiveRound:
    visible_cards_before: int
    running_count: int
    true_count: int
    public_remaining_counts: tuple[int, ...]
    bet_action: BetAction


@dataclass(frozen=True, slots=True)
class _UnstitchedRound:
    shoe_id: int
    round_index: int
    visible_cards_before: int
    running_count: int
    true_count: int
    public_remaining_counts: tuple[int, ...]
    bet_action: BetAction
    profit_units: float


@dataclass(slots=True)
class _ShoeWorld:
    evaluation_shoe: EvaluationShoe
    shoe: Shoe
    prior_history: tuple[Card, ...] = ()
    round_index: int = 0
    game: BlackjackRound | None = None
    active_round: _ActiveRound | None = None
    outcomes: list[_UnstitchedRound] = field(
        default_factory=lambda: list[_UnstitchedRound]()
    )

    @property
    def finished(self) -> bool:
        return self.game is None and self.shoe.reached_cut_card


def load_evaluation_shoes(
    manifest_path: Path,
    *,
    split: str = "validation",
) -> tuple[EvaluationShoe, ...]:
    """Read only the requested shoe replays from a dataset manifest."""

    if split != "validation":
        raise ValueError("bankroll model selection must use validation only")
    raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(raw, "manifest")
    raw_shoes = _list(_required(manifest, "shoes"), "shoes")
    shoes: list[EvaluationShoe] = []
    for raw_shoe in raw_shoes:
        shoe = _mapping(raw_shoe, "shoe")
        shoe_split = _string(_required(shoe, "split"), "shoe split")
        if shoe_split != split:
            continue
        ranks = tuple(
            Rank(_string(item, "card rank"))
            for item in _list(_required(shoe, "cards"), "cards")
        )
        replay = ShoeReplay(
            cards=tuple(Card(rank) for rank in ranks),
            cut_card_position=_integer(
                _required(shoe, "cut_card_position"),
                "cut card position",
            ),
        )
        shoes.append(
            EvaluationShoe(
                shoe_id=_integer(_required(shoe, "shoe_id"), "shoe ID"),
                replay=replay,
            )
        )
    if not shoes:
        raise ValueError(f"manifest contains no {split} shoes")
    return tuple(shoes)


def generate_evaluation_shoes(
    shoe_count: int,
    *,
    seed: int,
    shoe_start: int = 0,
) -> tuple[EvaluationShoe, ...]:
    """Create a fresh reproducible shoe corpus without dataset labels."""

    if shoe_count <= 0:
        raise ValueError("evaluation shoe count must be positive")
    if shoe_start < 0:
        raise ValueError("evaluation shoe start cannot be negative")
    if not 0 <= seed < 2**64:
        raise ValueError("evaluation seed must fit in unsigned 64 bits")
    rng = Random(seed)
    shoe_seeds: list[int] = []
    used: set[int] = set()
    required_seed_count = shoe_start + shoe_count
    while len(shoe_seeds) < required_seed_count:
        shoe_seed = rng.getrandbits(63)
        if shoe_seed in used:
            continue
        used.add(shoe_seed)
        shoe_seeds.append(shoe_seed)
    selected_seeds = shoe_seeds[shoe_start:required_seed_count]
    return tuple(
        EvaluationShoe(
            shoe_id=shoe_start + offset,
            replay=Shoe.shuffled(shoe_seed, FIXED_RULES).replay,
        )
        for offset, shoe_seed in enumerate(selected_seeds)
    )


def simulate_policy(
    shoes: tuple[EvaluationShoe, ...],
    policy: RuntimePolicy,
    *,
    initial_bankroll: float = 100.0,
    progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> PolicyTrajectory:
    """Carry one bankroll through a deterministic sequence of shoe replays."""

    if initial_bankroll <= 0:
        raise ValueError("initial bankroll must be positive")
    if progress_interval <= 0:
        raise ValueError("progress interval must be positive")
    bankroll = initial_bankroll
    global_round_index = 0
    records: list[RoundBankrollRecord] = []
    started = perf_counter()
    total_shoes = len(shoes)
    for shoe_offset, evaluation_shoe in enumerate(shoes, start=1):
        shoe = Shoe.from_replay(evaluation_shoe.replay)
        prior_history: tuple[Card, ...] = ()
        round_index = 0
        while not shoe.reached_cut_card:
            visible_before = len(prior_history)
            running_count = _running_count(prior_history)
            true_count = _true_count(prior_history)
            remaining_counts = _remaining_counts(prior_history)
            bet_action = policy.bet(prior_history)
            bet_fraction = _BET_FRACTIONS[bet_action]
            game = BlackjackRound(shoe, Fraction(1), FIXED_RULES)
            while game.public_state.phase is not RoundPhase.SETTLED:
                state = game.public_state
                context = state.model_context
                if context is None:
                    raise AssertionError(
                        "an unsettled round needs a public model context"
                    )
                if state.phase is RoundPhase.INSURANCE:
                    game.decide_insurance(policy.insurance(prior_history, context))
                elif state.phase is RoundPhase.PLAYER_ACTIONS:
                    game.act(policy.play(prior_history, context))
                else:
                    raise AssertionError(
                        f"unexpected decision phase: {state.phase}"
                    )
            settlement = game.public_state.settlement
            if settlement is None:
                raise AssertionError("a settled round needs a settlement")
            profit_units = float(settlement.total_profit)
            bankroll_before = bankroll
            bankroll *= 1 + bet_fraction * profit_units
            if bankroll <= 0:
                raise AssertionError("the fixed bet fractions exhausted bankroll")
            records.append(
                RoundBankrollRecord(
                    shoe_id=evaluation_shoe.shoe_id,
                    round_index=round_index,
                    global_round_index=global_round_index,
                    visible_cards_before=visible_before,
                    penetration=visible_before / _INITIAL_CARD_COUNT,
                    running_count=running_count,
                    true_count=true_count,
                    public_remaining_counts=remaining_counts,
                    bet_action=bet_action,
                    bet_fraction=bet_fraction,
                    profit_units=profit_units,
                    bankroll_before=bankroll_before,
                    bankroll_after=bankroll,
                )
            )
            prior_history = (
                *prior_history,
                *game.public_state.visible_card_history,
            )
            round_index += 1
            global_round_index += 1
        if (
            progress is not None
            and (
                shoe_offset % progress_interval == 0
                or shoe_offset == total_shoes
            )
        ):
            progress(
                policy.name,
                shoe_offset,
                total_shoes,
                global_round_index,
                shoe_offset / total_shoes,
                perf_counter() - started,
            )
    return PolicyTrajectory(
        policy=policy.name,
        initial_bankroll=initial_bankroll,
        rounds=tuple(records),
    )


def simulate_transformer_policy(
    shoes: tuple[EvaluationShoe, ...],
    policy: TransformerRuntimePolicy,
    *,
    initial_bankroll: float = 100.0,
    progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> PolicyTrajectory:
    """Advance many shoes together so live transformer decisions are batched."""

    if initial_bankroll <= 0:
        raise ValueError("initial bankroll must be positive")
    if progress_interval <= 0:
        raise ValueError("progress interval must be positive")
    worlds = [
        _ShoeWorld(
            evaluation_shoe=evaluation_shoe,
            shoe=Shoe.from_replay(evaluation_shoe.replay),
        )
        for evaluation_shoe in shoes
    ]
    started = perf_counter()
    next_progress = min(progress_interval, len(worlds))
    while any(not world.finished for world in worlds):
        pending_worlds: list[_ShoeWorld] = []
        requests: list[_RuntimeDecisionRequest] = []
        for world in worlds:
            request = _next_request(world)
            if request is not None:
                pending_worlds.append(world)
                requests.append(request)
        if not requests:
            if any(not world.finished for world in worlds):
                raise AssertionError("live shoe worlds made no progress")
            break
        predictions = policy.predict_batch(tuple(requests))
        if len(predictions) != len(requests):
            raise AssertionError("runtime prediction count does not match requests")
        for world, request, prediction in zip(
            pending_worlds,
            requests,
            predictions,
            strict=True,
        ):
            _apply_runtime_prediction(world, request.kind, prediction)
        completed = sum(world.finished for world in worlds)
        work_fraction = sum(
            min(
                world.shoe.dealt_count
                / max(
                    world.evaluation_shoe.replay.cut_card_position - 1,
                    1,
                ),
                1.0,
            )
            for world in worlds
        ) / len(worlds)
        equivalent_completed = round(work_fraction * len(worlds))
        if progress is not None and (
            equivalent_completed >= next_progress
            or completed == len(worlds)
        ):
            progress(
                policy.name,
                completed,
                len(worlds),
                sum(len(world.outcomes) for world in worlds),
                work_fraction,
                perf_counter() - started,
            )
            while next_progress <= equivalent_completed:
                next_progress += progress_interval
    if progress is not None:
        progress(
            policy.name,
            len(worlds),
            len(worlds),
            sum(len(world.outcomes) for world in worlds),
            1.0,
            perf_counter() - started,
        )
    unstitched = tuple(
        outcome
        for world in worlds
        for outcome in world.outcomes
    )
    return _stitch_trajectory(
        EvaluationPolicyName.TRANSFORMER,
        initial_bankroll,
        unstitched,
    )


def evaluate_paired_bankrolls(
    shoes: tuple[EvaluationShoe, ...],
    transformer: RuntimePolicy,
    hi_lo: RuntimePolicy,
    *,
    initial_bankroll: float = 100.0,
    simulation_seed: int | None = None,
    shoe_start: int = 0,
    progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> PairedBankrollEvaluation:
    """Evaluate two policies on independent cursors over identical shoe replays."""

    if transformer.name is not EvaluationPolicyName.TRANSFORMER:
        raise ValueError("transformer policy has the wrong name")
    if hi_lo.name is not EvaluationPolicyName.HI_LO:
        raise ValueError("Hi-Lo policy has the wrong name")
    transformer_trajectory = (
        simulate_transformer_policy(
            shoes,
            transformer,
            initial_bankroll=initial_bankroll,
            progress=progress,
            progress_interval=progress_interval,
        )
        if isinstance(transformer, TransformerRuntimePolicy)
        else simulate_policy(
            shoes,
            transformer,
            initial_bankroll=initial_bankroll,
            progress=progress,
            progress_interval=progress_interval,
        )
    )
    return PairedBankrollEvaluation(
        corpus="fresh deterministic simulation shoes",
        shoe_count=len(shoes),
        shoe_start=shoe_start,
        simulation_seed=simulation_seed,
        transformer=transformer_trajectory,
        hi_lo=simulate_policy(
            shoes,
            hi_lo,
            initial_bankroll=initial_bankroll,
            progress=progress,
            progress_interval=progress_interval,
        ),
        transformer_context=(
            transformer.context_statistics
            if isinstance(transformer, TransformerRuntimePolicy)
            else None
        ),
    )


def _next_request(
    world: _ShoeWorld,
) -> _RuntimeDecisionRequest | None:
    game = world.game
    if game is None:
        if world.shoe.reached_cut_card:
            return None
        return _RuntimeDecisionRequest(
            kind=_RuntimeDecisionKind.BET,
            tokens=encode_bet_input(world.prior_history),
            legal_tokens=tuple(action.value for action in BetAction),
        )
    state = game.public_state
    if state.phase is RoundPhase.SETTLED:
        _finish_world_round(world)
        return _next_request(world)
    context = state.model_context
    if context is None:
        raise AssertionError("an unsettled round needs a public model context")
    if state.phase is RoundPhase.INSURANCE:
        return _RuntimeDecisionRequest(
            kind=_RuntimeDecisionKind.INSURANCE,
            tokens=encode_decision_input(world.prior_history, context),
            legal_tokens=tuple(token.value for token in InsuranceToken),
        )
    if state.phase is RoundPhase.PLAYER_ACTIONS:
        return _RuntimeDecisionRequest(
            kind=_RuntimeDecisionKind.PLAY,
            tokens=encode_decision_input(world.prior_history, context),
            legal_tokens=tuple(
                _play_token(action) for action in context.legal_player_actions
            ),
        )
    raise AssertionError(f"unexpected decision phase: {state.phase}")


def _apply_runtime_prediction(
    world: _ShoeWorld,
    kind: _RuntimeDecisionKind,
    prediction: str,
) -> None:
    if kind is _RuntimeDecisionKind.BET:
        if world.game is not None or world.active_round is not None:
            raise AssertionError("bet prediction arrived during an active round")
        action = BetAction(prediction)
        world.active_round = _ActiveRound(
            visible_cards_before=len(world.prior_history),
            running_count=_running_count(world.prior_history),
            true_count=_true_count(world.prior_history),
            public_remaining_counts=_remaining_counts(world.prior_history),
            bet_action=action,
        )
        world.game = BlackjackRound(world.shoe, Fraction(1), FIXED_RULES)
        return
    if world.game is None:
        raise AssertionError("round decision arrived without an active game")
    if kind is _RuntimeDecisionKind.INSURANCE:
        world.game.decide_insurance(_INSURANCE_ACTIONS[prediction])
    else:
        world.game.act(_PLAY_ACTIONS[prediction])


def _finish_world_round(world: _ShoeWorld) -> None:
    if world.game is None or world.active_round is None:
        raise AssertionError("cannot finish an incomplete world round")
    state = world.game.public_state
    settlement = state.settlement
    if settlement is None:
        raise AssertionError("a settled round needs a settlement")
    active = world.active_round
    world.outcomes.append(
        _UnstitchedRound(
            shoe_id=world.evaluation_shoe.shoe_id,
            round_index=world.round_index,
            visible_cards_before=active.visible_cards_before,
            running_count=active.running_count,
            true_count=active.true_count,
            public_remaining_counts=active.public_remaining_counts,
            bet_action=active.bet_action,
            profit_units=float(settlement.total_profit),
        )
    )
    world.prior_history = (*world.prior_history, *state.visible_card_history)
    world.round_index += 1
    world.game = None
    world.active_round = None


def _stitch_trajectory(
    policy: EvaluationPolicyName,
    initial_bankroll: float,
    outcomes: tuple[_UnstitchedRound, ...],
) -> PolicyTrajectory:
    bankroll = initial_bankroll
    records: list[RoundBankrollRecord] = []
    for global_round_index, outcome in enumerate(outcomes):
        bet_fraction = _BET_FRACTIONS[outcome.bet_action]
        bankroll_before = bankroll
        bankroll *= 1 + bet_fraction * outcome.profit_units
        records.append(
            RoundBankrollRecord(
                shoe_id=outcome.shoe_id,
                round_index=outcome.round_index,
                global_round_index=global_round_index,
                visible_cards_before=outcome.visible_cards_before,
                penetration=(
                    outcome.visible_cards_before / _INITIAL_CARD_COUNT
                ),
                running_count=outcome.running_count,
                true_count=outcome.true_count,
                public_remaining_counts=outcome.public_remaining_counts,
                bet_action=outcome.bet_action,
                bet_fraction=bet_fraction,
                profit_units=outcome.profit_units,
                bankroll_before=bankroll_before,
                bankroll_after=bankroll,
            )
        )
    return PolicyTrajectory(
        policy=policy,
        initial_bankroll=initial_bankroll,
        rounds=tuple(records),
    )


def evaluation_data(evaluation: PairedBankrollEvaluation) -> dict[str, object]:
    paired_advantage = _paired_log_growth_advantage(evaluation)
    transformer_context = evaluation.transformer_context
    return {
        "methodology": {
            "corpus": evaluation.corpus,
            "simulation_seed": evaluation.simulation_seed,
            "shoe_start": evaluation.shoe_start,
            "shoe_pairing": "same replay order and cut-card position",
            "round_pairing": (
                "card allocation may diverge after policies choose different actions"
            ),
            "bankroll_staking": (
                "selected fraction of current bankroll, settled at exact engine return"
            ),
            "transformer_context_overflow": (
                "when a live input exceeds the trained context window, remove "
                "only the minimum number of oldest visible-history card tokens; "
                "preserve the history marker, complete current hand, dealer "
                "upcard, structure markers, and query"
            ),
        },
        "transformer_context_statistics": (
            None
            if transformer_context is None
            else {
                "context_length": transformer_context.context_length,
                "decision_count": transformer_context.decision_count,
                "truncated_decision_count": (
                    transformer_context.truncated_decision_count
                ),
                "total_history_tokens_dropped": (
                    transformer_context.total_history_tokens_dropped
                ),
                "maximum_original_length": (
                    transformer_context.maximum_original_length
                ),
                "maximum_tokens_dropped": (
                    transformer_context.maximum_tokens_dropped
                ),
            }
        ),
        "shoe_count": evaluation.shoe_count,
        "paired_transformer_log_growth_advantage_per_shoe": (
            _estimate_data(paired_advantage)
        ),
        "breakdowns": {
            "penetration": _comparison_breakdown(
                evaluation,
                _penetration_bin,
            ),
            "true_count": _comparison_breakdown(
                evaluation,
                _true_count_bin,
            ),
            "public_high_card_share": _comparison_breakdown(
                evaluation,
                _high_card_share_bin,
            ),
        },
        "policies": {
            trajectory.policy.value: _trajectory_data(trajectory)
            for trajectory in (
                evaluation.transformer,
                evaluation.hi_lo,
            )
        },
    }


def write_evaluation(
    evaluation: PairedBankrollEvaluation,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(evaluation_data(evaluation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def _trajectory_data(trajectory: PolicyTrajectory) -> dict[str, object]:
    round_returns = _shoe_means(
        trajectory.rounds,
        lambda record: record.bankroll_return,
    )
    unit_returns = _shoe_means(
        trajectory.rounds,
        lambda record: record.profit_units,
    )
    log_growth_per_100_rounds = _shoe_means(
        trajectory.rounds,
        lambda record: 100
        * log(record.bankroll_after / record.bankroll_before),
    )
    return {
        "initial_bankroll": trajectory.initial_bankroll,
        "final_bankroll": trajectory.final_bankroll,
        "log_growth": trajectory.log_growth,
        "round_count": len(trajectory.rounds),
        "mean_bankroll_return_per_round": _estimate_data(
            _mean_estimate(round_returns)
        ),
        "mean_profit_units_per_round": _estimate_data(
            _mean_estimate(unit_returns)
        ),
        "mean_log_growth_per_100_rounds": _estimate_data(
            _mean_estimate(log_growth_per_100_rounds)
        ),
        "trajectory": [
            {
                "global_round_index": record.global_round_index,
                "bankroll_after": record.bankroll_after,
            }
            for record in _downsampled_records(trajectory.rounds)
        ],
    }


def _downsampled_records(
    records: tuple[RoundBankrollRecord, ...],
    *,
    maximum_points: int = 5_000,
) -> tuple[RoundBankrollRecord, ...]:
    if len(records) <= maximum_points:
        return records
    step = len(records) // maximum_points
    sampled = records[step - 1 :: step]
    if sampled[-1] is records[-1]:
        return sampled
    return (*sampled, records[-1])


def _comparison_breakdown(
    evaluation: PairedBankrollEvaluation,
    classifier: Callable[[RoundBankrollRecord], str],
) -> dict[str, object]:
    return {
        trajectory.policy.value: _trajectory_breakdown(
            trajectory,
            classifier,
        )
        for trajectory in (evaluation.transformer, evaluation.hi_lo)
    }


def _trajectory_breakdown(
    trajectory: PolicyTrajectory,
    classifier: Callable[[RoundBankrollRecord], str],
) -> list[dict[str, object]]:
    grouped: dict[str, list[RoundBankrollRecord]] = defaultdict(list)
    for record in trajectory.rounds:
        grouped[classifier(record)].append(record)
    return [
        {
            "bin": name,
            "round_count": len(records),
            "mean_log_growth_per_100_rounds": _estimate_data(
                _mean_estimate(
                    _shoe_means(
                        tuple(records),
                        lambda record: 100
                        * log(
                            record.bankroll_after
                            / record.bankroll_before
                        ),
                    )
                )
            ),
        }
        for name, records in grouped.items()
    ]


def _shoe_means(
    records: tuple[RoundBankrollRecord, ...],
    value: Callable[[RoundBankrollRecord], float],
) -> tuple[float, ...]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.shoe_id].append(value(record))
    return tuple(
        sum(values) / len(values)
        for _, values in sorted(grouped.items())
    )


def _penetration_bin(record: RoundBankrollRecord) -> str:
    penetration = record.penetration
    if penetration < 0.2:
        return "0%-20%"
    if penetration < 0.4:
        return "20%-40%"
    if penetration < 0.6:
        return "40%-60%"
    return "60%-80%"


def _true_count_bin(record: RoundBankrollRecord) -> str:
    true_count = record.true_count
    if true_count <= -3:
        return "<=-3"
    if true_count <= -1:
        return "-2 to -1"
    if true_count <= 1:
        return "0 to 1"
    if true_count <= 3:
        return "2 to 3"
    return ">=4"


def _high_card_share_bin(record: RoundBankrollRecord) -> str:
    counts = record.public_remaining_counts
    remaining = sum(counts)
    high_share = (counts[0] + counts[-1]) / remaining
    if high_share < 0.37:
        return "<37%"
    if high_share < 0.385:
        return "37%-38.5%"
    if high_share < 0.4:
        return "38.5%-40%"
    return ">=40%"


@dataclass(frozen=True, slots=True)
class MeanEstimate:
    mean: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    sample_count: int


def _mean_estimate(values: tuple[float, ...]) -> MeanEstimate:
    if not values:
        raise ValueError("an estimate needs at least one value")
    sample_count = len(values)
    mean = sum(values) / sample_count
    if sample_count == 1:
        standard_error = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (
            sample_count - 1
        )
        standard_error = sqrt(variance / sample_count)
    margin = 1.96 * standard_error
    return MeanEstimate(
        mean=mean,
        standard_error=standard_error,
        confidence_interval_95=(mean - margin, mean + margin),
        sample_count=sample_count,
    )


def _paired_log_growth_advantage(
    evaluation: PairedBankrollEvaluation,
) -> MeanEstimate:
    transformer = _log_growth_by_shoe(evaluation.transformer)
    hi_lo = _log_growth_by_shoe(evaluation.hi_lo)
    if transformer.keys() != hi_lo.keys():
        raise AssertionError("paired policies evaluated different shoes")
    differences = tuple(
        transformer[shoe_id] - hi_lo[shoe_id]
        for shoe_id in sorted(transformer)
    )
    return _mean_estimate(differences)


def _log_growth_by_shoe(
    trajectory: PolicyTrajectory,
) -> dict[int, float]:
    growth: dict[int, float] = {}
    for record in trajectory.rounds:
        growth[record.shoe_id] = growth.get(record.shoe_id, 0.0) + log(
            record.bankroll_after / record.bankroll_before
        )
    return growth


def _estimate_data(estimate: MeanEstimate) -> dict[str, object]:
    return {
        "mean": estimate.mean,
        "standard_error": estimate.standard_error,
        "confidence_interval_95": list(estimate.confidence_interval_95),
        "sample_count": estimate.sample_count,
    }


def _card_values(cards: tuple[Card, ...]) -> tuple[CardValue, ...]:
    return tuple(CardValue.from_card(card) for card in cards)


def _remaining_counts(cards: tuple[Card, ...]) -> tuple[int, ...]:
    return Composition.full_shoe(FIXED_RULES.decks).remove_cards(cards).counts


def _running_count(cards: tuple[Card, ...]) -> int:
    return sum(
        1
        if card.value in range(2, 7)
        else -1
        if card.rank is Rank.ACE or card.is_ten_valued
        else 0
        for card in cards
    )


def _true_count(cards: tuple[Card, ...]) -> int:
    return floored_true_count(_card_values(cards))


def _play_token(action: PlayerAction) -> str:
    return {
        PlayerAction.HIT: PlayToken.HIT.value,
        PlayerAction.STAND: PlayToken.STAND.value,
        PlayerAction.DOUBLE: PlayToken.DOUBLE.value,
        PlayerAction.SPLIT: PlayToken.SPLIT.value,
        PlayerAction.SURRENDER: PlayToken.SURRENDER.value,
    }[action]


def _required(data: dict[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"missing JSON field: {key}")
    return data[key]


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    untyped = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in untyped):
        raise ValueError(f"{field} keys must be strings")
    return {str(key): item for key, item in untyped.items()}


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay held-out validation shoes through the transformer and Hi-Lo."
        )
    )
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--shoe-count", type=int, default=1_000)
    parser.add_argument("--shoe-start", type=int, default=0)
    parser.add_argument("--simulation-seed", type=int, default=20260801)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--progress-every-shoes", type=int, default=500)
    parser.add_argument("--initial-bankroll", type=float, default=100.0)
    parser.add_argument("--chart-path", type=Path)
    parser.add_argument(
        "--device",
        type=TrainingDevice,
        choices=tuple(TrainingDevice),
        default=TrainingDevice.AUTO,
    )
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    shoes = generate_evaluation_shoes(
        arguments.shoe_count,
        seed=arguments.simulation_seed,
        shoe_start=arguments.shoe_start,
    )
    transformer = TransformerRuntimePolicy(
        arguments.artifact_directory,
        device_selection=arguments.device,
        batch_size=arguments.inference_batch_size,
    )
    evaluation = evaluate_paired_bankrolls(
        shoes,
        transformer,
        HiLoRuntimePolicy(),
        initial_bankroll=arguments.initial_bankroll,
        simulation_seed=arguments.simulation_seed,
        shoe_start=arguments.shoe_start,
        progress=_print_progress,
        progress_interval=arguments.progress_every_shoes,
    )
    write_evaluation(evaluation, arguments.output_path)
    if arguments.chart_path is not None:
        from blackjack.analysis.bankroll_svg import write_bankroll_chart

        write_bankroll_chart(evaluation, arguments.chart_path)
    print(
        json.dumps(
            {
                "output_path": str(arguments.output_path),
                "shoe_count": evaluation.shoe_count,
                "transformer_rounds": len(evaluation.transformer.rounds),
                "transformer_final_bankroll": (
                    evaluation.transformer.final_bankroll
                ),
                "hi_lo_rounds": len(evaluation.hi_lo.rounds),
                "hi_lo_final_bankroll": evaluation.hi_lo.final_bankroll,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_progress(
    policy: EvaluationPolicyName,
    completed_shoes: int,
    total_shoes: int,
    completed_rounds: int,
    work_fraction: float,
    elapsed_seconds: float,
) -> None:
    eta_seconds = elapsed_seconds * (
        (1 - work_fraction) / max(work_fraction, 1e-9)
    )
    percentage = 100 * work_fraction
    print(
        f"[{policy.value} work] {percentage:5.1f}% | "
        f"{completed_shoes:,}/{total_shoes:,} shoes reached cut | "
        f"{completed_rounds:,} rounds | "
        f"elapsed {_duration(elapsed_seconds)} | "
        f"ETA {_duration(eta_seconds)}",
        flush=True,
    )


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remaining_seconds:02d}s"
    return f"{minutes:d}m {remaining_seconds:02d}s"


if __name__ == "__main__":
    main()
