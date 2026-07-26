"""Deterministic complete-shoe generation with leakage-safe dataset splits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import floor
from random import Random
from time import perf_counter
from typing import Protocol

from blackjack.dataset.labeling import (
    LabeledDecision,
    ProductionDatasetOracle,
    insurance_action_for_token,
    player_action_for_token,
)
from blackjack.dataset.records import (
    DatasetBundle,
    DatasetConfiguration,
    DatasetManifest,
    DatasetSplit,
    DecisionExample,
    DecisionKind,
    ShoeManifest,
)
from blackjack.dataset.tokens import encode_bet_input, encode_decision_input
from blackjack.engine import (
    BlackjackRound,
    Card,
    HandSnapshot,
    ModelContext,
    PlayerAction,
    RoundPhase,
    Shoe,
    ShoeReplay,
)
from blackjack.oracle import (
    Composition,
    OracleHand,
    PeekCondition,
    ResolvedHand,
    RoundPlayerSituation,
    cards_to_values,
)

SCHEMA_VERSION = 4


class DatasetOracle(Protocol):
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


@dataclass(frozen=True, slots=True)
class PreparedShoe:
    manifest: ShoeManifest
    replay: ShoeReplay


class CheckpointMismatchError(RuntimeError):
    """Raised when cached rows do not describe the replayed decision state."""


@dataclass(frozen=True, slots=True)
class ShoeGenerationResult:
    examples: tuple[DecisionExample, ...]
    new_decisions: int
    cached_decisions: int
    completed: bool


type DecisionCallback = Callable[[DecisionExample, bool, float], None]


def generate_dataset(
    configuration: DatasetConfiguration,
    *,
    oracle: DatasetOracle | None = None,
) -> DatasetBundle:
    """Generate every decision along deterministic exploratory shoe trajectories."""

    labeler: DatasetOracle = oracle if oracle is not None else ProductionDatasetOracle()
    prepared = prepare_shoes(configuration)
    identifier = dataset_id(configuration, prepared)
    examples: list[DecisionExample] = []

    for item in prepared:
        result = generate_shoe_examples(
            item,
            identifier,
            configuration,
            labeler,
        )
        if not result.completed:
            raise AssertionError("unbounded in-memory generation must finish")
        examples.extend(result.examples)

    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        dataset_id=identifier,
        configuration=configuration,
        shoes=tuple(item.manifest for item in prepared),
    )
    return DatasetBundle(manifest=manifest, examples=tuple(examples))


def prepare_shoes(
    configuration: DatasetConfiguration,
) -> tuple[PreparedShoe, ...]:
    shoe_rng = Random(configuration.master_seed)
    seeds: list[int] = []
    used_seeds: set[int] = set()
    while len(seeds) < configuration.shoe_count:
        seed = shoe_rng.getrandbits(63)
        if seed not in used_seeds:
            used_seeds.add(seed)
            seeds.append(seed)

    splits = _assign_splits(configuration)
    prepared: list[PreparedShoe] = []
    for shoe_id, (seed, split) in enumerate(zip(seeds, splits, strict=True)):
        shoe = Shoe.shuffled(seed, configuration.rules)
        replay = shoe.replay
        prepared.append(
            PreparedShoe(
                manifest=ShoeManifest(
                    shoe_id=shoe_id,
                    seed=seed,
                    split=split,
                    cards=tuple(card.rank for card in replay.cards),
                    cut_card_position=replay.cut_card_position,
                ),
                replay=replay,
            )
        )
    return tuple(prepared)


def _assign_splits(
    configuration: DatasetConfiguration,
) -> tuple[DatasetSplit, ...]:
    weights = (
        configuration.train_fraction,
        configuration.validation_fraction,
        configuration.test_fraction,
    )
    raw_counts = tuple(configuration.shoe_count * weight for weight in weights)
    counts = [floor(value) for value in raw_counts]
    leftovers = configuration.shoe_count - sum(counts)
    remainder_order = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[:leftovers]:
        counts[index] += 1
    for empty_index, count in enumerate(counts):
        if count != 0:
            continue
        donor_index = max(range(3), key=counts.__getitem__)
        if counts[donor_index] <= 1:
            raise AssertionError("three shoes must support three non-empty splits")
        counts[donor_index] -= 1
        counts[empty_index] += 1

    assignments = [
        *([DatasetSplit.TRAIN] * counts[0]),
        *([DatasetSplit.VALIDATION] * counts[1]),
        *([DatasetSplit.TEST] * counts[2]),
    ]
    Random(configuration.split_seed).shuffle(assignments)
    return tuple(assignments)


def dataset_id(
    configuration: DatasetConfiguration,
    shoes: tuple[PreparedShoe, ...],
) -> str:
    rules = configuration.rules
    pieces = [
        str(SCHEMA_VERSION),
        str(configuration.master_seed),
        str(configuration.split_seed),
        str(configuration.exploration_seed),
        str(configuration.exploration_probability),
        str(configuration.bet_rollout_seed),
        str(configuration.bet_rollouts),
        str(configuration.play_rollout_seed),
        str(configuration.play_rollouts),
        str(configuration.train_fraction),
        str(configuration.validation_fraction),
        str(configuration.test_fraction),
        ",".join(
            f"{token.token.value}:{token.bankroll_fraction}"
            for token in configuration.bet_vocabulary.tokens
        ),
        configuration.bet_evaluation_method.value,
        configuration.play_evaluation_method.value,
        repr(rules),
    ]
    for shoe in shoes:
        pieces.extend(
            (
                str(shoe.manifest.shoe_id),
                str(shoe.manifest.seed),
                shoe.manifest.split.value,
                str(shoe.manifest.cut_card_position),
                ",".join(rank.value for rank in shoe.manifest.cards),
            )
        )
    digest = sha256("|".join(pieces).encode()).hexdigest()[:16]
    return f"blackjack-decisions-v{SCHEMA_VERSION}-{digest}"


def generate_shoe_examples(
    prepared: PreparedShoe,
    dataset_id: str,
    configuration: DatasetConfiguration,
    oracle: DatasetOracle,
    *,
    cached_examples: tuple[DecisionExample, ...] = (),
    maximum_new_decisions: int | None = None,
    cache_must_complete: bool = False,
    callback: DecisionCallback | None = None,
) -> ShoeGenerationResult:
    if maximum_new_decisions is not None and maximum_new_decisions <= 0:
        raise ValueError("maximum new decisions must be positive")
    shoe = Shoe.from_replay(prepared.replay)
    prior_history: tuple[Card, ...] = ()
    examples: list[DecisionExample] = []
    exploration_rng = Random(
        _shoe_exploration_seed(configuration.exploration_seed, prepared.manifest)
    )
    round_index = 0
    decision_index = 0
    new_decisions = 0
    cached_decisions = 0

    while not shoe.reached_cut_card:
        composition = Composition.full_shoe(configuration.rules.decks).remove_cards(
            prior_history
        )
        unavailable = _unseen_unavailable(
            composition.total,
            shoe.remaining,
        )
        bet_input = encode_bet_input(prior_history)
        cached = _cached_example(
            cached_examples,
            prepared,
            dataset_id,
            round_index,
            decision_index,
            DecisionKind.BET,
            bet_input,
            composition,
            unavailable,
        )
        if cached is None:
            if cache_must_complete:
                raise CheckpointMismatchError(
                    "completed shoe shard ends before the cut card"
                )
            started = perf_counter()
            bet = oracle.label_bet(composition, unavailable)
            example = _example(
                prepared,
                dataset_id,
                round_index,
                decision_index,
                DecisionKind.BET,
                bet_input,
                bet,
                bet.target_token,
            )
            elapsed = perf_counter() - started
            new_decisions += 1
        else:
            bet = LabeledDecision(cached.target_token, cached.metadata)
            example = cached
            elapsed = 0.0
            cached_decisions += 1
        examples.append(example)
        if callback is not None:
            callback(example, cached is not None, elapsed)
        decision_index += 1
        if _reached_limit(new_decisions, maximum_new_decisions):
            return ShoeGenerationResult(
                tuple(examples),
                new_decisions,
                cached_decisions,
                completed=False,
            )

        game = BlackjackRound(shoe, Fraction(1), configuration.rules)
        while game.public_state.phase is not RoundPhase.SETTLED:
            state = game.public_state
            context = state.model_context
            if context is None:
                raise AssertionError("an unsettled decision phase needs model context")
            visible = (*prior_history, *state.visible_card_history)
            composition = Composition.full_shoe(configuration.rules.decks).remove_cards(
                visible
            )
            unavailable = _unseen_unavailable(
                composition.total,
                shoe.remaining,
            )

            situation: RoundPlayerSituation | None = None
            if state.phase is RoundPhase.INSURANCE:
                kind = DecisionKind.INSURANCE
            elif state.phase is RoundPhase.PLAYER_ACTIONS:
                situation = _round_situation(
                    state.player_hands,
                    state.active_hand_index,
                    context,
                    composition,
                    unavailable,
                    configuration,
                )
                kind = DecisionKind.PLAY
            else:
                raise AssertionError(f"unexpected decision phase: {state.phase}")

            decision_input = encode_decision_input(prior_history, context)
            cached = _cached_example(
                cached_examples,
                prepared,
                dataset_id,
                round_index,
                decision_index,
                kind,
                decision_input,
                composition,
                unavailable,
            )
            if cached is None:
                if cache_must_complete:
                    raise CheckpointMismatchError(
                        "completed shoe shard ends before the cut card"
                    )
                started = perf_counter()
                label = (
                    oracle.label_insurance(composition, unavailable)
                    if kind is DecisionKind.INSURANCE
                    else oracle.label_play(
                        _required_situation(situation),
                        state.legal_actions,
                    )
                )
                behavior = _behavior_token(
                    label,
                    configuration.exploration_probability,
                    exploration_rng,
                )
                example = _example(
                    prepared,
                    dataset_id,
                    round_index,
                    decision_index,
                    kind,
                    decision_input,
                    label,
                    behavior,
                )
                elapsed = perf_counter() - started
                new_decisions += 1
            else:
                label = LabeledDecision(cached.target_token, cached.metadata)
                behavior = _behavior_token(
                    label,
                    configuration.exploration_probability,
                    exploration_rng,
                )
                if behavior != cached.behavior_token:
                    raise CheckpointMismatchError(
                        "cached behavior does not match the exploration replay"
                    )
                example = cached
                elapsed = 0.0
                cached_decisions += 1
            examples.append(example)
            if callback is not None:
                callback(example, cached is not None, elapsed)
            decision_index += 1
            _apply_behavior(game, kind, behavior)
            if _reached_limit(new_decisions, maximum_new_decisions):
                return ShoeGenerationResult(
                    tuple(examples),
                    new_decisions,
                    cached_decisions,
                    completed=False,
                )

        prior_history = (*prior_history, *game.public_state.visible_card_history)
        round_index += 1

    if len(cached_examples) > len(examples):
        raise CheckpointMismatchError("checkpoint contains rows beyond the cut card")
    return ShoeGenerationResult(
        tuple(examples),
        new_decisions,
        cached_decisions,
        completed=True,
    )


def _shoe_exploration_seed(
    exploration_seed: int,
    manifest: ShoeManifest,
) -> int:
    payload = f"{exploration_seed}:{manifest.shoe_id}:{manifest.seed}".encode()
    return int.from_bytes(sha256(payload).digest()[:8])


def _reached_limit(current: int, maximum: int | None) -> bool:
    return maximum is not None and current >= maximum


def _cached_example(
    cached_examples: tuple[DecisionExample, ...],
    prepared: PreparedShoe,
    dataset_id: str,
    round_index: int,
    decision_index: int,
    kind: DecisionKind,
    input_tokens: tuple[str, ...],
    composition: Composition,
    unseen_unavailable: int,
) -> DecisionExample | None:
    if decision_index >= len(cached_examples):
        return None
    cached = cached_examples[decision_index]
    expected = (
        cached.schema_version == SCHEMA_VERSION
        and cached.dataset_id == dataset_id
        and cached.shoe_id == prepared.manifest.shoe_id
        and cached.shoe_seed == prepared.manifest.seed
        and cached.split is prepared.manifest.split
        and cached.round_index == round_index
        and cached.decision_index == decision_index
        and cached.kind is kind
        and cached.input_tokens == input_tokens
        and cached.metadata.shoe_composition == composition
        and cached.metadata.unseen_unavailable == unseen_unavailable
    )
    if not expected:
        raise CheckpointMismatchError(
            f"checkpoint diverges at shoe {prepared.manifest.shoe_id}, "
            f"decision {decision_index}"
        )
    return cached


def _unseen_unavailable(
    public_composition_total: int,
    physical_cards_remaining: int,
) -> int:
    unavailable = public_composition_total - physical_cards_remaining
    if unavailable < 0:
        raise AssertionError("physical shoe cannot exceed the public composition")
    return unavailable


def _required_situation(
    situation: RoundPlayerSituation | None,
) -> RoundPlayerSituation:
    if situation is None:
        raise AssertionError("play decision requires a round situation")
    return situation


def _round_situation(
    hands: tuple[HandSnapshot, ...],
    active_index: int | None,
    context: ModelContext,
    composition: Composition,
    unseen_unavailable: int,
    configuration: DatasetConfiguration,
) -> RoundPlayerSituation:
    if active_index is None:
        raise AssertionError("player decision needs an active hand")
    active_snapshot = hands[active_index]
    active = _oracle_hand(
        active_snapshot,
        can_double=PlayerAction.DOUBLE in context.legal_player_actions,
        can_surrender=PlayerAction.SURRENDER in context.legal_player_actions,
    )
    pending = tuple(
        _oracle_hand(
            hand,
            can_double=(
                len(hand.cards) == 2
                and not hand.split_aces
                and (not hand.from_split or configuration.rules.double_after_split)
            ),
            can_surrender=False,
        )
        for index, hand in enumerate(hands)
        if index != active_index and not hand.finished
    )
    finished = tuple(
        sorted(
            (
                ResolvedHand.from_hand(
                    _oracle_hand(
                        hand,
                        can_double=False,
                        can_surrender=False,
                    ),
                    surrendered=hand.surrendered,
                )
                for index, hand in enumerate(hands)
                if index != active_index and hand.finished
            ),
            key=lambda hand: (
                hand.surrendered,
                hand.is_bust,
                hand.is_natural_blackjack,
                hand.total,
                hand.wager,
            ),
        )
    )
    upcard = context.dealer_upcard
    dealer_value = cards_to_values((upcard,))[0]
    peek = (
        PeekCondition.NO_BLACKJACK
        if dealer_value.value in ("A", "10")
        else PeekCondition.NONE
    )
    return RoundPlayerSituation(
        composition=composition,
        active_hand=active,
        pending_hands=pending,
        finished_hands=finished,
        dealer_upcard=dealer_value,
        peek_condition=peek,
        rules=configuration.rules,
        unseen_unavailable=unseen_unavailable,
    )


def _oracle_hand(
    snapshot: HandSnapshot,
    *,
    can_double: bool,
    can_surrender: bool,
) -> OracleHand:
    return OracleHand(
        cards=cards_to_values(snapshot.cards),
        wager=snapshot.wager,
        from_split=snapshot.from_split,
        split_aces=snapshot.split_aces,
        can_double=can_double,
        can_surrender=can_surrender,
    )


def _behavior_token(
    label: LabeledDecision,
    exploration_probability: Fraction,
    rng: Random,
) -> str:
    alternatives = tuple(
        token
        for token in label.metadata.legal_target_tokens
        if token != label.target_token
    )
    if alternatives and rng.random() < float(exploration_probability):
        return rng.choice(alternatives)
    return label.target_token


def _apply_behavior(
    game: BlackjackRound,
    kind: DecisionKind,
    token: str,
) -> None:
    if kind is DecisionKind.INSURANCE:
        game.decide_insurance(insurance_action_for_token(token))
    elif kind is DecisionKind.PLAY:
        game.act(player_action_for_token(token))
    else:
        raise AssertionError("bet behavior does not mutate an active round")


def _example(
    prepared: PreparedShoe,
    dataset_id: str,
    round_index: int,
    decision_index: int,
    kind: DecisionKind,
    input_tokens: tuple[str, ...],
    label: LabeledDecision,
    behavior_token: str,
) -> DecisionExample:
    manifest = prepared.manifest
    return DecisionExample(
        schema_version=SCHEMA_VERSION,
        dataset_id=dataset_id,
        shoe_id=manifest.shoe_id,
        shoe_seed=manifest.seed,
        split=manifest.split,
        round_index=round_index,
        decision_index=decision_index,
        kind=kind,
        input_tokens=input_tokens,
        target_token=label.target_token,
        behavior_token=behavior_token,
        metadata=label.metadata,
    )
