"""The complete one-player American-hole-card blackjack state machine."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from blackjack.actions import InsuranceAction, PlayerAction, RoundPhase
from blackjack.cards import Card, Rank
from blackjack.events import (
    EventType,
    EventVisibility,
    InternalEvent,
    PublicEvent,
    public_events,
)
from blackjack.hands import Hand, HandValue, calculate_hand_value
from blackjack.rules import FIXED_RULES, CasinoRules
from blackjack.settlement import (
    HandOutcome,
    RoundSettlement,
    settle_hand,
    settle_insurance,
)
from blackjack.shoe import Shoe


class BlackjackStateError(RuntimeError):
    """Base class for invalid blackjack state transitions."""


class InvalidPhaseError(BlackjackStateError):
    """Raised when a decision is submitted in the wrong round phase."""


class IllegalActionError(BlackjackStateError):
    """Raised when a player action is not legal for the active hand."""


@dataclass(slots=True)
class _PlayerHandState:
    cards: list[Card]
    wager: Fraction
    from_split: bool = False
    split_aces: bool = False
    finished: bool = False
    surrendered: bool = False
    doubled: bool = False
    actions_taken: int = 0

    @property
    def hand(self) -> Hand:
        return Hand(cards=tuple(self.cards), from_split=self.from_split)


@dataclass(frozen=True, slots=True)
class HandSnapshot:
    cards: tuple[Card, ...]
    value: HandValue
    wager: Fraction
    from_split: bool
    split_aces: bool
    doubled: bool
    surrendered: bool
    finished: bool
    is_active: bool
    outcome: HandOutcome | None
    profit: Fraction | None


@dataclass(frozen=True, slots=True)
class PublicRoundState:
    phase: RoundPhase
    player_hands: tuple[HandSnapshot, ...]
    active_hand_index: int | None
    dealer_cards: tuple[Card, ...]
    dealer_hole_revealed: bool
    legal_actions: tuple[PlayerAction, ...]
    insurance_available: bool
    insurance_decision: InsuranceAction | None
    visible_card_history: tuple[Card, ...]
    events: tuple[PublicEvent, ...]
    settlement: RoundSettlement | None


@dataclass(frozen=True, slots=True)
class InternalRoundState:
    phase: RoundPhase
    player_hands: tuple[HandSnapshot, ...]
    active_hand_index: int | None
    dealer_cards: tuple[Card, ...]
    dealer_hole_revealed: bool
    insurance_decision: InsuranceAction | None
    visible_card_history: tuple[Card, ...]
    events: tuple[InternalEvent, ...]
    settlement: RoundSettlement | None


class BlackjackRound:
    """A stateful round whose snapshots never expose mutable engine lists."""

    __slots__ = (
        "_active_hand_index",
        "_dealer_cards",
        "_dealer_hole_revealed",
        "_events",
        "_insurance_decision",
        "_original_wager",
        "_phase",
        "_player_hands",
        "_rules",
        "_settlement",
        "_shoe",
        "_visible_cards",
    )

    def __init__(
        self,
        shoe: Shoe,
        wager: Fraction | int,
        rules: CasinoRules = FIXED_RULES,
    ) -> None:
        original_wager = Fraction(wager)
        if original_wager <= 0:
            raise ValueError("wager must be positive")
        self._shoe = shoe
        self._rules = rules
        self._original_wager = original_wager
        self._phase = RoundPhase.PLAYER_ACTIONS
        self._player_hands: list[_PlayerHandState] = []
        self._dealer_cards: list[Card] = []
        self._active_hand_index: int | None = None
        self._dealer_hole_revealed = False
        self._insurance_decision: InsuranceAction | None = None
        self._visible_cards: list[Card] = []
        self._events: list[InternalEvent] = []
        self._settlement: RoundSettlement | None = None
        self._initial_deal()

    @property
    def public_state(self) -> PublicRoundState:
        snapshots = self._hand_snapshots()
        dealer_cards = (
            tuple(self._dealer_cards)
            if self._dealer_hole_revealed
            else (self._dealer_cards[0],)
        )
        return PublicRoundState(
            phase=self._phase,
            player_hands=snapshots,
            active_hand_index=self._active_hand_index,
            dealer_cards=dealer_cards,
            dealer_hole_revealed=self._dealer_hole_revealed,
            legal_actions=self.legal_actions,
            insurance_available=self._phase is RoundPhase.INSURANCE,
            insurance_decision=self._insurance_decision,
            visible_card_history=tuple(self._visible_cards),
            events=public_events(tuple(self._events)),
            settlement=self._settlement,
        )

    @property
    def internal_state(self) -> InternalRoundState:
        return InternalRoundState(
            phase=self._phase,
            player_hands=self._hand_snapshots(),
            active_hand_index=self._active_hand_index,
            dealer_cards=tuple(self._dealer_cards),
            dealer_hole_revealed=self._dealer_hole_revealed,
            insurance_decision=self._insurance_decision,
            visible_card_history=tuple(self._visible_cards),
            events=tuple(self._events),
            settlement=self._settlement,
        )

    @property
    def legal_actions(self) -> tuple[PlayerAction, ...]:
        if (
            self._phase is not RoundPhase.PLAYER_ACTIONS
            or self._active_hand_index is None
        ):
            return ()
        state = self._player_hands[self._active_hand_index]
        if state.finished:
            return ()
        hand = state.hand
        legal: list[PlayerAction] = []
        if not state.split_aces and not hand.value.is_bust and hand.value.total < 21:
            legal.append(PlayerAction.HIT)
        legal.append(PlayerAction.STAND)
        if (
            len(state.cards) == 2
            and state.actions_taken == 0
            and not state.split_aces
            and (not state.from_split or self._rules.double_after_split)
        ):
            legal.append(PlayerAction.DOUBLE)
        if (
            len(state.cards) == 2
            and state.actions_taken == 0
            and hand.is_pair
            and len(self._player_hands) < self._rules.maximum_player_hands
            and not (state.from_split and state.cards[0].rank is Rank.ACE)
        ):
            legal.append(PlayerAction.SPLIT)
        if (
            self._rules.late_surrender
            and not state.from_split
            and len(self._player_hands) == 1
            and len(state.cards) == 2
            and state.actions_taken == 0
        ):
            legal.append(PlayerAction.SURRENDER)
        return tuple(legal)

    def decide_insurance(self, action: InsuranceAction) -> None:
        if self._phase is not RoundPhase.INSURANCE:
            raise InvalidPhaseError("insurance is not awaiting a decision")
        self._insurance_decision = action
        self._record(
            EventType.INSURANCE_DECIDED,
            EventVisibility.PUBLIC,
            insurance_action=action,
            amount=(
                self._original_wager * self._rules.insurance_fraction
                if action is InsuranceAction.TAKE
                else Fraction(0)
            ),
        )
        self._peek_dealer()

    def act(self, action: PlayerAction) -> None:
        if self._phase is not RoundPhase.PLAYER_ACTIONS:
            raise InvalidPhaseError("the round is not accepting player actions")
        if self._active_hand_index is None:
            raise InvalidPhaseError("there is no active player hand")
        if action not in self.legal_actions:
            raise IllegalActionError(f"{action.value} is not legal in this state")

        index = self._active_hand_index
        state = self._player_hands[index]
        self._record(
            EventType.PLAYER_ACTED,
            EventVisibility.PUBLIC,
            hand_index=index,
            player_action=action,
        )
        if action is PlayerAction.HIT:
            state.actions_taken += 1
            self._deal_to_player(index)
            if state.hand.value.total >= 21:
                state.finished = True
        elif action is PlayerAction.STAND:
            state.actions_taken += 1
            state.finished = True
        elif action is PlayerAction.DOUBLE:
            state.actions_taken += 1
            state.wager *= 2
            state.doubled = True
            self._deal_to_player(index)
            state.finished = True
        elif action is PlayerAction.SURRENDER:
            state.actions_taken += 1
            state.surrendered = True
            state.finished = True
        else:
            self._split_active_hand()
        self._advance_player_turn()

    def _initial_deal(self) -> None:
        self._record(
            EventType.SHOE_BURNED,
            EventVisibility.INTERNAL,
            card=self._shoe.burn_card,
        )
        first_player = self._shoe.deal()
        self._visible_cards.append(first_player)
        self._record(
            EventType.PLAYER_CARD_DEALT,
            EventVisibility.PUBLIC,
            card=first_player,
            hand_index=0,
        )
        dealer_upcard = self._shoe.deal()
        self._dealer_cards.append(dealer_upcard)
        self._visible_cards.append(dealer_upcard)
        self._record(
            EventType.DEALER_UPCARD_DEALT,
            EventVisibility.PUBLIC,
            card=dealer_upcard,
        )
        second_player = self._shoe.deal()
        self._visible_cards.append(second_player)
        self._record(
            EventType.PLAYER_CARD_DEALT,
            EventVisibility.PUBLIC,
            card=second_player,
            hand_index=0,
        )
        self._player_hands.append(
            _PlayerHandState(
                cards=[first_player, second_player],
                wager=self._original_wager,
            )
        )
        hole_card = self._shoe.deal()
        self._dealer_cards.append(hole_card)
        self._record(
            EventType.DEALER_HOLE_CARD_DEALT,
            EventVisibility.INTERNAL,
            card=hole_card,
        )

        if dealer_upcard.rank is Rank.ACE:
            self._phase = RoundPhase.INSURANCE
            return
        if dealer_upcard.is_ten_valued and self._rules.dealer_peeks:
            self._peek_dealer()
            return
        self._begin_player_actions()

    def _peek_dealer(self) -> None:
        self._record(EventType.DEALER_PEEKED, EventVisibility.INTERNAL)
        if self._dealer_hand.is_natural_blackjack:
            self._reveal_hole_card()
            self._settle()
            return
        self._begin_player_actions()

    def _begin_player_actions(self) -> None:
        self._phase = RoundPhase.PLAYER_ACTIONS
        self._active_hand_index = 0
        if self._player_hands[0].hand.is_natural_blackjack:
            self._player_hands[0].finished = True
        self._advance_player_turn()

    def _deal_to_player(self, hand_index: int) -> Card:
        card = self._shoe.deal()
        self._player_hands[hand_index].cards.append(card)
        self._visible_cards.append(card)
        self._record(
            EventType.PLAYER_CARD_DEALT,
            EventVisibility.PUBLIC,
            card=card,
            hand_index=hand_index,
        )
        return card

    def _split_active_hand(self) -> None:
        if self._active_hand_index is None:
            raise InvalidPhaseError("there is no active hand to split")
        index = self._active_hand_index
        state = self._player_hands[index]
        first, second = state.cards
        split_aces = first.rank is Rank.ACE and second.rank is Rank.ACE
        left = _PlayerHandState(
            cards=[first],
            wager=state.wager,
            from_split=True,
            split_aces=split_aces,
        )
        right = _PlayerHandState(
            cards=[second],
            wager=state.wager,
            from_split=True,
            split_aces=split_aces,
        )
        self._player_hands[index : index + 1] = [left, right]
        self._record(
            EventType.HAND_SPLIT,
            EventVisibility.PUBLIC,
            hand_index=index,
            player_action=PlayerAction.SPLIT,
        )
        self._deal_to_player(index)
        self._deal_to_player(index + 1)
        if split_aces and self._rules.split_aces_one_card_only:
            left.finished = True
            right.finished = True
        else:
            if left.hand.value.total >= 21:
                left.finished = True
            if right.hand.value.total >= 21:
                right.finished = True

    def _advance_player_turn(self) -> None:
        start = self._active_hand_index if self._active_hand_index is not None else 0
        next_index = next(
            (
                index
                for index in range(start, len(self._player_hands))
                if not self._player_hands[index].finished
            ),
            None,
        )
        if next_index is not None:
            self._active_hand_index = next_index
            return
        self._active_hand_index = None
        if all(
            state.surrendered or state.hand.value.is_bust
            for state in self._player_hands
        ):
            self._settle()
            return
        self._play_dealer()

    def _play_dealer(self) -> None:
        self._phase = RoundPhase.DEALER_PLAY
        self._reveal_hole_card()
        if all(state.hand.is_natural_blackjack for state in self._player_hands):
            self._settle()
            return
        while self._dealer_should_hit:
            card = self._shoe.deal()
            self._dealer_cards.append(card)
            self._visible_cards.append(card)
            self._record(
                EventType.DEALER_HIT,
                EventVisibility.PUBLIC,
                card=card,
            )
        self._settle()

    @property
    def _dealer_should_hit(self) -> bool:
        value = self._dealer_hand.value
        return value.total < 17 or (
            value.total == 17 and value.is_soft and self._rules.dealer_hits_soft_17
        )

    @property
    def _dealer_hand(self) -> Hand:
        return Hand(cards=tuple(self._dealer_cards))

    def _reveal_hole_card(self) -> None:
        if self._dealer_hole_revealed:
            return
        self._dealer_hole_revealed = True
        hole_card = self._dealer_cards[1]
        self._visible_cards.append(hole_card)
        self._record(
            EventType.DEALER_HOLE_REVEALED,
            EventVisibility.PUBLIC,
            card=hole_card,
        )

    def _settle(self) -> None:
        dealer = self._dealer_hand if self._dealer_hole_revealed else None
        hand_settlements = tuple(
            settle_hand(
                hand_index=index,
                player=state.hand,
                wager=state.wager,
                dealer=dealer,
                surrendered=state.surrendered,
                rules=self._rules,
            )
            for index, state in enumerate(self._player_hands)
        )
        insurance = (
            settle_insurance(
                original_wager=self._original_wager,
                dealer_has_blackjack=self._dealer_hand.is_natural_blackjack,
                rules=self._rules,
            )
            if self._insurance_decision is InsuranceAction.TAKE
            else None
        )
        self._settlement = RoundSettlement(
            hands=hand_settlements,
            insurance=insurance,
        )
        self._phase = RoundPhase.SETTLED
        self._active_hand_index = None
        self._record(
            EventType.ROUND_SETTLED,
            EventVisibility.PUBLIC,
            amount=self._settlement.total_profit,
        )

    def _hand_snapshots(self) -> tuple[HandSnapshot, ...]:
        settlements = (
            {item.hand_index: item for item in self._settlement.hands}
            if self._settlement is not None
            else {}
        )
        return tuple(
            HandSnapshot(
                cards=tuple(state.cards),
                value=calculate_hand_value(state.cards),
                wager=state.wager,
                from_split=state.from_split,
                split_aces=state.split_aces,
                doubled=state.doubled,
                surrendered=state.surrendered,
                finished=state.finished,
                is_active=index == self._active_hand_index,
                outcome=(settlements[index].outcome if index in settlements else None),
                profit=(settlements[index].profit if index in settlements else None),
            )
            for index, state in enumerate(self._player_hands)
        )

    def _record(
        self,
        event_type: EventType,
        visibility: EventVisibility,
        *,
        card: Card | None = None,
        hand_index: int | None = None,
        player_action: PlayerAction | None = None,
        insurance_action: InsuranceAction | None = None,
        amount: Fraction | None = None,
    ) -> None:
        self._events.append(
            InternalEvent(
                sequence=len(self._events),
                event_type=event_type,
                visibility=visibility,
                card=card,
                hand_index=hand_index,
                player_action=player_action,
                insurance_action=insurance_action,
                amount=amount,
            )
        )
