"""Exact blackjack and insurance settlement using rational arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from blackjack.hands import Hand
from blackjack.rules import CasinoRules, FIXED_RULES


class HandOutcome(str, Enum):
    BLACKJACK = "blackjack"
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    SURRENDER = "surrender"
    BUST = "bust"


class InsuranceOutcome(str, Enum):
    WON = "won"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class HandSettlement:
    hand_index: int
    outcome: HandOutcome
    wager: Fraction
    profit: Fraction
    payout: Fraction


@dataclass(frozen=True, slots=True)
class InsuranceSettlement:
    outcome: InsuranceOutcome
    stake: Fraction
    profit: Fraction
    payout: Fraction


@dataclass(frozen=True, slots=True)
class RoundSettlement:
    hands: tuple[HandSettlement, ...]
    insurance: InsuranceSettlement | None

    @property
    def total_profit(self) -> Fraction:
        insurance_profit = (
            self.insurance.profit if self.insurance is not None else Fraction(0)
        )
        return sum(
            (hand.profit for hand in self.hands),
            start=insurance_profit,
        )

    @property
    def total_payout(self) -> Fraction:
        insurance_payout = (
            self.insurance.payout if self.insurance is not None else Fraction(0)
        )
        return sum(
            (hand.payout for hand in self.hands),
            start=insurance_payout,
        )


def settle_hand(
    *,
    hand_index: int,
    player: Hand,
    wager: Fraction,
    dealer: Hand | None,
    surrendered: bool,
    rules: CasinoRules = FIXED_RULES,
) -> HandSettlement:
    """Settle one player hand and return net profit plus gross payout."""

    if surrendered:
        returned = wager / 2
        return HandSettlement(
            hand_index=hand_index,
            outcome=HandOutcome.SURRENDER,
            wager=wager,
            profit=-wager / 2,
            payout=returned,
        )
    if player.value.is_bust:
        return HandSettlement(
            hand_index=hand_index,
            outcome=HandOutcome.BUST,
            wager=wager,
            profit=-wager,
            payout=Fraction(0),
        )
    if dealer is None:
        raise ValueError("a live player hand requires a revealed dealer hand")

    if dealer.is_natural_blackjack:
        if player.is_natural_blackjack:
            return HandSettlement(
                hand_index=hand_index,
                outcome=HandOutcome.PUSH,
                wager=wager,
                profit=Fraction(0),
                payout=wager,
            )
        return HandSettlement(
            hand_index=hand_index,
            outcome=HandOutcome.LOSS,
            wager=wager,
            profit=-wager,
            payout=Fraction(0),
        )
    if player.is_natural_blackjack:
        profit = wager * rules.blackjack_profit
        return HandSettlement(
            hand_index=hand_index,
            outcome=HandOutcome.BLACKJACK,
            wager=wager,
            profit=profit,
            payout=wager + profit,
        )
    if dealer.value.is_bust or player.value.total > dealer.value.total:
        profit = wager * rules.ordinary_win_profit
        return HandSettlement(
            hand_index=hand_index,
            outcome=HandOutcome.WIN,
            wager=wager,
            profit=profit,
            payout=wager + profit,
        )
    if player.value.total == dealer.value.total:
        return HandSettlement(
            hand_index=hand_index,
            outcome=HandOutcome.PUSH,
            wager=wager,
            profit=Fraction(0),
            payout=wager,
        )
    return HandSettlement(
        hand_index=hand_index,
        outcome=HandOutcome.LOSS,
        wager=wager,
        profit=-wager,
        payout=Fraction(0),
    )


def settle_insurance(
    *,
    original_wager: Fraction,
    dealer_has_blackjack: bool,
    rules: CasinoRules = FIXED_RULES,
) -> InsuranceSettlement:
    stake = original_wager * rules.insurance_fraction
    if dealer_has_blackjack:
        profit = stake * rules.insurance_profit
        return InsuranceSettlement(
            outcome=InsuranceOutcome.WON,
            stake=stake,
            profit=profit,
            payout=stake + profit,
        )
    return InsuranceSettlement(
        outcome=InsuranceOutcome.LOST,
        stake=stake,
        profit=-stake,
        payout=Fraction(0),
    )
