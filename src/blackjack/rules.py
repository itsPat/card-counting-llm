"""The fixed casino rules used throughout the experiment."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class CasinoRules:
    decks: int = 6
    blackjack_profit: Fraction = Fraction(3, 2)
    ordinary_win_profit: Fraction = Fraction(1, 1)
    dealer_hits_soft_17: bool = True
    dealer_peeks: bool = True
    double_after_split: bool = True
    maximum_player_hands: int = 4
    resplit_aces: bool = False
    split_aces_one_card_only: bool = True
    late_surrender: bool = True
    insurance_fraction: Fraction = Fraction(1, 2)
    insurance_profit: Fraction = Fraction(2, 1)
    minimum_penetration: Fraction = Fraction(7, 10)
    maximum_penetration: Fraction = Fraction(4, 5)
    burn_cards: int = 1


FIXED_RULES = CasinoRules()
