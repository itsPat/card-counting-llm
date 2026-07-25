"""Exact composition-dependent blackjack oracle."""

from blackjack.oracle.composition import (
    CARD_VALUES,
    CardValue,
    Composition,
    Draw,
    cards_to_values,
)
from blackjack.oracle.dealer import (
    DealerDistribution,
    DealerOutcome,
    DealerOutcomeProbability,
    PeekCondition,
    dealer_blackjack_probability,
    dealer_distribution,
    hidden_hole_draws,
)
from blackjack.oracle.distributions import (
    ReturnDistribution,
    ReturnOutcome,
)
from blackjack.oracle.insurance import (
    InsuranceEvaluation,
    evaluate_insurance,
    optimal_insurance,
)
from blackjack.oracle.player import (
    ActionEvaluation,
    OracleHand,
    OracleHandValue,
    PlayerSituation,
    evaluate_actions,
    legal_actions,
    optimal_action,
    optimal_return_distribution,
    oracle_hand_value,
)

__all__ = [
    "CARD_VALUES",
    "ActionEvaluation",
    "CardValue",
    "Composition",
    "DealerDistribution",
    "DealerOutcome",
    "DealerOutcomeProbability",
    "Draw",
    "InsuranceEvaluation",
    "OracleHand",
    "OracleHandValue",
    "PeekCondition",
    "PlayerSituation",
    "ReturnDistribution",
    "ReturnOutcome",
    "cards_to_values",
    "dealer_blackjack_probability",
    "dealer_distribution",
    "evaluate_actions",
    "evaluate_insurance",
    "hidden_hole_draws",
    "legal_actions",
    "optimal_action",
    "optimal_insurance",
    "optimal_return_distribution",
    "oracle_hand_value",
]
