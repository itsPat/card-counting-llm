"""Exact composition-dependent blackjack oracle."""

from blackjack.oracle.cdz import (
    BetSplitPolicy,
    ExhaustiveNumericRoundAnalysis,
    exhaustive_cdz_round_return_distribution,
)
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
from blackjack.oracle.kelly import (
    KellyRecommendation,
    expected_log_growth,
    kelly_recommendation,
)
from blackjack.oracle.monte_carlo import (
    FixedPolicyActionEstimate,
    FixedPolicyRoundEstimate,
    NativeSimulationBuildError,
    ensure_native_simulation_kernel,
    fixed_policy_play_action_estimates,
    fixed_policy_play_rollout_seed,
    fixed_policy_rollout_seed,
    fixed_policy_round_return_estimate,
    native_simulation_library_path,
)
from blackjack.oracle.player import (
    ActionEvaluation,
    OracleHand,
    OracleHandValue,
    PlayerSituation,
    ResolvedHand,
    RoundPlayerSituation,
    evaluate_actions,
    evaluate_round_actions,
    legal_actions,
    optimal_action,
    optimal_return_distribution,
    optimal_round_action,
    oracle_hand_value,
)
from blackjack.oracle.profiling import (
    OracleCacheCounter,
    OracleCacheProfile,
    clear_oracle_caches,
    oracle_cache_profile,
)
from blackjack.oracle.round_returns import (
    RoundReturnAnalysis,
    round_return_distribution,
)

__all__ = [
    "CARD_VALUES",
    "ActionEvaluation",
    "BetSplitPolicy",
    "CardValue",
    "Composition",
    "DealerDistribution",
    "DealerOutcome",
    "DealerOutcomeProbability",
    "Draw",
    "ExhaustiveNumericRoundAnalysis",
    "FixedPolicyActionEstimate",
    "FixedPolicyRoundEstimate",
    "InsuranceEvaluation",
    "KellyRecommendation",
    "NativeSimulationBuildError",
    "OracleCacheCounter",
    "OracleCacheProfile",
    "OracleHand",
    "OracleHandValue",
    "PeekCondition",
    "PlayerSituation",
    "ResolvedHand",
    "ReturnDistribution",
    "ReturnOutcome",
    "RoundPlayerSituation",
    "RoundReturnAnalysis",
    "cards_to_values",
    "clear_oracle_caches",
    "dealer_blackjack_probability",
    "dealer_distribution",
    "ensure_native_simulation_kernel",
    "evaluate_actions",
    "evaluate_insurance",
    "evaluate_round_actions",
    "exhaustive_cdz_round_return_distribution",
    "expected_log_growth",
    "fixed_policy_play_action_estimates",
    "fixed_policy_play_rollout_seed",
    "fixed_policy_rollout_seed",
    "fixed_policy_round_return_estimate",
    "hidden_hole_draws",
    "kelly_recommendation",
    "legal_actions",
    "native_simulation_library_path",
    "optimal_action",
    "optimal_insurance",
    "optimal_return_distribution",
    "optimal_round_action",
    "oracle_cache_profile",
    "oracle_hand_value",
    "round_return_distribution",
]
