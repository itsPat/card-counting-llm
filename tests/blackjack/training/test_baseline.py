from __future__ import annotations

from blackjack.dataset import DecisionKind
from blackjack.training import (
    BLACKJACK_VOCABULARY,
    DecisionDataset,
    EncodedDecision,
)
from blackjack.training.baseline import (
    BasicStrategyBaseline,
    LegalFrequencyBaseline,
    evaluate_decision_baseline,
    evaluate_frequency_baseline,
)


def _encoded(
    index: int,
    target: str,
    legal: tuple[str, ...],
) -> EncodedDecision:
    return EncodedDecision(
        input_ids=BLACKJACK_VOCABULARY.encode(
            ("<HISTORY>", "<PLAY_QUERY>")
        ),
        target_id=BLACKJACK_VOCABULARY.id_for(target),
        legal_token_ids=BLACKJACK_VOCABULARY.encode(legal),
        kind=DecisionKind.PLAY,
        shoe_id=0,
        decision_index=index,
    )


def test_frequency_baseline_uses_counts_but_always_respects_legality() -> None:
    dataset = DecisionDataset(
        (
            _encoded(0, "<STAND>", ("<HIT>", "<STAND>")),
            _encoded(1, "<STAND>", ("<HIT>", "<STAND>")),
            _encoded(2, "<HIT>", ("<HIT>", "<STAND>")),
            _encoded(3, "<DOUBLE>", ("<HIT>", "<DOUBLE>")),
            _encoded(4, "<DOUBLE>", ("<DOUBLE>",)),
        )
    )
    baseline = LegalFrequencyBaseline.fit(dataset)
    metrics = evaluate_frequency_baseline(baseline, dataset, batch_size=2)
    assert metrics.correct == 4
    assert metrics.total == 5


def test_basic_strategy_baseline_reads_current_hand_but_not_history() -> None:
    examples = (
        EncodedDecision(
            input_ids=BLACKJACK_VOCABULARY.encode(
                (
                    "<HISTORY>",
                    "2",
                    "3",
                    "<CURRENT_HAND>",
                    "<PLAYER>",
                    "10",
                    "6",
                    "<DEALER>",
                    "10",
                    "<PLAY_QUERY>",
                )
            ),
            target_id=BLACKJACK_VOCABULARY.id_for("<SURRENDER>"),
            legal_token_ids=BLACKJACK_VOCABULARY.encode(
                ("<HIT>", "<STAND>", "<SURRENDER>")
            ),
            kind=DecisionKind.PLAY,
            shoe_id=0,
            decision_index=0,
        ),
    )
    dataset = DecisionDataset(examples)
    metrics = evaluate_decision_baseline(
        BasicStrategyBaseline(dataset),
        dataset,
    )
    assert metrics.correct == 1
