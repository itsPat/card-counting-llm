from __future__ import annotations

import pytest
import torch

from blackjack.training import BLACKJACK_VOCABULARY
from blackjack.training.model import (
    BlackjackTransformer,
    PositionScheme,
    TransformerConfiguration,
)


def _configuration() -> TransformerConfiguration:
    return TransformerConfiguration(
        vocabulary_size=len(BLACKJACK_VOCABULARY),
        context_length=8,
        embedding_dimension=16,
        head_count=4,
        layer_count=2,
        feed_forward_dimension=32,
        dropout=0,
    )


def _model() -> BlackjackTransformer:
    generator = torch.Generator()
    generator.manual_seed(19)
    torch.set_rng_state(generator.get_state())
    model = BlackjackTransformer(
        _configuration(),
        padding_index=BLACKJACK_VOCABULARY.pad_id,
    )
    model.eval()
    return model


def test_transformer_returns_one_vocabulary_logit_per_position() -> None:
    model = _model()
    inputs = torch.tensor(((1, 2, 3), (4, 5, 0)), dtype=torch.long)
    mask = inputs != 0
    logits = model(inputs, mask)
    assert logits.shape == (2, 3, len(BLACKJACK_VOCABULARY))
    assert torch.isfinite(logits).all()
    assert model.parameter_count > 0


def test_causal_logits_do_not_depend_on_future_tokens() -> None:
    model = _model()
    first = torch.tensor(((1, 2, 3, 4),), dtype=torch.long)
    second = torch.tensor(((1, 2, 8, 9),), dtype=torch.long)
    mask = torch.ones_like(first, dtype=torch.bool)
    first_logits = model(first, mask)
    second_logits = model(second, mask)
    assert torch.allclose(first_logits[:, :2], second_logits[:, :2])
    assert not torch.allclose(first_logits[:, 2:], second_logits[:, 2:])


def test_real_tokens_are_unchanged_by_right_padding() -> None:
    model = _model()
    short = torch.tensor(((1, 2, 3),), dtype=torch.long)
    padded = torch.tensor(((1, 2, 3, 0, 0),), dtype=torch.long)
    short_logits = model(short, short != 0)
    padded_logits = model(padded, padded != 0)
    assert torch.allclose(short_logits, padded_logits[:, :3])


def test_query_relative_positions_anchor_each_real_sequence_end() -> None:
    configuration = TransformerConfiguration(
        vocabulary_size=len(BLACKJACK_VOCABULARY),
        context_length=8,
        embedding_dimension=16,
        head_count=4,
        layer_count=1,
        feed_forward_dimension=32,
        dropout=0,
        position_scheme=PositionScheme.QUERY_RELATIVE,
    )
    model = BlackjackTransformer(
        configuration,
        padding_index=BLACKJACK_VOCABULARY.pad_id,
    )
    mask = torch.tensor(
        ((True, True, True), (True, True, False)),
        dtype=torch.bool,
    )
    assert torch.equal(
        model.position_ids(mask),
        torch.tensor(((5, 6, 7), (6, 7, 0)), dtype=torch.long),
    )


def test_transformer_rejects_invalid_shapes_and_configuration() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfiguration(vocabulary_size=29, head_count=3)
    model = _model()
    with pytest.raises(ValueError, match="match"):
        model(
            torch.ones((1, 3), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="context"):
        model(
            torch.ones((1, 9), dtype=torch.long),
            torch.ones((1, 9), dtype=torch.bool),
        )
