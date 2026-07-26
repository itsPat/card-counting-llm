"""A compact causal transformer for blackjack decision sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class PositionScheme(StrEnum):
    ABSOLUTE = "absolute"
    QUERY_RELATIVE = "query-relative"


@dataclass(frozen=True, slots=True)
class TransformerConfiguration:
    vocabulary_size: int
    context_length: int = 256
    embedding_dimension: int = 128
    head_count: int = 4
    layer_count: int = 4
    feed_forward_dimension: int = 512
    dropout: float = 0.1
    position_scheme: PositionScheme = PositionScheme.QUERY_RELATIVE

    def __post_init__(self) -> None:
        positive_values = (
            self.vocabulary_size,
            self.context_length,
            self.embedding_dimension,
            self.head_count,
            self.layer_count,
            self.feed_forward_dimension,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("transformer dimensions must be positive")
        if self.embedding_dimension % self.head_count:
            raise ValueError(
                "embedding dimension must be divisible by head count"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")

    @property
    def head_dimension(self) -> int:
        return self.embedding_dimension // self.head_count


class CausalSelfAttention(nn.Module):
    """Multi-head attention written directly from the defining operations."""

    causal_mask: Tensor

    def __init__(self, configuration: TransformerConfiguration) -> None:
        super().__init__()
        self._head_count = configuration.head_count
        self._head_dimension = configuration.head_dimension
        self._scale = sqrt(configuration.head_dimension)
        self.query_key_value = nn.Linear(
            configuration.embedding_dimension,
            3 * configuration.embedding_dimension,
            bias=False,
        )
        self.output = nn.Linear(
            configuration.embedding_dimension,
            configuration.embedding_dimension,
            bias=False,
        )
        self.attention_dropout = nn.Dropout(configuration.dropout)
        self.output_dropout = nn.Dropout(configuration.dropout)
        causal_mask = torch.tril(
            torch.ones(
                (configuration.context_length, configuration.context_length),
                dtype=torch.bool,
            )
        )
        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, embedding_dimension = hidden_states.shape
        combined = self.query_key_value(hidden_states)
        query, key, value = combined.chunk(3, dim=-1)

        head_shape = (
            batch_size,
            sequence_length,
            self._head_count,
            self._head_dimension,
        )
        query = query.reshape(head_shape).transpose(1, 2)
        key = key.reshape(head_shape).transpose(1, 2)
        value = value.reshape(head_shape).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self._scale
        causal = self.causal_mask[:sequence_length, :sequence_length]
        visible_keys = attention_mask[:, None, None, :].to(
            device=scores.device,
            dtype=torch.bool,
        )
        allowed = causal[None, None, :, :] & visible_keys
        scores = scores.masked_fill(
            ~allowed,
            torch.finfo(scores.dtype).min,
        )
        weights = functional.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, value)
        merged = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            embedding_dimension,
        )
        return self.output_dropout(self.output(merged))


class FeedForward(nn.Module):
    def __init__(self, configuration: TransformerConfiguration) -> None:
        super().__init__()
        self.input = nn.Linear(
            configuration.embedding_dimension,
            configuration.feed_forward_dimension,
        )
        self.output = nn.Linear(
            configuration.feed_forward_dimension,
            configuration.embedding_dimension,
        )
        self.dropout = nn.Dropout(configuration.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        expanded = self.input(hidden_states)
        activated = functional.gelu(expanded)
        return self.dropout(self.output(activated))


class TransformerBlock(nn.Module):
    def __init__(self, configuration: TransformerConfiguration) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            configuration.embedding_dimension
        )
        self.attention = CausalSelfAttention(configuration)
        self.feed_forward_norm = nn.LayerNorm(
            configuration.embedding_dimension
        )
        self.feed_forward = FeedForward(configuration)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
        )
        return hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )


class BlackjackTransformer(nn.Module):
    """Decoder-only transformer returning one logit vector per input token."""

    def __init__(
        self,
        configuration: TransformerConfiguration,
        *,
        padding_index: int,
    ) -> None:
        super().__init__()
        if not 0 <= padding_index < configuration.vocabulary_size:
            raise ValueError("padding index is outside the vocabulary")
        self.configuration = configuration
        self.token_embedding = nn.Embedding(
            configuration.vocabulary_size,
            configuration.embedding_dimension,
            padding_idx=padding_index,
        )
        self.position_embedding = nn.Embedding(
            configuration.context_length,
            configuration.embedding_dimension,
        )
        self.embedding_dropout = nn.Dropout(configuration.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(configuration)
            for _ in range(configuration.layer_count)
        )
        self.final_norm = nn.LayerNorm(configuration.embedding_dimension)
        self.language_model_head = nn.Linear(
            configuration.embedding_dimension,
            configuration.vocabulary_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input IDs must have shape [batch, time]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention mask must match the input shape")
        sequence_length = input_ids.shape[1]
        if sequence_length > self.configuration.context_length:
            raise ValueError("input exceeds the transformer context length")

        positions = self.position_ids(attention_mask)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        return self.language_model_head(self.final_norm(hidden_states))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def position_ids(self, attention_mask: Tensor) -> Tensor:
        """Return absolute or decision-query-relative learned positions."""

        if attention_mask.ndim != 2:
            raise ValueError("attention mask must have shape [batch, time]")
        sequence_length = attention_mask.shape[1]
        base = torch.arange(
            sequence_length,
            device=attention_mask.device,
        )
        if (
            self.configuration.position_scheme
            is PositionScheme.ABSOLUTE
        ):
            return base
        lengths = attention_mask.sum(dim=1, dtype=torch.long)
        positions = (
            self.configuration.context_length
            - lengths[:, None]
            + base[None, :]
        )
        return positions.masked_fill(~attention_mask, 0)
