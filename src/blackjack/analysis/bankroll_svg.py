"""Dependency-free SVG chart for paired bankroll trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from pathlib import Path

from blackjack.training.bankroll import (
    PairedBankrollEvaluation,
    PolicyTrajectory,
)

_WIDTH = 1_200
_HEIGHT = 600
_LEFT = 86
_RIGHT = 24
_TOP = 54
_BOTTOM = 72
_MAXIMUM_PLOTTED_POINTS = 2_500


@dataclass(frozen=True, slots=True)
class BankrollChartPoint:
    round_number: int
    bankroll: float


def write_bankroll_chart(
    evaluation: PairedBankrollEvaluation,
    output_path: Path,
) -> None:
    """Plot both live policies on one completed-round bankroll axis."""

    transformer = _trajectory_points(evaluation.transformer)
    hi_lo = _trajectory_points(evaluation.hi_lo)
    write_bankroll_series_chart(
        transformer,
        hi_lo,
        output_path,
        title="Bankroll growth on fresh deterministic six-deck H17 shoes",
    )


def write_bankroll_series_chart(
    transformer: tuple[BankrollChartPoint, ...],
    hi_lo: tuple[BankrollChartPoint, ...],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Plot positive bankroll series on a shared logarithmic scale."""

    all_points = (*transformer, *hi_lo)
    if not all_points:
        raise ValueError("a bankroll chart needs at least one point")
    if any(point.bankroll <= 0 for point in all_points):
        raise ValueError("a logarithmic bankroll chart needs positive values")
    maximum_round = max(point.round_number for point in all_points)
    log_values = tuple(log(point.bankroll) for point in all_points)
    minimum_log_bankroll = min(log_values)
    maximum_log_bankroll = max(log_values)
    span = max(maximum_log_bankroll - minimum_log_bankroll, 1.0)
    minimum_log_bankroll -= span * 0.08
    maximum_log_bankroll += span * 0.08

    transformer_path = _polyline(
        transformer,
        maximum_round,
        minimum_log_bankroll,
        maximum_log_bankroll,
    )
    hi_lo_path = _polyline(
        hi_lo,
        maximum_round,
        minimum_log_bankroll,
        maximum_log_bankroll,
    )
    grid = _grid_lines(
        maximum_round,
        minimum_log_bankroll,
        maximum_log_bankroll,
    )
    start_y = _y(
        log(transformer[0].bankroll),
        minimum_log_bankroll,
        maximum_log_bankroll,
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
  width="{_WIDTH}" height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}"
  role="img" aria-labelledby="title description">
  <title id="title">Transformer and Hi-Lo bankroll trajectories</title>
  <desc id="description">
    Both policies start with 100 bankroll units and play fresh deterministic
    six-deck H17 shoes.
  </desc>
  <rect width="{_WIDTH}" height="{_HEIGHT}" fill="#ffffff"/>
  <text x="{_LEFT}" y="28" font-family="system-ui, sans-serif"
    font-size="20" font-weight="600" fill="#1f2937">
    {title}
  </text>
  {grid}
  <line x1="{_LEFT}" y1="{start_y:.2f}" x2="{_WIDTH - _RIGHT}"
    y2="{start_y:.2f}" stroke="#9ca3af" stroke-width="1"
    stroke-dasharray="5 5"/>
  <polyline points="{hi_lo_path}" fill="none" stroke="#2563eb"
    stroke-width="2"/>
  <polyline points="{transformer_path}" fill="none" stroke="#dc2626"
    stroke-width="2"/>
  <line x1="{_LEFT}" y1="{_HEIGHT - _BOTTOM}"
    x2="{_WIDTH - _RIGHT}" y2="{_HEIGHT - _BOTTOM}"
    stroke="#374151" stroke-width="1"/>
  <line x1="{_LEFT}" y1="{_TOP}" x2="{_LEFT}"
    y2="{_HEIGHT - _BOTTOM}" stroke="#374151" stroke-width="1"/>
  <text x="{(_LEFT + _WIDTH - _RIGHT) / 2:.1f}" y="{_HEIGHT - 22}"
    text-anchor="middle" font-family="system-ui, sans-serif"
    font-size="14" fill="#374151">Completed rounds</text>
  <text x="20" y="{(_TOP + _HEIGHT - _BOTTOM) / 2:.1f}"
    text-anchor="middle"
    transform="rotate(-90 20 {(_TOP + _HEIGHT - _BOTTOM) / 2:.1f})"
    font-family="system-ui, sans-serif" font-size="14"
    fill="#374151">Bankroll (log scale; start = 100)</text>
  <line x1="{_WIDTH - 246}" y1="24" x2="{_WIDTH - 214}" y2="24"
    stroke="#2563eb" stroke-width="2"/>
  <text x="{_WIDTH - 206}" y="29" font-family="system-ui, sans-serif"
    font-size="13" fill="#374151">Hi-Lo</text>
  <line x1="{_WIDTH - 140}" y1="24" x2="{_WIDTH - 108}" y2="24"
    stroke="#dc2626" stroke-width="2"/>
  <text x="{_WIDTH - 100}" y="29" font-family="system-ui, sans-serif"
    font-size="13" fill="#374151">Transformer</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def _trajectory_points(
    trajectory: PolicyTrajectory,
) -> tuple[BankrollChartPoint, ...]:
    points = [BankrollChartPoint(0, trajectory.initial_bankroll)]
    step = max(1, len(trajectory.rounds) // _MAXIMUM_PLOTTED_POINTS)
    for index in range(step - 1, len(trajectory.rounds), step):
        record = trajectory.rounds[index]
        points.append(
            BankrollChartPoint(
                record.global_round_index + 1,
                record.bankroll_after,
            )
        )
    if trajectory.rounds:
        final = trajectory.rounds[-1]
        if points[-1].round_number != final.global_round_index + 1:
            points.append(
                BankrollChartPoint(
                    final.global_round_index + 1,
                    final.bankroll_after,
                )
            )
    return tuple(points)


def _polyline(
    points: tuple[BankrollChartPoint, ...],
    maximum_round: int,
    minimum_log_bankroll: float,
    maximum_log_bankroll: float,
) -> str:
    return " ".join(
        (
            f"{_x(point.round_number, maximum_round):.2f},"
            f"{_y(log(point.bankroll), minimum_log_bankroll, maximum_log_bankroll):.2f}"
        )
        for point in points
    )


def _grid_lines(
    maximum_round: int,
    minimum_log_bankroll: float,
    maximum_log_bankroll: float,
) -> str:
    pieces: list[str] = []
    for tick in range(6):
        fraction = tick / 5
        x = _LEFT + fraction * (_WIDTH - _LEFT - _RIGHT)
        round_number = round(fraction * maximum_round)
        pieces.append(
            f'<line x1="{x:.2f}" y1="{_TOP}" x2="{x:.2f}" '
            f'y2="{_HEIGHT - _BOTTOM}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{x:.2f}" y="{_HEIGHT - _BOTTOM + 24}" '
            'text-anchor="middle" font-family="system-ui, sans-serif" '
            f'font-size="12" fill="#6b7280">{round_number:,}</text>'
        )
        log_bankroll = maximum_log_bankroll - fraction * (
            maximum_log_bankroll - minimum_log_bankroll
        )
        bankroll = exp(log_bankroll)
        y = _TOP + fraction * (_HEIGHT - _TOP - _BOTTOM)
        pieces.append(
            f'<line x1="{_LEFT}" y1="{y:.2f}" x2="{_WIDTH - _RIGHT}" '
            f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{_LEFT - 10}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="system-ui, sans-serif" font-size="12" '
            f'fill="#6b7280">{_bankroll_label(bankroll)}</text>'
        )
    return "\n  ".join(pieces)


def _x(round_number: int, maximum_round: int) -> float:
    if maximum_round == 0:
        return float(_LEFT)
    return _LEFT + (
        round_number / maximum_round * (_WIDTH - _LEFT - _RIGHT)
    )


def _y(
    log_bankroll: float,
    minimum_log_bankroll: float,
    maximum_log_bankroll: float,
) -> float:
    fraction = (log_bankroll - minimum_log_bankroll) / (
        maximum_log_bankroll - minimum_log_bankroll
    )
    return _HEIGHT - _BOTTOM - fraction * (_HEIGHT - _TOP - _BOTTOM)


def _bankroll_label(bankroll: float) -> str:
    if bankroll < 10_000:
        return f"{bankroll:,.0f}"
    return f"{bankroll:.1e}"
