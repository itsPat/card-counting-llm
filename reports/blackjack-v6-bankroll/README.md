# Blackjack v6 Bankroll Evaluation

The retained v6 transformer outperformed the predeclared Hi-Lo control on the
primary metric across 116,000 fresh deterministic six-deck shoes.

| Result | Transformer | Hi-Lo |
| --- | ---: | ---: |
| Completed rounds | 5,015,984 | 5,025,662 |
| EV per round | 0.01737 min-bet units | 0.01258 min-bet units |
| EV per 100 rounds | 1.737 min-bet units | 1.258 min-bet units |
| 95% CI, EV per 100 rounds | [1.447, 2.027] | [0.993, 1.524] |
| Bankroll EV per 100 rounds | 0.1737% | 0.1258% |
| Mean log growth per 100 rounds | 0.001191 | 0.000796 |
| Mean unscaled profit per round | -0.003677 units | -0.004621 units |

The policies played 10,041,646 combined policy-rounds. Different play choices
consume different cards, so their round counts differ even though they receive
the same 116,000 shuffled shoe orders and cut-card positions.

The primary paired transformer-minus-Hi-Lo log-growth advantage is
`0.00016734` per shoe, with a 95% confidence interval of
`[0.00006587, 0.00026880]`. The full interval is above zero, satisfying the
outperformance criterion declared before the scale run.

For the more conventional card-counting interpretation, one minimum-bet unit
is `<BET_MIN>`, or 0.1% of bankroll. The transformer earns an estimated 1.737
minimum-bet units per 100 rounds, versus 1.258 for Hi-Lo: a descriptive
difference of 0.479 units per 100 rounds, or about 38% more EV under this bet
spread. These intervals are clustered by shoe.

EV per hour depends on table speed, so the report stores scenarios rather than
asserting one universal game rate:

| Rounds/hour | Transformer units/hour | Hi-Lo units/hour |
| ---: | ---: | ---: |
| 60 | 1.042 | 0.755 |
| 100 | 1.737 | 1.258 |
| 150 | 2.606 | 1.887 |

For a concrete conversion, if one minimum-bet unit is $25, the 100-round/hour
row is approximately $43.43/hour for the transformer and $31.46/hour for
Hi-Lo. That example implies the project's 1-to-13 spread is $25 to $325 and
does not subtract travel, tipping, errors, heat, or other real-world costs.

This is evidence against one specific, documented control rather than every
possible card-counting system. The Hi-Lo control combines six-deck H17 basic
strategy with a documented subset of Illustrious 18/Fab 4 deviations,
insurance at true count +3, and the project bet ramp.

## Context-boundary correction

One live decision in the final shard contained 257 tokens, one more than the
retained model's 256-token context. The evaluator now removes only the minimum
number of oldest visible-history card tokens needed to fit, while retaining the
history marker, full current hand, dealer upcard, structural markers, and query.
It records every occurrence.

The resumed shard made 233,436 transformer decisions. Exactly one was
truncated, by exactly one oldest history-card token. The other 40 reports were
created under fail-on-overflow behavior; their successful completion proves
that none of their inputs crossed the boundary. No round was discarded and no
fallback policy was substituted.

## Interpretation

The strongest transformer advantage appears at 40%–80% penetration and when
at least 40% of publicly remaining cards are high cards. The transformer is
slightly worse in the 20%–40% penetration band, at true counts -2 to -1 and
2 to 3, and when the high-card share is 37%–38.5%. Those slices are the first
places to inspect before changing training.

Both policies have negative mean *unscaled* profit per round because most
ordinary blackjack states retain a house edge. Their positive bankroll growth
comes from varying wager fractions with favorable composition. This is why the
predeclared primary metric is compounded log-bankroll growth rather than raw
win rate or flat-bet profit.

The realized bankroll endpoints are enormous because each policy compounds a
fraction of current bankroll for more than five million rounds. They are useful
for visualizing growth rates, not as a literal claim about casino limits,
table maximums, liquidity, or practical bet camouflage. The confidence
interval on per-shoe paired log growth is the inferential result.

Files:

- [`aggregate.json`](aggregate.json) contains the pooled estimates, confidence
  intervals, exact round totals, breakdowns, context instrumentation, and
  downsampled trajectories.
- [`trajectory.svg`](trajectory.svg) compares the realized compounded
  trajectories on a logarithmic bankroll axis.
- [`../../notebooks/03_bankroll_evaluation.ipynb`](../../notebooks/03_bankroll_evaluation.ipynb)
  recreates the charts and explains the evaluation as first-person notes.
