# Card-Counting Transformer: Full Project Report

## Executive summary

I built a causal transformer from scratch to test whether a model can learn
composition-dependent blackjack play and betting directly from genuinely
visible card history. The final retained model has 831,488 parameters and a
29-token vocabulary. It learned from 409,207 training decisions generated
across 4,000 six-deck shoes and was selected only on a fixed 50,925-decision
validation split.

The result is positive under the experiment I declared:

- the transformer reached 97.56% overall validation accuracy;
- it reached 97.52% play accuracy and 70.18% accuracy on states where the
  composition-dependent oracle departed from basic strategy;
- it reduced mean play regret to 0.000473 wager units;
- in a fresh 116,000-shoe evaluation, it completed 5,015,984 rounds while the
  Hi-Lo control completed 5,025,662;
- it earned an estimated 1.737 minimum-bet units per 100 rounds, versus 1.258
  for Hi-Lo;
- its paired mean log-bankroll advantage over Hi-Lo was `0.00016734` per shoe,
  with a 95% confidence interval of `[0.00006587, 0.00026880]`.

Because that full interval is above zero, the transformer meets the
outperformance criterion I wrote down before the scale evaluation. This is
evidence against the project's specific documented Hi-Lo control, not a claim
that it beats every possible professional counting system or that the
exponentially compounded bankroll path is achievable in a real casino.

I did all local computation on a MacBook Pro with an Apple M4 Max and 64 GB of
unified memory. I came into the project with no prior experience doing this
end-to-end and built it with OpenAI Codex as my coding and research partner. In
the process, I used three complete Codex weekly reset allocations.

## What I built

### A typed casino engine

The infrastructure is a strictly typed blackjack package with:

- exact card, hand, and multi-Ace valuation;
- a six-deck shoe, deterministic shuffling, burn card, cut card, and replay;
- the complete one-player American-hole-card H17 round state machine;
- hit, stand, double, split, surrender, and insurance legality;
- four-hand split limits, double after split, no resplitting Aces, and
  one-card-only split Aces;
- exact `Fraction` settlement for naturals, ordinary wins, pushes, surrender,
  insurance, doubles, and splits;
- typed internal events plus redacted public events and snapshots;
- public card history that never exposes a burn card or unrevealed dealer hole
  card.

The suite now contains 317 passing automated tests across the engine, oracle,
dataset, model, training, and evaluation code.

### A reproducible mathematical labeling pipeline

I first explored exact rational and exhaustive solvers. They were valuable
audit tools but too slow for production-scale labels. I then implemented a
native seeded Monte Carlo evaluator:

- bet labels use one million complete-round rollouts under fixed six-deck H17
  basic-strategy continuation;
- play labels use one million rollouts per legal first action followed by the
  same documented fixed continuation;
- insurance remains an inexpensive exact composition calculation;
- every row retains legal actions, action values, empirical return
  distributions, random seeds, uncertainty, and provenance;
- SQLite caches canonical oracle states across worker processes;
- atomic shoe checkpoints and complete-shoe sharding make interruption and
  resumption safe.

This is an explicit methodological concession. The labels approximate the
best first action under a fixed continuation policy; they do not pretend to be
an exhaustive recursively optimal policy at every later node.

### A clean supervised-learning experiment

Each example consists of public card tokens, the complete current decision
state, and one legal target token. The model never receives the burn card,
unrevealed hole card, private engine state, or future cards.

The retained transformer uses:

- a 256-token context;
- 128-dimensional embeddings;
- four causal-attention layers with four heads;
- 512-dimensional feed-forward layers;
- query-relative positions;
- deterministic card-order permutation augmentation;
- natural training sampling;
- masked legal-action cross-entropy.

The dataset contains only target actions. It does not include intentionally bad
actions as target examples or apply reinforcement-learning punishment. Expected
value and regret are retained for evaluation and possible later value-aware
post-training.

## Dataset evidence

The corpus grew through bounded milestones:

| Corpus | Shoes | Decisions | Measured generation wall time |
| --- | ---: | ---: | ---: |
| Integration v4 | 100 | 10,206 | 11m 13s |
| Learning-curve v5 | 1,000 | 102,484 | 54m 25s |
| Final v6 | 5,000 | 511,145 | 3h 36m 07s |

The final v6 split is:

| Split | Shoes | Decisions |
| --- | ---: | ---: |
| Training | 4,000 | 409,207 |
| Validation | 500 | 50,925 |
| Test | 500 | 51,013 |

Shoes are assigned to a split before examples are generated, preventing
history fragments from the same shoe from leaking across splits. The full
quality audit found no schema, provenance, duplicate-index, or shoe-split
integrity errors. The generated test split remains sealed: I have not used or
reported its model metrics.

The production cache contains 498,356 unique oracle states:

- 212,657 bet states, representing 212.657 billion complete-round rollouts;
- 269,182 play states with 971,486 legal first-action evaluations,
  representing 971.486 billion rollouts;
- 16,517 exact insurance states.

The production cache therefore records 1.184143 trillion seeded Monte Carlo
rollouts. Fresh-cache worker benchmarks add 15.244 billion rollouts and the
bet-token pilot adds 2.4 million, for approximately **1.1994 trillion
complete-round rollouts** performed on the MacBook. This is a workload count,
not a floating-point-operation count: each rollout contains variable random
draws, branches, hand evaluations, and settlement work.

## Training evidence

I retained 16 training artifacts covering natural versus balanced sampling,
absolute versus query-relative positions, card-order augmentation, nested
100/300/1,000-shoe curves, and the final 1,000/5,000-shoe comparison.

Together these artifacts record:

- 220 epochs;
- 12,714,735 training-example presentations;
- 2,688,275 validation-example presentations;
- 28,999.98 seconds, or **8h 03m 20s**, of measured epoch wall time.

The final controlled comparison used the same model initialization, optimizer,
15-epoch budget, and 50,925-row validation set:

| Policy/model | Training shoes | Overall accuracy | Play accuracy | Composition deviations | Mean play regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| H17 Hi-Lo | — | 95.04% | 95.93% | 40.2% | 0.00125 |
| 1,000-prefix transformer | 779 | 96.15% | 95.81% | 60.8% | 0.00191 |
| Full transformer | 4,000 | **97.56%** | **97.52%** | **70.2%** | **0.00047** |

Scaling to 4,000 training shoes added 1.41 percentage points overall and
lowered play regret by 75% relative to the prefix model. Against Hi-Lo, the
full transformer added 1.59 points of play accuracy and reduced mean play
regret by about 62%.

Across 200,305 changed-input permutation comparisons, the full model preserved
its prediction 99.11% of the time. Chronological and permuted validation
accuracies were 97.56% and 97.57%, so invariance did not conceal an accuracy
tradeoff.

### Rough neural calculation estimate

The training code did not record hardware FLOPs. A common dense-transformer
back-of-the-envelope estimate is roughly six parameter operations per token
for a training pass, with fewer for validation. Using 831,488 parameters,
12.7 million training-example presentations, and plausible effective sequence
lengths between roughly 110 and the 256-token padded limit gives an
order-of-magnitude total of **about 8 to 20 quadrillion neural floating-point
operations** across the recorded experiments.

That range is deliberately broad. It omits or approximates attention's
sequence-length-dependent cost, padding, fused Metal kernels, optimizer work,
and the smaller earlier configurations. It should be read as scale, not as a
benchmark-grade FLOP measurement.

## Fresh evaluation evidence

The final comparison uses simulation seed `20260801` and deterministic shoe IDs
0 through 115,999. These shoes are generated separately from dataset creation.
Each policy receives the same shuffled order and cut-card position but advances
its own cursor, so actions can change later card allocation.

The primary run contains:

| Measurement | Count |
| --- | ---: |
| Fresh six-deck shoes | 116,000 |
| Transformer rounds | 5,015,984 |
| Hi-Lo rounds | 5,025,662 |
| Combined policy-rounds | **10,041,646** |

Including retained pilots, scale checks, and concurrency benchmarks, the
evaluation artifacts represent 132,120 shoe evaluations, 5,713,029 transformer
rounds, and 5,724,105 Hi-Lo rounds: **11,437,134 total policy-rounds** actually
simulated while developing the evaluation.

### EV per round and per hour

One minimum-bet unit is `<BET_MIN>`, or 0.1% of bankroll. The bet vocabulary is
a 1-to-13 spread: 1, 5, 9, or 13 minimum units.

| EV estimate | Transformer | Hi-Lo |
| --- | ---: | ---: |
| Minimum-bet units/round | 0.01737 | 0.01258 |
| Minimum-bet units/100 rounds | **1.737** | **1.258** |
| 95% CI, units/100 rounds | [1.447, 2.027] | [0.993, 1.524] |
| Bankroll EV/100 rounds | 0.1737% | 0.1258% |

The descriptive gap is 0.479 minimum-bet units per 100 rounds, about 38% more
EV under this bet spread.

EV per hour depends on table speed:

| Rounds/hour | Transformer units/hour | Hi-Lo units/hour |
| ---: | ---: | ---: |
| 60 | 1.042 | 0.755 |
| 100 | 1.737 | 1.258 |
| 150 | 2.606 | 1.887 |

At a $25 minimum and 100 rounds/hour, those estimates are approximately
$43.43/hour and $31.46/hour. That example implies a $25-to-$325 spread and does
not subtract travel, tipping, imperfect execution, heat, backoffs, table
limits, or other real-world costs.

### Predeclared paired result

I retain paired log-bankroll growth as the inferential test because the wager
fractions were selected from half-Kelly calculations and returns compound.

The transformer-minus-Hi-Lo result is:

- mean advantage: `0.000167336` log-growth units per shoe;
- standard error: `0.000051766`;
- 95% confidence interval: `[0.000065874, 0.000268798]`;
- sample: 116,000 paired shoes.

The entire interval is above zero. This satisfies the predeclared criterion
for transformer outperformance. EV per 100 rounds is the clearer
card-counting headline; it is a conversion of the already-retained arithmetic
round-return estimate rather than a replacement metric chosen after seeing the
answer.

### Where the model wins and loses

The transformer is strongest relative to Hi-Lo:

- at 40%–60% shoe penetration;
- at 60%–80% penetration;
- when at least 40% of publicly remaining cards are Aces or ten-valued.

It remains slightly worse:

- at 20%–40% penetration;
- at true counts -2 to -1;
- at true counts 2 to 3;
- when public remaining high-card share is 37%–38.5%.

These are diagnostic slices, not new hypothesis tests. They identify where
interpretability and targeted post-training could be useful without changing
the primary result.

## The Hi-Lo comparison

The control is stronger than a static basic-strategy chart. It uses:

- six-deck H17 basic strategy;
- a documented H17 subset of Illustrious 18/Fab 4 play deviations;
- insurance at true count +3;
- a true-count bet ramp into the same four wager levels as the transformer.

The transformer result should therefore be described as outperforming this
specific capable Hi-Lo implementation. It does not prove superiority over all
index sets, risk-of-ruin systems, side-count strategies, Wonging rules, or
casino conditions.

## Stanford CS230 comparison

Two primary Stanford CS230 reports provide useful context:

- [2018 Deep Q-Learning for Blackjack](https://cs230.stanford.edu/files_winter_2018/projects/6940282.pdf)
- [2021 Learning Blackjack with PPO](https://cs230.stanford.edu/projects_fall_2021/reports/103066753.pdf)

The 2021 PPO project is the closest conceptual precedent. It encoded dealt-rank
counts, learned play and dynamic betting, and reported positive average reward
against random, tabular, rule-based, Hi-Lo, and neural controls over 50,000
episodes. It supports the broad finding that a learned policy can exploit
composition and betting information.

The numeric returns are not directly comparable. Its primary environment used
one deck, omitted splitting, used early surrender and a dealer-hit-17 rule,
and had a different action and reward design. This project uses six decks,
American hole-card rules, H17, splits, late surrender, insurance, exact
settlement, and visible history tokens. Stanford trained with deep
reinforcement learning; I trained a small transformer by supervised imitation
of reproducible Monte Carlo labels.

The defensible comparison is qualitative: both learned systems exceeded their
own documented controls under their own environments. This project adds a much
larger final evaluation, full casino mechanics, strict public/private
visibility, deterministic replay, exact payout settlement, and a paired
confidence interval.

## Compute accounting

I distinguish physical wall time from parallel worker time.

| Stage | Physical active wall time | Basis |
| --- | ---: | --- |
| Production dataset generation | 4h 41m 45s | Three measured corpus runs |
| Generation benchmarks and abandoned exact approaches | about 20–30m | Measured worker benchmark plus bounded timeouts/smoke runs |
| Recorded model training | **8h 03m 20s** | Sum of `elapsed_seconds` for all 220 epochs |
| Bankroll evaluation and scale benchmarks | about 1.9h | Reconstructed from report timestamps and measured benchmark durations |
| **Total major active compute** | **about 15 hours** | Excludes ordinary editing, tests, plotting, and idle gaps |

The production generation jobs used four or eight processes. Their 4h 42m of
physical elapsed time corresponds to about 36.8 worker-process-hours; the
fresh-cache concurrency benchmark adds about 0.56, for roughly **37.4
CPU-worker-hours**. Parallelism reduced elapsed time but did not make that CPU
work disappear.

Training used one Metal/MPS process on the M4 Max. Evaluation mixed batched MPS
inference with CPU engine simulation and used up to eight concurrent evaluator
processes. Worker-hours are not added to MPS wall time because those processes
share one physical GPU and memory system.

These totals are conservative. They exclude many short unit-test runs, notebook
execution, compilation, static analysis, plotting, file compression, and
interactive research. They also exclude human and Codex thinking time. The
auditable large-workload summary is:

- approximately 1.1994 trillion complete-round Monte Carlo label rollouts;
- 220 recorded training epochs;
- 15.4 million total recorded train/validation example presentations;
- roughly 8–20 quadrillion neural tensor FLOPs by a broad formula-based
  estimate;
- 11.44 million retained evaluation policy-rounds;
- about 15 physical hours of major active MacBook compute;
- about 37.4 CPU-worker-hours for the parallel generation stages alone.

## One observed boundary correction

One live decision in the final scale shard contained 257 tokens, one more than
the retained model's 256-token context. I did not discard the hand or replace
the model with Hi-Lo. The evaluator now removes only the minimum number of
oldest visible-history card tokens while preserving the history marker,
complete current hand, dealer upcard, structural markers, and query.

In the resumed shard, this happened exactly once among 233,436 transformer
decisions and removed exactly one history token. The other 40 reports used the
former fail-on-overflow behavior; because they completed, none contained an
overlength decision. The correction is documented because it was added after
the rare failure was observed.

## Limitations

- Production play labels are Monte Carlo estimates under fixed continuation,
  not an exhaustive recursive optimum.
- The final model was selected on validation and the generated test split
  remains sealed.
- The scale result compares one transformer with one declared Hi-Lo control.
- EV/hour is a table-speed conversion, not a field study.
- The simulator omits table limits, heat, countermeasures, travel costs,
  errors, and bankroll constraints beyond fractional betting.
- Composition breakdowns are diagnostic and were inspected after the primary
  result.
- One exceptionally long context required a one-token oldest-history
  truncation.

## Reproducibility

The repository contains:

- the complete typed engine, dataset, training, Hi-Lo, evaluation, and
  aggregation source;
- deterministic seeds and shoe-ID ranges;
- strict tests and Pyright configuration;
- the selected model artifacts;
- the compact aggregate JSON and charts;
- an executed educational notebook that recreates the analysis;
- a GitHub release with train/validation data and the atomic evaluation-report
  archive.

The evaluation archive contains the initial 25,000-shoe report and all 40
atomic 2,275-shoe reports. Those summaries are sufficient to reproduce the
pooled within- and between-shard variance exactly. The underlying shoes do not
need to be stored because the public seed, shoe IDs, shuffle implementation,
and selected model recreate them deterministically.

## Conclusion

The experiment achieved its intended milestone. A small causal transformer,
trained from scratch on public blackjack history, learned more than static
basic strategy and more than a documented Hi-Lo control. It learned rare
composition-dependent deviations, reduced decision regret, remained almost
invariant to semantically irrelevant card ordering, and produced higher
estimated EV and statistically higher paired log-bankroll growth over more
than five million rounds per policy.

The most useful next work is not another blind scale run. It is to inspect what
the network represents: probe hidden states for running count, true count, and
rank composition; visualize attention over exposed cards; analyze the narrow
regions where Hi-Lo is still ahead; and test whether value-aware post-training
can improve costly mistakes without damaging common play.
