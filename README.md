# Learning How Transformers Work Through Blackjack

## Why I Am Doing This

I am an Applied AI Engineer, but most of my work happens above the level where
the underlying model mechanics are visible. I want a better understanding of
how an AI model works at a lower level: how information becomes tokens, how a
transformer represents context, how it learns from a loss signal, and how its
predictions can be inspected rather than treated as magic.

I am building the model myself and learning each part through small,
explainable experiments. The goal is not merely to produce a working model. I
want to understand why it works, where it fails, and what it has learned
internally.

## Why Blackjack?

Blackjack is fun, but it also makes an unusually good learning environment.

I can generate as much training data as I need instead of depending on a
fixed text corpus. Every generated example can be reproduced from a known shoe
and a fixed set of rules. The quality of a prediction can be measured against
an explicit mathematical strategy instead of judged subjectively.

Card counting also gives the transformer a real reason to use context. A
decision depends not only on the cards currently on the table, but on the cards
that have already been exposed. This creates an experiment with a concrete,
testable question:

> Can a transformer learn to represent the changing composition of a blackjack
> shoe and use that information when it bets and plays?

---

## Methodology

### What I Want the Model to Do

I will train one causal transformer to make both betting and playing decisions.
It will use one language-model output head rather than separate models or
free-form text generation.

The transformer will predict exactly one special decision token at a time:

- Bet tokens: `<BET_MIN>`, `<BET_LOW>`, `<BET_MEDIUM>`, or `<BET_HIGH>`
- Play tokens: `<HIT>`, `<STAND>`, `<DOUBLE>`, `<SPLIT>`, or `<SURRENDER>`
- Insurance tokens: `<INSURANCE>` or `<NO_INSURANCE>`

The blackjack engine will restrict prediction to the legal tokens for the
current decision. This means the model never needs to produce prose, JSON, or a
tool call. Its output is a token ID that maps directly to a typed blackjack
decision.

Using the same transformer for every decision is important to the experiment.
The model must build one internal representation of the shoe and use it for
both bet sizing and playing strategy.

### Model Input

I will provide only information that can affect the decision.

For a bet, the input contains the values of all cards exposed since the most
recent shuffle:

```text
<HISTORY> 10 4 6 A 3 10 8 5 ... <BET_QUERY>
```

For a playing decision, the input also identifies the current player cards and
the dealer's visible card:

```text
<HISTORY> 10 4 6 A 3 10 8 5 ...
<CURRENT_HAND> <PLAYER> 10 6 <DEALER> 10 <PLAY_QUERY>
```

For an insurance decision:

```text
<HISTORY> 10 4 6 A 3 10 8 5 ...
<CURRENT_HAND> <PLAYER> 10 6 <DEALER> A <INSURANCE_QUERY>
```

`<HISTORY>` contains every exposed card that is not represented in the current
player hand or dealer upcard. This normally means cards from completed hands,
but it can also include visible cards from another hand created by a split.
Cards in `<CURRENT_HAND>` are not repeated in `<HISTORY>`, so each visible card
is represented exactly once.

Historical cards do not need player or dealer labels. Once a card has been
exposed, only its value matters to the composition of the remaining shoe.
Suits are irrelevant without side bets, and Jack, Queen, and King are
equivalent to `10`, so all ten-valued cards share one token.

The model will not receive:

- the running count;
- the true count;
- the number of decks remaining;
- previous bets, actions, results, or payouts;
- bankroll information;
- card suits;
- hidden cards; or
- casino-rule tokens.

The running count, penetration, and true count are deliberately withheld. If
the model needs those concepts, it must derive useful representations of them
from the exposed cards. The casino rules are also omitted because every example
uses the same rules; constant rule tokens would consume context without adding
information.

### Model Output

The transformer produces logits with shape:

```text
(batch_size, sequence_length, vocabulary_size)
```

Only the logits at the final query position are used for a decision:

```text
(batch_size, vocabulary_size)
```

Tokens that are not valid for the current decision are masked before selecting
the prediction. The highest-scoring remaining token maps directly to an action
or bet. There is no ambiguous string parsing step.

Training loss will be calculated on decision tokens rather than on every event
in the sequence. The exposed cards are context; the bet, play, and insurance
choices are the behavior the model is learning.

### Casino Rules

I will model a representative six-deck American casino shoe with one player
and one dealer.

| Rule | Setting |
| --- | --- |
| Players | One player against one dealer |
| Decks | Six |
| Natural blackjack | Pays 3:2 |
| Other winning hands, including 21 after a split | Pay 1:1 |
| Dealer soft 17 | Dealer hits |
| Dealer hole card | American style |
| Dealer peek | Dealer checks for blackjack with a 10 or Ace showing |
| Double down | Allowed on any initial two cards |
| Double after split | Allowed |
| Splitting | Up to four hands |
| Resplitting Aces | Not allowed |
| Split Aces | One additional card per hand |
| Late surrender | Allowed after the dealer checks for blackjack |
| Insurance | Offered when the dealer shows an Ace |
| Side bets | None |
| Shoe penetration | Randomized between 70% and 80% |
| Burn card | One unseen card after each shuffle |

Limiting the table to one player keeps the event sequence and split-hand state
easy to inspect. It does not remove the central card-counting problem: every
visible card still changes the composition of the remaining shoe.

A natural blackjack means an Ace and a ten-valued card in the original
two-card hand. It pays 3:2. A 21 made with additional cards or after a split is
an ordinary winning hand and pays 1:1.

The dealer's hole card remains hidden until it would be revealed in a real
game. If a decision follows a dealer peek, the existence of that decision
already establishes that the dealer does not have blackjack; a redundant
`<NO_BLACKJACK>` token is unnecessary.

### Dataset Generation

#### Ground-Truth Decisions

For every play and insurance decision, I will calculate the expected value of
each legal action from the information available to the player at that moment.
The calculation will account for every card that has been exposed and average
over cards that remain unknown, including the dealer's hidden card.

For play and insurance, the training target will always be the legal action
with the highest expected value in that exact state. I will not use a
basic-strategy chart, a Hi-Lo decision table, or the action produced by another
learned model as ground truth. Those approaches compress or approximate the
underlying probabilities; this experiment is intended to teach the transformer
the optimal composition-dependent play decision directly.

Alongside the selected action, the generated dataset will retain the expected
value of every legal action as evaluation metadata. This will make it possible
to distinguish a costly mistake from a prediction whose expected value was
nearly tied with the optimum.

#### Ground-Truth Bet Sizing

I will use half Kelly to determine the target bet size. For each pre-deal shoe
composition, I will first calculate the full-Kelly bankroll fraction that
maximizes expected logarithmic growth:

```text
full_kelly = argmax E[log(bankroll_after_round)]
```

The target fraction will be half that amount:

```text
target_bet_fraction = 0.5 × full_kelly
```

Full Kelly maximizes long-run bankroll growth, but it also produces substantial
volatility. Half Kelly retains approximately 75% of the growth rate while
cutting variance approximately in half. I prefer that tradeoff because it gives
the betting policy a precise mathematical objective without making maximum
growth the experiment's only concern.

The calculation uses the complete empirical distribution of possible round
returns, not the shortcut of dividing estimated advantage by an assumed
variance. The distribution incorporates natural blackjack, ordinary wins and
losses, pushes, surrender, insurance, doubles, resplits, and correlated
split-hand outcomes.

For tractability, production bet distributions and play action values use the
documented fixed H17 basic-strategy continuation policy described below. The
current play action is still evaluated composition by composition: I force
each legal first action, simulate the rest of that round under the fixed
policy, and select the action with the greatest empirical expected return.
This means the two tasks answer related but distinct questions:

- A play label asks which legal action has the highest expected value in the
  current composition when later actions in that rollout follow fixed H17
  basic strategy.
- A bet label asks what wager is justified by the exact pre-round composition
  if subsequent play follows the fixed H17 policy.

The model will predict a discrete bankroll fraction rather than a dollar
amount. The blackjack engine will convert that fraction into a wager using the
current bankroll, so bankroll does not need to appear in the model input.

I will select the discrete bet-token scale only after a pilot analysis of the
half-Kelly fractions produced by representative shoe compositions. This avoids
choosing the minimum, maximum, or spacing of the betting vocabulary by
intuition.

#### Bet-Token Pilot Methodology

The pilot uses 96 pre-deal compositions sampled from 24 deterministic six-deck
shoes. Four samples from each shoe are stratified across the playable
penetration range. For each composition, I simulate 25,000 complete rounds, for
2.4 million seeded rollouts in total. The simulator retains the complete return
outcomes produced by naturals, ordinary wins and losses, pushes, surrender,
insurance, doubles, and correlated split hands.

I make one explicit approximation for this pilot: rollout play follows the
[Blackjack Apprenticeship H17 basic-strategy chart](https://www.blackjackapprenticeship.com/wp-content/uploads/2024/09/H17-Basic-Strategy.pdf),
including double after split and late surrender, instead of recomputing the
exact composition-dependent optimal policy inside every rollout. Exact
rational enumeration of a realistic composition with sequential split-hand
correlations takes longer than one minute per state, which makes a
representative batch impractical at this stage. Insurance remains
composition-dependent because its exact decision is the inexpensive comparison
between the visible ten-valued-card probability and the one-third break-even
threshold.

The production dataset now adopts the same fixed-policy boundary for bet
labels. The pilot estimates a useful bet range and resolution, while production
uses many more rollouts per state and a separately versioned random stream.
Both are deterministic from their configuration, report Monte Carlo standard
errors, and retain their complete empirical return distributions.

After correctly marginalizing the one unknown burn card, the observed
continuous half-Kelly fractions ranged from 0% to 0.995% of bankroll. The mean
standard error of the simulated expected return was 0.722 percentage points,
so a very fine token grid would claim more precision than the pilot supports.
I retained the four-token vocabulary because it covers the observed range,
keeps the output task compact, and has a 95th-percentile absolute rounding
error of approximately 0.15 percentage points. The upper token deliberately
provides headroom beyond this small pilot rather than being justified by a
single rare sample.

The token names are intentionally categorical. Their bankroll fractions are a
versioned experiment mapping rather than semantics the transformer is expected
to infer from token spelling. Changing the mapping requires regenerating the
dataset and retraining the model.

| Token | Bankroll fraction |
| --- | ---: |
| `<BET_MIN>` | 0.10% |
| `<BET_LOW>` | 0.50% |
| `<BET_MEDIUM>` | 0.90% |
| `<BET_HIGH>` | 1.30% |

Across the 96 sampled compositions, the four classes contained 80, 10, 6, and
0 examples. The imbalance is itself useful evidence: most representative
states have no estimated positive edge and therefore map to the minimum wager,
while 96 states are not enough to populate the deliberately reserved upper
class. The pilot notebook compares this vocabulary against finer grids and
shows that additional classes improve rounding error less than the Monte Carlo
uncertainty would justify.

#### Dataset Pipeline Methodology

I represent one model decision as one row. The only model-facing fields are an
ordered `input_tokens` sequence and one `target_token`. Everything else is
evaluation or reproducibility data and will be excluded by the training data
loader.

An abbreviated play row can look like this:

```json
{
  "input_tokens": [
    "<HISTORY>", "10", "4", "6", "A",
    "<CURRENT_HAND>", "<PLAYER>", "10", "6",
    "<DEALER>", "10", "<PLAY_QUERY>"
  ],
  "target_token": "<STAND>",
  "behavior_token": "<HIT>",
  "metadata": {
    "legal_target_tokens": [
      "<HIT>", "<STAND>", "<DOUBLE>", "<SURRENDER>"
    ],
    "shoe_composition": [23, 24, 24, 23, 24, 22, 24, 24, 24, 93],
    "unseen_unavailable": 2,
    "evaluation_method": "seeded_monte_carlo_fixed_h17_continuation",
    "action_values": [
      {
        "token": "<STAND>",
        "expected_profit": {"numerator": -53, "denominator": 100},
        "monte_carlo": {
          "seed": 10407177902910792742,
          "rollouts": 1000000,
          "expected_profit_standard_error": 0.00084,
          "expected_profit_confidence_interval_95": [-0.53165, -0.52835]
        }
      }
    ]
  }
}
```

The real row includes an `action_values` object for every legal token and a
complete empirical return distribution inside each object; I show only one
action and omit its larger distribution in this compact example.

The different target and behavior tokens are intentional. Insurance uses the
exact rational oracle. Play and bet targets use their production Monte Carlo
methods described below, with the exact play oracle retained as a verifier.
The behavior token is the action used to continue the simulated shoe. By
default, I take a uniformly sampled non-optimal legal action 20% of the time
and the target action otherwise. This seeded exploration visits states that a
purely greedy trajectory would never reach without corrupting their labels.
Bet decisions do not need an exploratory behavior because the normalized
wager does not change the cards that become visible.

For a play decision after a split, I send the oracle the active hand, every
pending hand, and every resolved hand. This preserves the shared dealer
outcome, finite-shoe card depletion, and the four-hand split limit instead of
labeling each split hand as an unrelated round. The retained action values are
therefore complete round-profit distributions, including correlated split-hand
outcomes.

I construct the oracle composition by subtracting only publicly visible cards
from a fresh six-deck composition. The burn card, the current dealer hole card,
and any hole card that remained hidden because every player hand busted or
surrendered stay inside that public composition. I separately record how many
of those unknown cards are no longer physically available. The oracle
marginalizes their identities, so a hidden card changes the possible future
shoe without leaking its value into either the tokens or metadata.

The label methods are explicit in every row:

- A bet row retains the empirical production round-return distribution,
  continuous half-Kelly fraction, log-growth value of each bet token, selected
  discrete fraction, rollout seed and count, expected-return standard error,
  and approximate 95% confidence interval. Its `evaluation_method` is
  `seeded_monte_carlo_fixed_h17_basic_strategy`.
- An insurance row retains the exact return distribution and expected profit
  for taking and declining. A break-even tie resolves to
  `<NO_INSURANCE>`.
- A play row retains the empirical complete-round return distribution,
  expected profit, rollout seed and count, standard error, and approximate 95%
  confidence interval for every legal action. Its `evaluation_method` is
  `seeded_monte_carlo_fixed_h17_continuation`. Empirical ties follow the stable
  action order: hit, stand, double, split, then surrender.
- Every ten-valued physical rank is serialized as `10`, while the manifest
  retains the physical rank sequence needed for exact engine replay.

#### Production Bet-Oracle Methodology and Scoped Concession

I estimate each production bet distribution with 1,000,000 complete seeded
round rollouts under the fixed six-deck H17 basic-strategy policy. The policy
includes late surrender, double on any initial two cards, double after split,
up to four hands, no resplitting Aces, and one-card-only split Aces. Naturals,
ordinary payouts, doubles, surrender, insurance, and every split hand
contribute to one correlated round return.

Insurance remains composition-dependent. At an Ace upcard, the simulation
takes insurance exactly when the public probability of a ten-valued hole card
exceeds its one-third break-even point. Unknown burned cards and previously
unrevealed holes are sampled as unavailable before each rollout, but their
sampled identities are not allowed to influence the insurance decision. This
preserves the same information boundary seen by the model.

The simulation uses an explicitly implemented SplitMix64 stream and
rejection-sampled bounded draws. I derive a stable 64-bit seed from the master
rollout seed, public composition, and count of unknown unavailable cards. I do
not rely on an undocumented standard-library shuffle, so the complete outcome
histogram can be replayed exactly.

The stored histogram uses exact integer rollout counts divided by the recorded
rollout count. I calculate half Kelly from that complete empirical
distribution. Each row also records the estimated expected-return standard
error and an approximate 95% confidence interval. These quantify sampling
uncertainty; they do not convert the fixed playing policy into an optimal one.

The important concession is therefore about the policy being valued, not the
visible shoe information. The bet oracle still responds to the exact remaining
composition through natural frequency, dealer outcomes, double and split
returns, surrender, and insurance. It omits only the additional return gained
by changing ordinary play decisions as the shoe becomes depleted.

Published results give a useful scale for this bias, although none exactly
matches my rules:

- The Wizard of Odds calculation for a fresh six-deck H17/DAS game reports
  only a 0.0031-percentage-point benefit from composition-dependent rather than
  total-dependent basic play. The benefit is tiny before meaningful depletion:
  [Total Dependent vs. Composition Dependent Basic](https://wizardofodds.com/games/blackjack/composition-dependent-benefit/).
- A 2025 study of 50,000 sampled eight-deck compositions at 75% penetration
  reported average round EV of -0.79% under fixed basic strategy and -0.56%
  under its semi-optimal composition-dependent policy. Positive-EV rounds
  increased from 17.7% to 21.9%. Its rules differ materially—S17, no surrender
  or insurance, no DAS, and approximate splitting—so I treat the
  0.23-percentage-point difference as scale evidence rather than a correction
  factor:
  [Optimal Blackjack Betting Strategies Through Dynamic Programming and Expected Utility Theory](https://arxiv.org/abs/2505.00724).
- A single-deck H17 experiment at 50% penetration reported about 0.449
  percentage points of improvement from perfect composition-dependent
  deviations, with roughly 0.086 points attributed to insurance. Single deck
  intentionally provides much more information than this six-deck experiment,
  so it is closer to an upper-bound example:
  [Rust Blackjack Composition Analyzer and Simulator](https://github.com/joshuaprince/blackjack_composition).

My working expectation is that fixed-policy bet labels conservatively
underestimate favorable-shoe EV by roughly 0.2 to 0.4 percentage points on
average, with almost no early-shoe effect and larger errors in unusual,
deeply-depleted compositions. This is an evidence-informed planning range, not
a measured result for my exact rules. Because optimal play can always choose
the fixed-policy action, the policy concession can systematically make the bet
smaller, but it cannot justify a larger bet than the corresponding optimal-play
distribution would justify. I accept that lower-growth, lower-risk bias to make
dataset generation computationally feasible.

The native rollout kernel is deliberately small. Tests compare its aggregate
distribution with the independent Python pilot simulator, verify exact seeded
replay, exercise unavailable-card marginalization, and cover deterministic
edge cases. The rational CDP oracle remains the exact verifier for play and the
production source for insurance.

#### Bounded Monte Carlo Play Validation

The exact play oracle still has a long computational tail, particularly around
large split trees. Before replacing production play enumeration, I ran a
bounded experiment to isolate the amount of error caused by sampling alone.
The reproducible implementation is
`blackjack.analysis.run_play_sampling_validation`.

I selected 14 full-shoe hard and soft states covering hit, stand, double, and
surrender decisions. Six deliberately have less than 0.01 wager units between
the exact best and second-best actions. For every state, I first calculated the
complete rational action-return distributions. I then sampled each legal
action distribution independently at four budgets and repeated the selection
200 times. Regret is always evaluated with the exact rational values, not the
sample means.

| Samples per legal action | Exact action agreement | Agreement when exact gap ≥ 0.01 | Mean exact EV regret | 95th-percentile regret |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 72.21% | 86.62% | 0.3408 pp | 2.7964 pp |
| 10,000 | 82.96% | 97.81% | 0.0948 pp | 0.7747 pp |
| 100,000 | 91.54% | 100.00% | 0.0231 pp | 0.0134 pp |
| 1,000,000 | 96.39% | 100.00% | 0.0015 pp | 0.0000 pp |

Here “pp” means percentage points of the initial wager. At one million samples,
the largest observed regret was 0.3412 pp and came from a near-tied state. More
than 95% of selections had exactly zero regret because they matched the
rational action.

This is intentionally an optimistic lower bound for a production Monte Carlo
play oracle. The experiment samples already-solved exact distributions, so
every branch inherits optimal continuation play for free. It excludes pairs,
resplits, pending split hands, depleted-shoe extremes, and the additional
search error of estimating later decisions. It establishes that one million
samples are enough to recover materially separated actions in this small
corpus; it does not yet establish that a particular rollout or tree-search
policy reproduces unrestricted CDP at the same rate.

I retained the exact oracle as a verifier and used these metrics as the
sampling-only baseline for the real rollout comparison below.

#### Production Play-Oracle Methodology and Measured Concession

For every production play state, I force each legal first action and run
1,000,000 complete-round rollouts. Later actions in each rollout follow the
same documented H17 basic-strategy policy used by the bet simulator. Dealer
play, doubles, surrender, all split hands, and their shared dealer outcome are
settled together. The resulting label therefore estimates
`Q_fixed(state, first_action)` rather than claiming to enumerate
`Q_optimal(state, first_action)`.

The simulator samples the current hole card first, conditioning it on the
public negative peek when applicable. It then samples any other burned or
previously unrevealed cards as unavailable. Their identities influence the
physical cards left to draw but never enter the public state, tokens, or policy
inputs. Each candidate action uses the same deterministic per-rollout random
stream until its path diverges, a common-random-numbers design that reduces
comparison noise. Every action stores its replay seed, rollout count, empirical
return distribution, standard error, and confidence interval.

I compared this real evaluator—not samples from pre-solved distributions—with
the exact rational oracle on the same 14-state full-shoe corpus:

| Rollouts per legal action | Exact action agreement | Agreement when exact gap ≥ 0.01 | Mean exact EV regret | Maximum exact EV regret | Mean absolute action-value error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 78.57% | 100.00% | 0.1391 pp | 0.8321 pp | 0.5777 pp |
| 100,000 | 92.86% | 100.00% | 0.0010 pp | 0.0134 pp | 0.3014 pp |
| 1,000,000 | 92.86% | 100.00% | 0.0010 pp | 0.0134 pp | 0.0706 pp |

At one million rollouts, the only action disagreement was a three-card hard 16
against a dealer 10. The exact oracle preferred stand to hit by 0.0134
percentage points of the initial wager. All eight states whose best action led
by at least one percentage point agreed. This result supports the production
switch for ordinary hard/soft decisions, but the corpus is deliberately
bounded: it does not prove the same agreement rate for every depleted
composition or large resplit tree. Native tests separately exercise current,
pending, and resolved split hands so those outcomes are represented correctly.

The remaining concession is policy bias. Sampling more rollouts narrows Monte
Carlo noise but cannot turn fixed continuation into rational continuation.
Greedily selecting the best first action against a fixed policy is one policy-
improvement step, not a full dynamic-programming solution. I accept that
measured approximation because it removes the exact play oracle's unbounded
runtime tail while keeping materially separated decisions correct in this
bounded audit. I will measure held-out exact EV regret during evaluation rather
than describing these targets as mathematically optimal.

I assign complete shoes to training, validation, or test before generating
rows. The default 100-shoe configuration produces an 80/10/10 shoe split.
No decision from one shoe can cross a split boundary, including exploratory
states. This prevents a model from seeing an earlier prefix of a held-out
shoe during training.

Each generated directory contains:

```text
manifest.json
.checkpoints/
shards/
train.jsonl
validation.jsonl
test.jsonl
```

The manifest records the schema version, fixed casino rules, bet-vocabulary
mapping inputs, split and exploration configuration, all random seeds, each
shoe's split, its cut-card position, and its complete top-to-bottom replay.
Fractions are serialized as integer numerator/denominator pairs. Rational play
and insurance values remain exact. Empirical bet probabilities are exact
rollout-count fractions whose common denominator is at most the configured
rollout count.

Labels can take long enough that interruption is part of the expected
workflow. I atomically checkpoint every completed decision before applying the
next behavior action. When I restart the same command, the engine replays the
shoe from its manifest, validates each cached state, reuses the stored labels,
and continues at the first missing decision. A configuration mismatch is
rejected instead of mixing incompatible rows.

Each shoe has its own exploration seed derived from the recorded exploration
seed, shoe ID, and shoe seed. Shoes can therefore be generated in any order or
by separate workers without changing their rows. For example, four workers can
run these commands against the same output directory:

```bash
uv run python -m blackjack.dataset data/generated/v4 --shard-count 4 --shard-index 0
uv run python -m blackjack.dataset data/generated/v4 --shard-count 4 --shard-index 1
uv run python -m blackjack.dataset data/generated/v4 --shard-count 4 --shard-index 2
uv run python -m blackjack.dataset data/generated/v4 --shard-count 4 --shard-index 3
```

Each worker owns complete shoes selected by `shoe_id % shard_count`, so
parallel execution cannot split one shoe across workers. Completed shoes are
written atomically under `shards/`. When every shoe shard exists, the runner
assembles the final train, validation, and test JSONL files in stable shoe
order. The earlier `--bet-workers` option remains accepted for command
compatibility, but the native fixed-policy simulator does not need nested
workers. Parallelism belongs at the complete-shoe shard level, where workers
share no mutable engine state.

A shared SQLite database stores only completed decision labels and short-lived
ownership claims. WAL mode lets complete-shoe workers reuse identical states
without putting SQL calls inside the recursive hot path. This is particularly
useful for the initial fresh-shoe bet state, which is identical for every
shoe. I do not use Redis: a local networked cache would add serialization and
coordination overhead to fine-grained states, while SQLite already supplies
the needed durable cross-process reuse.

I can time a bounded production run before launching the complete dataset:

```bash
uv run python -m blackjack.dataset data/generated/v4 --shoe-count 100 --benchmark
```

`--benchmark` computes at most two new labels, reports their individual times
and observed mean, saves them as ordinary checkpoints, and stops.
`--benchmark N` changes that bound. Re-running the command continues from those
checkpoints. After the first whole shoe completes, progress output also reports
a measured ETA for the remaining shoes in that worker's shard.

On July 25, 2026, the rational CDP implementation and the first pure-Python
float64 CDZ- implementation both exceeded the three-minute cutoff on the first
fresh-shoe bet label. A later native exhaustive CDZ-/no-resplit solver reduced
that to 32.0 seconds with four workers, but a 100-shoe dataset still remained
impractical.

The fixed-policy native simulator now completes 1,000,000 fresh-shoe rollouts
and the complete bet label in approximately 0.1 seconds on the development
machine after its one-time compile. The recorded smoke-test estimate was
-0.4301% with a 0.1141-percentage-point standard error. That is roughly a 320×
reduction from the previous production label time.

After adding the real first-action evaluator, a clean schema-v4 CLI smoke run
computed five consecutive production decisions—three bets and two play
labels—using 1,000,000 rollouts per bet or legal play action in 0.5 seconds
total. Individual play labels took 0.1 and 0.1 seconds in that sample. A
resumed run validated and reused all five checkpoints, then computed five more
labels in 0.8 seconds; its most expensive play label took 0.3 seconds. These
are bounded measurements on the development machine, not promises for every
deep split state, but they remove the multi-minute exact-oracle tail from the
production generator.

#### 100-Shoe Integration Dataset QA

I generated the complete schema-v4 100-shoe dataset with four complete-shoe
workers. The slowest worker finished in 11 minutes 13 seconds. The assembled
dataset contains 10,206 decisions: 8,157 training, 1,010 validation, and 1,039
test rows. Its three JSONL files occupy approximately 30 MB.

I can reproduce the structural and coverage audit with:

```bash
uv run python -m blackjack.dataset.quality data/generated/v4
```

The audit parses every row through the typed serializer and rejects duplicate
or non-contiguous decision indices, mixed dataset IDs or schema versions,
shoe-level split leakage, unexpected evaluation methods, and missing Monte
Carlo metadata. This run found no integrity errors. All 100 complete shoes are
represented in exactly one split. Exploration changed 20.5% of play behaviors
and 20.9% of insurance behaviors, matching the configured 20% probability
within ordinary sampling variation.

The longest model input has 245 tokens, so the complete integration dataset
fits a 256-token context window. Median context lengths are 111 tokens for bet
rows, 120 for insurance, and 119 for play. Of 5,470 play decisions, 192 have
less than 0.01 wager units between the best and second-best empirical actions;
only 21 have less than 0.001. I retain these close states and their action
values so evaluation can distinguish harmless token disagreements from costly
ones.

The raw target coverage is intentionally the natural generated distribution:

| Target | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `<BET_MIN>` | 3,125 | 345 | 413 |
| `<BET_LOW>` | 260 | 53 | 39 |
| `<BET_MEDIUM>` | 82 | 34 | 1 |
| `<BET_HIGH>` | 40 | 8 | 1 |
| `<HIT>` | 1,738 | 204 | 207 |
| `<STAND>` | 2,042 | 255 | 270 |
| `<DOUBLE>` | 359 | 56 | 45 |
| `<SPLIT>` | 103 | 6 | 8 |
| `<SURRENDER>` | 147 | 14 | 16 |
| `<INSURANCE>` | 23 | 9 | 3 |
| `<NO_INSURANCE>` | 238 | 26 | 36 |

This makes 100 shoes an integration and overfitting dataset, not a credible
final training corpus. In particular, one held-out example cannot measure a
bet class. Based on these observed rates, 5,000 shoes should put the scarcest
training targets around one to two thousand examples; 10,000 shoes should put
them in the low thousands and give substantially better held-out coverage. I
will choose between those sizes using learning curves from nested 100-, 300-,
and 1,000-shoe subsets rather than generating the largest corpus blindly.

I do not rewrite the raw distribution to make classes equal. For the first
training experiment I will compare natural row sampling with capped,
square-root class balancing. The balanced sampler will revisit rare valid
training rows more often, with weight proportional to the inverse square root
of class frequency and a maximum amplification cap. Validation and test remain
untouched so overall accuracy, EV regret, and bankroll results reflect real
shoe frequency. I will also report per-target metrics because natural aggregate
accuracy can hide complete failure on split, surrender, insurance, or a rare
bet class. If a class has too few distinct rows, more shoes or valid targeted
state generation—not unlimited resampling—must provide the missing diversity.

SQLite reuse, atomic checkpoints, and deterministic shoe shards still make
interruption safe. Generated data and locally compiled native artifacts stay
out of Git because the source, configuration, and manifest make them
reproducible. Dataset schema v4 intentionally rejects earlier output
directories; empirical per-action distributions and uncertainty metadata must
not be mixed with older exact-play labels.

#### Complete-Shoe Concurrency Benchmark

I timed the same 16 deterministic shoes at the production one-million-rollout
settings with 1, 2, 4, and 8 complete-shoe processes. Every case started with
a fresh SQLite label cache, while the already-compiled native rollout kernel
was reused. Each case processed 1,621 decisions: 1,605 unique oracle states and
16 within-run cache hits.

```bash
uv run python -m blackjack.dataset.benchmark \
  data/generated/concurrency-benchmark \
  --shoe-count 16 \
  --workers 1,2,4,8
```

The benchmark ran on an ARM Mac with 16 logical CPUs:

| Workers | Wall time | Decisions/second | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 441.4 s | 3.7 | 1.00× |
| 2 | 243.8 s | 6.6 | 1.81× |
| 4 | 125.3 s | 12.9 | 3.52× |
| 8 | 72.4 s | 22.4 | 6.10× |

Eight workers are the measured choice for the 1,000-shoe run. A linear
projection from this bounded sample is about 75 minutes, before credit for
states already present in the 100-shoe cache. This is an operational
benchmark, not a general scaling claim: 16 shoes still expose some
shoe-to-shoe imbalance, and I briefly ran lightweight type and unit checks
during the one- and two-worker cases. The four- and eight-worker cases were
left isolated. The gap is large enough to select eight workers, but I will use
the completed 1,000-shoe run—not this projection—as the durable throughput
measurement.

The coordinated generator preserves the same per-decision checkpoints and
complete-shoe shards as the single-process command:

```bash
uv run python -m blackjack.dataset.parallel data/generated/v5 \
  --shoe-count 1000 \
  --workers 8 \
  --label-cache data/generated/v4/oracle-labels.sqlite3
```

Re-running the exact command resumes completed work. The shared existing cache
can reuse canonical oracle states, but the v5 dataset still receives its own
manifest, checkpoints, shards, and final split files.

#### 1,000-Shoe Corpus

The measured eight-worker run completed all 1,000 shoes in 3,265.4 seconds
(54 minutes 25 seconds), producing 102,484 decisions. This is about 31.4
decisions per second end to end, including shoe simulation, one-million-rollout
labels, checkpoint writes, and final split assembly. No decision checkpoints
were reused during this run, so this is a clean scaled-generation measurement.

I validated the assembled corpus before training. It contains 81,973 training,
10,235 validation, and 10,276 test decisions, with complete-shoe isolation
between splits. Contexts remain within the 256-token model window: the longest
bet, insurance, and play contexts are 242, 243, and 248 tokens. All 1,000 shoes
contribute minimum-bet, hit, and stand examples; the rarer labels now include
4,633 doubles, 1,801 surrenders, 1,202 splits, 524 high bets, and 341 insurance
takes. The high-bet class appears in 111 distinct shoes and insurance in 196,
which is enough to measure learning trends but not yet evidence that the final
corpus is large enough.

The bet-label Monte Carlo standard error has a median of 0.00114 wager units.
For play, 2,003 of 54,832 decisions have a best-versus-second-best margin below
one percentage point, including 225 below one tenth of a percentage point.
Those close decisions are retained with their uncertainty metadata rather than
silently treated as equally certain labels. The learning curves must therefore
be read together with expected-value regret: disagreeing on a near-tie is not
equivalent to missing a large-margin action.

### Training

#### Training Input Boundary

I convert each schema-v4 JSONL row into an immutable training item containing
only token IDs, the target ID, the legal target IDs, the decision kind, and
row provenance. Oracle action values, return distributions, shoe composition,
and Monte Carlo uncertainty never cross this boundary, so the model cannot
accidentally learn from evaluation-only information.

The vocabulary has 29 stable entries: padding, ten visible card values, seven
structural tokens, and eleven decision tokens. Batches are padded only to the
longest sequence in that batch. Loss is evaluated at the final query token,
not across the historical event tokens, and illegal decisions are masked
before both loss and accuracy are calculated.

Training epochs are deterministic from a recorded seed. Natural sampling
visits each row once in a shuffled order. The experimental balanced sampler
draws with replacement using the capped inverse-square-root target weights
described above. Batches are yielded lazily rather than materialized as a
whole epoch so this interface can also handle the scaled corpus. Validation
and test loaders will always use their natural distributions.

Before building the educational transformer, I ran a deliberately disposable
GRU smoke model against 16 real training rows:

```bash
uv run python -m blackjack.training.overfit data/generated/v4 \
  --examples 16 \
  --updates 100
```

It moved from 18.8% to 100% training accuracy and reduced decision loss from
1.2878 to 0.000061. This is evidence that the training plumbing can overfit a
tiny batch; it is not a model baseline, a candidate architecture, or a
concession in the transformer experiment. The from-scratch causal transformer
still belongs in the notebook course.

#### First Working Transformer

I implement the initial decoder-only transformer directly from tensor
operations rather than calling `torch.nn.Transformer` or fused attention. The
model uses learned token and positional embeddings, pre-normalized causal
multi-head self-attention, feed-forward layers, residual connections, and a
vocabulary projection at every position. Tests demonstrate that changing a
future token cannot change an earlier logit and that right-padding cannot
change a real-token logit.

The default baseline has four layers, four 32-dimensional heads, a
128-dimensional embedding, a 512-dimensional feed-forward layer, and 831,488
parameters. This is intentionally small enough for repeated Apple Metal
experiments. Each real token receives a learned position relative to its
decision query, with the query anchored at position 255. This retains the
order and recency of every visible card while placing the current hand and
dealer upcard at stable offsets. An absolute-position ablation demonstrated
why this inductive bias matters: with only 80 training shoes, the same
current-hand pattern otherwise has to be relearned at many history-dependent
positions.

The trainer seeds initialization, dropout, and epoch sampling; clips
gradients; retains the best validation-loss checkpoint; and records overall,
per-kind, and per-target accuracy. Evaluation joins predictions back to the
metadata only by shoe and decision index, allowing expected-profit regret for
play and insurance and expected-log-growth regret for bets without putting
oracle values into model inputs.

I am proving this implementation and the training settings before turning it
into the educational notebook sequence. The notebooks will contain the
working implementation, first-person notes, small tensor examples, and visual
explanations; I can then comment out cells and rebuild each component myself.

#### 100-Shoe Baseline Results

I trained matched 15-epoch query-relative models with natural and capped
inverse-square-root sampling. Both used the same 831,488-parameter model,
initialization seed, optimizer, batch size, and untouched 1,010-decision
validation split. I selected checkpoints by minimum validation loss and did
not inspect the test split. I retain the absolute-position result below as the
ablation that motivated the position change.

| Model/sampler | Best epoch | Validation accuracy | Play accuracy | Mean play regret | Mean half-Kelly fraction error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legal-set frequency | — | 62.0% | 47.7% | 0.1545 wager units | 0.2012 pp |
| H17 basic strategy | — | 86.9% | 94.8% | 0.00110 wager units | 0.2012 pp |
| Absolute/natural | 4 | 64.3% | 48.4% | 0.1580 wager units | 0.1641 pp |
| Query-relative/natural | 12 | 86.6% | 90.1% | 0.0141 wager units | 0.1516 pp |
| Query-relative/balanced | 11 | 85.9% | 87.5% | 0.0295 wager units | 0.1433 pp |

The frequency control never reads the token sequence. It fits natural target
counts on the training split and chooses the most frequent currently legal
token, which reduces here to minimum bet, stand, and decline insurance when
those actions are available. The basic-strategy control reads only the current
hand, dealer upcard, and legal actions; it ignores visible-card history and
uses minimum bet and declined insurance. Its 94.8% agreement with empirical
play labels shows that composition-dependent deviations are a small minority
of natural play states.

The half-Kelly errors above are percentage points of bankroll and include both
the finite token vocabulary's rounding error and the model's prediction error.
During the scaled-run audit I found that the earlier evaluator had instead
compared predictions with the discrete token that maximized expected log
growth. That is a full-Kelly-oriented comparator and did not match the
half-Kelly policy used to create the target. The categorical targets, training
loss, checkpoints, play metrics, and insurance metrics were unaffected. I
replaced that provisional metric and rescored every model reported here from
its retained checkpoint. Evaluation now reports absolute fraction error and
absolute expected-log-growth change relative to the continuous half-Kelly
policy; it does not relabel a larger, riskier bet as an improvement.

```bash
uv run python -m blackjack.training.baseline data/generated/v4
uv run python -m blackjack.training.baseline \
  data/generated/v4 \
  --policy basic-strategy
```

The absolute model kept improving its training accuracy through 30 epochs but
failed to generalize basic strategy. Query-relative positions changed that
result immediately: natural play accuracy reached 90.1%, and mean play regret
fell by more than 11 times. The model now learns the conventional core
strategy, but it does not yet beat that control on play.

Balancing exposes a real tradeoff. Relative to natural sampling, it recovered
more medium bets (16 of 34 versus 3), doubles (48 of 56 versus 42), surrenders
(11 of 14 versus 8), and insurance takes (3 of 9 versus 1). It reduced mean
absolute half-Kelly fraction error, but lost 0.7 percentage points overall and
more than doubled mean play regret. Neither result is reliable for the eight
high-bet validation rows. I will carry both samplers into the nested learning
curves rather than tuning the cap against this small split.

The composition-dependent slice makes that tradeoff especially relevant. Basic
strategy agrees with the oracle on 507 of 535 validation play decisions. On
those common states, the natural and balanced models score 93.3% and 89.0%.
On the 28 states where the oracle departs from basic strategy, natural sampling
gets 9 correct (32.1%) while balancing gets 17 correct (60.7%). Twenty-eight
examples are too few for a conclusion, but this conditional metric is much
closer to the experiment's purpose than aggregate accuracy.

```bash
uv run python -m blackjack.training.compare \
  data/generated/v4 \
  artifacts/training/v4-query-relative-15
```

These results verify a credible learning system without yet establishing
composition-dependent play. The 1,000-shoe corpus is needed both for rare
targets and for the roughly 5% of play states where the empirical oracle
departs from basic strategy.

The v4 integration run is not the formal 100-shoe point on the later learning
curve because split assignment depends on the total corpus size. After v5 is
complete, every learning-curve run will use the same full v5 validation split
and an untouched v5 test split. Training rows will expand by original shoe-ID
prefix: IDs below 100, below 300, and below 1,000. These prefixes contain only
the training-assigned shoes within each range, so they are strictly nested
without allowing a validation shoe to become training data. The training
artifacts record the resulting shoe and decision counts.

#### Natural-Sampling Learning Curve

The formal natural-sampling curve uses 78, 246, and 800 training-assigned
shoes from those three prefixes. Every point uses the same 10,235-decision
validation split, the same initialization and optimizer configuration, and a
checkpoint selected by minimum validation cross-entropy. The test split
remains unopened.

| Training prefix | Training decisions | Best epoch | Overall accuracy | Play accuracy | Basic-strategy agreement | Composition deviations | Mean play regret | Mean half-Kelly fraction error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 8,006 | 11 | 90.3% | 89.6% | 91.8% | 46.1% | 0.01365 wagers | 0.1209 pp |
| 300 | 25,173 | 13 | 94.4% | 93.8% | 95.8% | 55.4% | 0.00922 wagers | 0.1041 pp |
| 1,000 | 81,973 | 10 | 95.8% | 95.7% | 98.0% | 50.6% | 0.00430 wagers | 0.1016 pp |

The larger corpus clearly improves overall accuracy, ordinary play, mean play
regret, and bet sizing. At 1,000 shoes the model slightly exceeds the
history-blind basic-strategy control's 95.1% play accuracy, but its 0.00430
mean play regret is still worse than the control's 0.00152 because a small
number of model mistakes are expensive.

Composition-dependent accuracy is not monotonic at the selected checkpoints.
The 1,000-shoe model reaches 55.8% on that slice at epoch 13, but the
predeclared minimum-loss selector chooses epoch 10 at 50.6%. I retain and
report both facts rather than choosing the checkpoint post hoc.

#### Balanced-Sampling Learning Curve

The balanced runs use the same nested training rows, fixed validation set,
model initialization, optimizer, and minimum-loss selector. Only the
training-row sampler changes to capped inverse-square-root target weighting
with replacement.

| Training prefix | Best epoch | Overall accuracy | Play accuracy | Basic-strategy agreement | Composition deviations | Mean play regret | Mean half-Kelly fraction error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 13 | 86.8% | 83.2% | 85.2% | 42.7% | 0.03107 wagers | 0.1209 pp |
| 300 | 15 | 92.8% | 92.1% | 94.7% | 40.8% | 0.01241 wagers | 0.1088 pp |
| 1,000 | 15 | 95.8% | 95.3% | 97.3% | 56.6% | 0.00318 wagers | 0.1011 pp |

Balancing is harmful when the rare classes contain too little distinct data:
at 100 and 300 shoes it repeatedly exposes a narrow set of rows and gives back
too much common-strategy accuracy. At 1,000 shoes the tradeoff changes. Natural
and balanced sampling are essentially tied overall—9,802 versus 9,804 correct
validation decisions—but balancing gets 151 of 267 composition-dependent
play decisions correct instead of 135 and lowers mean play regret by 26%.

The improvement is not universal. Relative to natural sampling at 1,000
shoes, balancing recovers more low and medium bets, splits, surrenders, and
insurance takes, but fewer high bets and hits. It also has slightly worse
validation cross-entropy (0.1078 versus 0.1032) and 20 fewer correct play
decisions overall. I therefore treat the balanced 1,000-shoe model as the
better composition-sensitive candidate, not as an unqualified winner.

Both curves still improve materially between 300 and 1,000 shoes. This
supports eventually scaling beyond 1,000, but the cheap permutation-
augmentation and Hi-Lo control experiments should come first: they can reveal
whether the next constraint is invariance, model behavior, or genuinely
independent oracle-labeled compositions before I spend several more hours
generating data.

#### Six-Deck H17 Hi-Lo Control

I added a visibility-matched Hi-Lo control before interpreting the transformer
as a card counter. It assigns `+1` to 2–6, `0` to 7–9, and `-1` to ten-valued
cards and Aces, then floors the running count divided by estimated decks
remaining. These are the standard balanced tags and true-count conversion
described by the
[Wizard of Odds Hi-Lo reference](https://wizardofodds.com/games/blackjack/card-counting/high-low/).
For play, the control starts from the same six-deck H17 basic strategy used
elsewhere in this project and applies the H17 Illustrious 18 and Fab 4 indices
from this
[six-deck H17 deviation chart](https://www.blackjacktrainer.fyi/charts/deviations).
It takes insurance at true count `+3` or higher.

Hi-Lo does not define one universal bet ramp. I inspected only the v5 training
split, fixed the thresholds before evaluating validation, and mapped the four
existing bankroll tokens as follows:

| Floored true count | Bet token |
| ---: | --- |
| Below `+2` | `<BET_MIN>` |
| `+2` to `+3` | `<BET_LOW>` |
| `+4` to `+5` | `<BET_MEDIUM>` |
| `+6` or higher | `<BET_HIGH>` |

This is a token-matched control rather than a claim about an ideal casino bet
spread. It receives exactly the model sequence and legal-action mask. In
particular, it estimates decks remaining as `(312 - visible token count) / 52`;
it does not receive a separate count of burned or face-down unavailable cards
because those values are absent from the model input too.

On the untouched 10,235-decision validation split:

| Control | Overall accuracy | Bet accuracy | Insurance accuracy | Play accuracy | Composition deviations | Mean play regret | Mean half-Kelly fraction error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H17 Hi-Lo | 94.8% | 93.9% | 90.1% | 95.8% | 34.8% | 0.00146 wagers | 0.1083 pp |
| Natural transformer | 95.8% | 96.0% | 94.9% | 95.7% | 50.6% | 0.00430 wagers | 0.1016 pp |
| Balanced transformer | 95.8% | 96.1% | 94.3% | 95.3% | 56.6% | 0.00318 wagers | 0.1011 pp |

The transformers recover more of the oracle's composition-dependent targets,
but Hi-Lo still has lower value-weighted play regret. This is an important
distinction: the learned models discover more deviations while also making
some costly errors on ordinary states that the mature counting system avoids.
Aggregate token accuracy alone would hide that result.

```bash
uv run python -m blackjack.training.baseline \
  data/generated/v5 \
  --training-shoe-prefix 1000 \
  --policy hi-lo
```

#### Card-Order Permutation Experiment

For blackjack composition, the multiset of previously exposed cards matters,
not the order in which those historical cards appeared. Card order within the
current hand is similarly irrelevant. I therefore added deterministic,
training-only augmentation that independently shuffles the `<HISTORY>` cards
and current `<PLAYER>` cards while leaving the dealer upcard, structure,
query, legal actions, target, and provenance unchanged.

The augmentation is dynamic rather than copied into the JSONL corpus. Each
original training row is still sampled once per natural epoch, but a seed
derived from the training seed, epoch, shoe ID, decision index, and sample
position produces a different reproducible Fisher–Yates permutation. This
provides up to fifteen presentations of the same labeled state without another
oracle call, without increasing the optimizer-update budget, and without
pretending those variants are independent labels. Validation and test remain
in their original chronological order for every primary metric.

I trained one matched 1,000-shoe natural-sampling model with this augmentation.
It used the same 81,973 training decisions, initialization, optimizer, batch
size, 15 epochs, validation set, and minimum-validation-loss checkpoint rule
as the unaugmented run.

| 1,000-shoe model | Best epoch | Overall accuracy | Play accuracy | Composition deviations | Mean play regret | Mean half-Kelly fraction error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural, chronological inputs | 10 | 95.769% | 95.664% | 50.6% | 0.00430 wagers | 0.1016 pp |
| Natural, permuted training inputs | 13 | 95.779% | 95.591% | 55.1% | 0.00261 wagers | 0.1012 pp |

Headline accuracy is tied: the augmented model gets one more validation
decision correct overall and four fewer play decisions correct. The useful
difference is value-sensitive. It gets 12 more of the 267
composition-dependent play states correct and lowers mean play regret by 39%.
Insurance regret rises from 0.00061 to 0.00087 wagers, so this is a measured
tradeoff rather than an unqualified win.

As a secondary audit, I applied four deterministic valid permutations to each
validation row without using those variants for checkpoint selection. Of
40,249 comparisons where a permutation actually changed the input, the
unaugmented model preserved its prediction 97.41% of the time and the augmented
model did so 97.91% of the time. The unaugmented model's accuracy fell from
95.77% to 95.61% on permuted inputs; the augmented model moved from 95.78% to
95.80%. This supports the intended invariance, although one training seed is
not enough to treat the exact improvement as a general result.

The evidence now justifies a bounded 5,000-shoe corpus rather than jumping
directly to 10,000. Both natural and balanced learning curves improve strongly
from 300 to 1,000 shoes; cheap valid augmentation reduces regret but does not
remove the need for independent compositions; and the learned model still has
higher play regret than Hi-Lo. Five thousand shoes should test whether that gap
is label-limited while providing roughly five times the rare-target coverage.
I will keep the test split sealed and run corpus QA before choosing the final
training schedule.

```bash
uv run python -m blackjack.training.run \
  data/generated/v5 \
  artifacts/training/v5-natural-1000-permuted \
  --training-shoe-prefix 1000 \
  --sampling natural \
  --card-order-augmentation permute \
  --device mps

uv run python -m blackjack.training.invariance \
  data/generated/v5 \
  artifacts/training/v5-natural-1000 \
  artifacts/training/v5-natural-1000-permuted \
  --permutations 4 \
  --device mps
```

The first formal point can be reproduced with:

```bash
uv run python -m blackjack.training.run \
  data/generated/v5 \
  artifacts/training/v5-natural-100 \
  --training-shoe-prefix 100 \
  --sampling natural \
  --device mps
```

On Apple Metal, steady-state epochs took approximately eight seconds for the
small v4 integration split, 75–92 seconds for the formal 100/300-shoe points,
and 123–143 seconds for the 1,000-shoe points. Repeating the seeded v4 natural
run reproduced every rounded metric, while corresponding weights differed by
at most `4.77e-7`. CPU tests are bit-exact; Metal training is seeded and
numerically reproducible but is not described as bit-for-bit deterministic.
Each run atomically stores the best validation-loss weights, every epoch's
weights, the complete configuration, the vocabulary, and every epoch's
aggregate, per-kind, per-target, objective-regret, and basic-strategy
agreement/deviation metrics. Retaining each epoch avoids assuming that minimum
cross-entropy, minimum bankroll regret, and maximum composition-deviation
accuracy select the same checkpoint.

### Evaluation

### Experiments and Interpretability

---

## Local Development

Install the project and every development/notebook dependency:

```bash
uv sync --all-groups
```

The production bet and play oracles also need a local C++20 compiler. On
macOS, the Command Line Tools provide `c++`; the first empirical label compiles
the small native kernel automatically and caches the ignored library beside
its source.

In VS Code, select `.venv/bin/python` as the Python interpreter. Then run:

```bash
uv run pytest
uv run pyright
uv run ruff check .
```

Open `notebooks/01_blackjack_engine.ipynb` for the visual engine walkthrough or
`notebooks/02_bet_token_pilot.ipynb` for the reproducible bet-vocabulary
analysis. Select the same `.venv/bin/python` environment for either notebook.

To benchmark the production generator with resumable checkpoints, choose an
output directory under the ignored `data/generated/` tree:

```bash
uv run python -m blackjack.dataset data/generated/v4 --shoe-count 100 --benchmark
```

Remove `--benchmark` when the measured cost and worker count are acceptable.

---

## TODOs

### 1. Define the Experiment

- [x] Establish the motivation and central research question.
- [x] Fix one representative casino ruleset.
- [x] Define the observable information available to the model.
- [x] Define constrained, single-token model outputs.
- [x] Define optimal play as the legal action with the highest expected value.
- [x] Define half Kelly as the bet-sizing policy.

### 2. Build the Blackjack Engine

- [x] Represent cards, hands, a six-deck shoe, and visible card history.
- [x] Implement deterministic shuffling and seeded replay.
- [x] Implement hand totals, soft Aces, naturals, and busts.
- [x] Implement the complete round state machine.
  - [x] Initial deal and hidden dealer hole card
  - [x] Dealer peek
  - [x] Insurance
  - [x] Hit and stand
  - [x] Double down
  - [x] Splitting and split-hand limits
  - [x] Split-Ace restrictions
  - [x] Late surrender
  - [x] Dealer play
  - [x] Settlement and payouts
- [x] Produce a complete internal event log for debugging and visualization.
- [x] Produce the minimal model context from the internal game state.
- [x] Test every rule and payout boundary.

### 3. Build the Blackjack Oracle

- [x] Calculate exact dealer outcome probabilities from a remaining shoe.
- [x] Calculate the expected value of every legal player action.
- [x] Average correctly over an unknown dealer hole card.
- [x] Condition probabilities on a negative dealer peek.
- [x] Calculate split-hand expected values.
- [x] Calculate the expected value of insurance.
- [x] Select the optimal composition-dependent action.
- [x] Calculate the complete probability distribution of round returns.
- [x] Calculate continuous full-Kelly and half-Kelly bankroll fractions.
- [x] Validate oracle results against independently published values and small
      brute-force cases.

The six-deck validation fixtures use the independently published
[Wizard of Odds H17 composition-dependent return table](https://wizardofodds.com/games/blackjack/appendix/9/6dh17r4/).

### 4. Run the Bet-Token Pilot Analysis

- [x] Sample representative pre-deal shoe compositions.
- [x] Plot the distribution of continuous half-Kelly fractions.
- [x] Compare candidate minimum bets, maximum bets, and token spacing.
- [x] Measure rounding error and expected log-growth regret for each candidate
      vocabulary.
- [x] Measure the number of examples that would belong to each bet class.
- [x] Select the smallest bet vocabulary that preserves meaningful precision.
- [x] Document the selected bet tokens and their bankroll fractions.

### 5. Build the Dataset Pipeline

- [x] Generate reproducible complete shoes with the blackjack engine.
- [x] Use the oracle to label every bet, insurance, and play decision.
- [x] Include enough state exploration to cover decisions beyond a single
      optimal trajectory.
- [x] Convert each decision into the minimal model input and one target token.
- [x] Retain action values, return distributions, and shoe composition as
      evaluation-only metadata.
- [x] Split training, validation, and test data by complete shoe.
- [x] Prevent decisions from the same shoe from crossing dataset splits.
- [x] Record generation configuration and random seeds with every dataset.
- [x] Add atomic decision checkpoints, exact replay validation, progress
      reporting, and independent complete-shoe worker shards.
- [x] Add a shared SQLite label cache for completed decision labels.
- [x] Add deterministic native fixed-policy Monte Carlo production bet labels,
      including unavailable-card marginalization and exact insurance.
- [x] Add deterministic native first-action Monte Carlo production play labels
      with complete correlated round settlement.
- [x] Compare real fixed-continuation play rollouts against an exact rational
      corpus and retain the exact oracle as an evaluation verifier.
- [x] Record rollout replay inputs, sampling uncertainty, and the playing-policy
      concession in every empirical row and in the methodology.
- [x] Make a one-million-rollout production bet benchmark complete in a
      measured 0.1 seconds after compilation.
- [x] Generate and assemble a 100-shoe schema-v4 integration dataset with four
      complete-shoe workers.
- [x] Add reproducible integrity, coverage, context-length, uncertainty, and
      close-action QA for assembled datasets.
- [x] Benchmark 1, 2, 4, and 8 uncached complete-shoe workers before the
      scaled generation run.
- [x] Generate and QA a 1,000-shoe corpus for 100/300/1,000-shoe learning
      curves.
- [x] Use the 100/300/1,000-shoe curves, Hi-Lo control, and permutation
      experiment to select 5,000 shoes as the next bounded scale point.
- [ ] Generate and QA the 5,000-shoe corpus before choosing the final training
      schedule; scale to 10,000 only if that result remains data-limited.

### 6. Build the Notebook Course and Transformer

- [ ] Design the ordered notebook curriculum.
- [x] Build the blackjack vocabulary and token encoder.
- [ ] Visualize token sequences, embeddings, and positional information.
- [x] Implement causal self-attention from scratch.
- [x] Implement multi-head attention, feed-forward layers, and transformer
      blocks.
- [x] Implement the causal language-model head.
- [x] Implement legal-token masking and typed decision decoding.
- [ ] Add shape checks, small examples, and visual explanations throughout.

### 7. Train the Model

- [x] Build typed decision-only datasets, lazy batches, legal masks, and
      deterministic natural and capped inverse-square-root samplers.
- [x] Verify the full pipeline by intentionally overfitting a tiny dataset.
- [x] Train the first natural-sampling baseline on the 100-shoe integration
      dataset.
- [x] Compare natural sampling with capped inverse-square-root target
      balancing while leaving validation and test untouched.
- [x] Plot nested 100-, 300-, and 1,000-shoe learning curves for accuracy and
      expected-value regret before choosing the final corpus size.
- [x] Train only on decision-token targets.
- [x] Establish seeded training and atomic best-model checkpointing, with exact
      CPU replay and the measured Metal numerical tolerance documented.
- [x] Select and validate a query-relative model that trains comfortably with
      Apple Silicon acceleration.
- [x] Record losses and decision-specific metrics.
- [x] Compare the unaugmented baseline with deterministic training-only
      permutations of exposed-card history and current-hand card order at the
      same optimizer-update budget; keep validation/test chronological and
      measure prediction consistency across equivalent permutations.
- [ ] If the supervised learning curve remains label-limited, pretrain the
      custom transformer on cheap unlabeled visible blackjack sequences and
      measure how much oracle-labeled data fine-tuning then requires.
- [ ] After preserving the supervised-from-scratch baseline, evaluate
      value-aware post-training that uses normalized oracle action values and
      Monte Carlo uncertainty to distinguish costly errors from noisy
      near-ties.
- [ ] Compare every additional training stage against the same untouched
      validation and test protocol rather than replacing the baseline.

### 8. Evaluate the Model

- [x] Measure exact decision accuracy by decision type.
- [x] Measure expected-value regret for playing mistakes.
- [x] Replace the provisional full-Kelly-oriented bet regret metric with
      half-Kelly-aligned bet-fraction error and absolute log-growth change;
      recompute every reported bet metric from retained checkpoints.
- [ ] Evaluate complete bankroll trajectories on held-out shoes.
- [x] Compare against no-history and basic-strategy controls plus a
      predeclared six-deck H17 Hi-Lo system with public running/true count, a
      tokenized bet ramp, insurance threshold, and documented play indices.
- [ ] Break results down by shoe penetration and remaining composition.
- [ ] Verify that evaluation shoes and states were not seen during training.

### 9. Inspect What the Model Learned

- [ ] Probe hidden states for running count, true count, and deck composition.
- [ ] Visualize attention across previously exposed cards.
- [ ] Remove or shuffle history and measure the effect on predictions.
- [ ] Make controlled card substitutions and observe decision changes.
- [ ] Compare models with different context lengths and capacities.
- [ ] Identify failure cases and design follow-up experiments.

## What I Want to Try Next

I want to finish this from-scratch experiment before replacing its clean,
controlled learning problem with pretrained knowledge. Then I want to repeat
the complete process on a second domain whose labels are cheap enough to
generate locally. A good candidate will have a small vocabulary, short
sequences, millions of reproducible synthetic examples, labels that can be
computed exactly or in milliseconds, a strong non-neural baseline, and a
metric tied directly to the real objective. Small board-game tablebases,
synthetic arithmetic or program execution, compact navigation policies, and
solver-backed scheduling problems are possible directions.

After I can design that second experiment alone, I want to fine-tune a small
existing model with parameter-efficient adapters on a language-oriented task.
That project should teach me the complementary problems that this one
deliberately avoids: working with an inherited tokenizer and pretrained
knowledge, formatting instruction data, measuring behavior drift and
catastrophic forgetting, and serving a model much larger than its trainable
adapter.

Between those projects, I can use this transformer as a bridge. I can pretrain
it from scratch on millions of cheap, unlabeled visible blackjack histories,
then fine-tune it on the expensive oracle decisions. That experiment would
measure label efficiency while preserving the custom vocabulary and a clear
account of exactly what prior experience the model received.
