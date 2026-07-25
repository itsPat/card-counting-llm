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

- Bet tokens: a discrete set of bankroll fractions to be selected through a
  pilot analysis
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

The training target will always be the legal action with the highest expected
value in that exact state. I will not use a basic-strategy chart, a Hi-Lo
decision table, or the action produced by another learned model as ground
truth. Those approaches compress or approximate the underlying probabilities;
this experiment is intended to teach the transformer the optimal
composition-dependent decision directly.

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

The calculation will use the complete distribution of possible round returns,
not the shortcut of dividing estimated advantage by an assumed variance. The
distribution will incorporate natural blackjack, ordinary wins and losses,
pushes, surrender, insurance, doubles, splits, and the optimal decisions made
later in the hand.

The model will predict a discrete bankroll fraction rather than a dollar
amount. The blackjack engine will convert that fraction into a wager using the
current bankroll, so bankroll does not need to appear in the model input.

I will select the discrete bet-token scale only after a pilot analysis of the
half-Kelly fractions produced by representative shoe compositions. This avoids
choosing the minimum, maximum, or spacing of the betting vocabulary by
intuition.

### Training

### Evaluation

### Experiments and Interpretability

---

## Local Development

Install the project and every development/notebook dependency:

```bash
uv sync --all-groups
```

In VS Code, select `.venv/bin/python` as the Python interpreter. Then run:

```bash
uv run pytest
uv run pyright
uv run ruff check .
```

Open `notebooks/01_blackjack_engine.ipynb` in VS Code and select the same
`.venv/bin/python` environment to run the visual engine walkthrough.

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

- [ ] Calculate exact dealer outcome probabilities from a remaining shoe.
- [ ] Calculate the expected value of every legal player action.
- [ ] Average correctly over an unknown dealer hole card.
- [ ] Condition probabilities on a negative dealer peek.
- [ ] Calculate split-hand expected values.
- [ ] Calculate the expected value of insurance.
- [ ] Select the optimal composition-dependent action.
- [ ] Calculate the complete probability distribution of round returns.
- [ ] Calculate continuous full-Kelly and half-Kelly bankroll fractions.
- [ ] Validate oracle results against independently published values and small
      brute-force cases.

### 4. Run the Bet-Token Pilot Analysis

- [ ] Sample representative pre-deal shoe compositions.
- [ ] Plot the distribution of continuous half-Kelly fractions.
- [ ] Compare candidate minimum bets, maximum bets, and token spacing.
- [ ] Measure rounding error and expected log-growth regret for each candidate
      vocabulary.
- [ ] Measure the number of examples that would belong to each bet class.
- [ ] Select the smallest bet vocabulary that preserves meaningful precision.
- [ ] Document the selected bet tokens and their bankroll fractions.

### 5. Build the Dataset Pipeline

- [ ] Generate reproducible complete shoes with the blackjack engine.
- [ ] Use the oracle to label every bet, insurance, and play decision.
- [ ] Include enough state exploration to cover decisions beyond a single
      optimal trajectory.
- [ ] Convert each decision into the minimal model input and one target token.
- [ ] Retain action values, return distributions, and shoe composition as
      evaluation-only metadata.
- [ ] Split training, validation, and test data by complete shoe.
- [ ] Prevent decisions from the same shoe from crossing dataset splits.
- [ ] Record generation configuration and random seeds with every dataset.

### 6. Build the Notebook Course and Transformer

- [ ] Design the ordered notebook curriculum.
- [ ] Build the blackjack vocabulary and token encoder.
- [ ] Visualize token sequences, embeddings, and positional information.
- [ ] Implement causal self-attention from scratch.
- [ ] Implement multi-head attention, feed-forward layers, and transformer
      blocks.
- [ ] Implement the causal language-model head.
- [ ] Implement legal-token masking and typed decision decoding.
- [ ] Add shape checks, small examples, and visual explanations throughout.

### 7. Train the Model

- [ ] Verify the full pipeline by intentionally overfitting a tiny dataset.
- [ ] Train only on decision-token targets.
- [ ] Balance bet, insurance, and play decisions during training.
- [ ] Establish deterministic training and checkpointing.
- [ ] Tune a model that trains comfortably with Apple Silicon acceleration.
- [ ] Record losses and decision-specific metrics.

### 8. Evaluate the Model

- [ ] Measure exact decision accuracy by decision type.
- [ ] Measure expected-value regret for playing mistakes.
- [ ] Measure bet-fraction error and expected log-growth regret.
- [ ] Evaluate complete bankroll trajectories on held-out shoes.
- [ ] Compare against no-history, basic-strategy, and conventional counting
      baselines.
- [ ] Break results down by shoe penetration and remaining composition.
- [ ] Verify that evaluation shoes and states were not seen during training.

### 9. Inspect What the Model Learned

- [ ] Probe hidden states for running count, true count, and deck composition.
- [ ] Visualize attention across previously exposed cards.
- [ ] Remove or shuffle history and measure the effect on predictions.
- [ ] Make controlled card substitutions and observe decision changes.
- [ ] Compare models with different context lengths and capacities.
- [ ] Identify failure cases and design follow-up experiments.
