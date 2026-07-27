# Blackjack v6 Dataset and Models

## Scope

This artifact set preserves the paired 1,000/5,000-shoe experiment described
in the project README.

| Field | Value |
| --- | --- |
| Dataset ID | `blackjack-decisions-v4-5df03f72d05a47d0` |
| Schema | 4 |
| Generated shoes | 5,000 |
| Training | 4,000 shoes / 409,207 decisions |
| Validation | 500 shoes / 50,925 decisions |
| Sealed test | 500 shoes / 51,013 decisions |
| Maximum context | 250 tokens |

The public release contains training and validation data only. I retain the
complete test JSONL and its full shoe manifest locally until the final
evaluation protocol is frozen. The committed test checksum below establishes
the file's identity without publishing its labels or replay shoes.

## Methodology

Every row comes from the fixed six-deck H17 rules in the main README. Bet
targets use deterministic one-million-rollout Monte Carlo estimates of the
complete round-return distribution under a documented fixed H17
basic-strategy continuation policy. Play targets use deterministic
one-million-rollout first-action evaluation with the same fixed continuation
concession. Insurance is evaluated exactly. Twenty percent deterministic
exploration exposes states beyond one greedy trajectory.

These fixed-policy continuations are a deliberate compute concession, not an
exact rational rollout of every future decision. Rows retain the per-action
values, replay inputs, uncertainty estimates, and generation configuration
needed to audit that choice.

Complete generation and quality methodology lives in the
[project README](../../README.md).

## Committed Models

The tag `blackjack-v6` contains the selected weights and complete per-epoch
metrics for both leakage-safe paired runs:

| Model | Training shoes | Best epoch | Validation accuracy | Mean play regret |
| --- | ---: | ---: | ---: | ---: |
| [`blackjack-v6-prefix`](../../models/blackjack-v6-prefix) | 779 | 15 | 96.15% | 0.001914 wagers |
| [`blackjack-v6-full`](../../models/blackjack-v6-full) | 4,000 | 15 | 97.56% | 0.000473 wagers |

Each directory contains only the selected `model.pt` state dictionary and its
`metrics.json`. Intermediate epoch checkpoints remain reproducible local
training artifacts and are not committed.

## GitHub Release

The [`blackjack-v6` release](https://github.com/itsPat/card-counting-llm/releases/tag/blackjack-v6)
contains:

| Asset | Compressed bytes | Uncompressed bytes |
| --- | ---: | ---: |
| `blackjack-v6-train.jsonl.gz` | 90,163,297 | 1,283,652,579 |
| `blackjack-v6-validation.jsonl.gz` | 11,214,361 | 159,860,591 |
| `blackjack-v6-train-validation-manifest.json.gz` | 1,247,158 | 19,042,663 |

The released manifest preserves the original dataset ID and generation
configuration but contains only train and validation shoe replays. It is named
separately so it cannot be mistaken for the complete sealed manifest.

I can download and unpack the public data with:

```bash
mkdir -p data/generated/v6
curl -L \
  https://github.com/itsPat/card-counting-llm/releases/download/blackjack-v6/blackjack-v6-train.jsonl.gz \
  -o data/generated/v6/train.jsonl.gz
curl -L \
  https://github.com/itsPat/card-counting-llm/releases/download/blackjack-v6/blackjack-v6-validation.jsonl.gz \
  -o data/generated/v6/validation.jsonl.gz
curl -L \
  https://github.com/itsPat/card-counting-llm/releases/download/blackjack-v6/blackjack-v6-train-validation-manifest.json.gz \
  -o data/generated/v6/release-manifest.json.gz
gzip -dk data/generated/v6/train.jsonl.gz
gzip -dk data/generated/v6/validation.jsonl.gz
gzip -dk data/generated/v6/release-manifest.json.gz
```

The released train and validation files are sufficient to load the committed
models and reproduce every reported validation metric:

```bash
uv run python -m blackjack.training.evaluate \
  data/generated/v6 \
  models/blackjack-v6-full \
  --device mps
```

## Integrity

From the repository root, I can verify committed weights and metrics with:

```bash
shasum -a 256 -c datasets/blackjack-v6/repository-models.sha256
```

After downloading the three release assets into one directory, I can verify
them with:

```bash
shasum -a 256 -c release-assets.sha256
```

The release includes `release-assets.sha256` as a fourth asset. The repository
also records the uncompressed train/validation hashes and the sealed local
test/manifest commitments in [`source-data.sha256`](source-data.sha256).
