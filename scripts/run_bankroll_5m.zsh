#!/bin/zsh

set -euo pipefail

readonly artifact_directory="models/blackjack-v6-full"
readonly output_directory="artifacts/evaluation/v6-bankroll-5m"
readonly simulation_seed=20260801
readonly first_shoe=25000
readonly shard_count=40
readonly shoes_per_shard=2275
readonly worker_count=8

mkdir -p "${output_directory}"

for ((wave = 0; wave < shard_count / worker_count; wave += 1)); do
  typeset -a worker_pids=()
  print "[scale] starting wave $((wave + 1))/$((shard_count / worker_count))"

  for ((lane = 0; lane < worker_count; lane += 1)); do
    shard_index=$((wave * worker_count + lane))
    shoe_start=$((first_shoe + shard_index * shoes_per_shard))
    output_path="${output_directory}/shard-${shard_index}.json"

    if [[ -s "${output_path}" ]]; then
      print "[scale] shard ${shard_index} already complete; skipping"
      continue
    fi

    print \
      "[scale] shard ${shard_index}: shoes ${shoe_start}-$((shoe_start + shoes_per_shard - 1))"
    uv run python -m blackjack.training.bankroll \
      "${artifact_directory}" \
      "${output_path}" \
      --shoe-count "${shoes_per_shard}" \
      --shoe-start "${shoe_start}" \
      --simulation-seed "${simulation_seed}" \
      --inference-batch-size 128 \
      --progress-every-shoes 500 \
      --device mps &
    worker_pids+=($!)
  done

  for worker_pid in "${worker_pids[@]}"; do
    wait "${worker_pid}"
  done
  print "[scale] wave $((wave + 1)) complete"
done

print "[scale] all ${shard_count} shards complete"
