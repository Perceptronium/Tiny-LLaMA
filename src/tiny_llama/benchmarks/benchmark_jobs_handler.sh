#!/usr/bin/env bash
set -euo pipefail

# Not enough memory on an A30 to run larger models
model_configs=("small" "medium" "large") # "xl" "2.7B")

d_models=(768 1024 1280) # 1600 2560)

d_ffs=(3072 4096 5120) # 6400 10240)

num_layerss=(12 24 36 48 32)

num_headss=(12 16 20 25 32)

vocab_size=10000
batch_size=4
context_length=256
learning_rate=0.001
rope_theta=10000

warmup_steps=5
time_execution_steps=10


for i in "${!model_configs[@]}"; do
    echo "${model_configs[$i]}"
    python benchmarks/end_to_end_benchmarking.py \
    --model_config "${model_configs[$i]}" \
    --d_model "${d_models[$i]}" \
    --d_ff "${d_ffs[$i]}" \
    --num_layers "${num_layerss[$i]}" \
    --num_heads "${num_headss[$i]}" \
    --rope_theta "$rope_theta" \
    --learning_rate "$learning_rate" \
    --vocab_size "$vocab_size" \
    --batch_size "$batch_size" \
    --context_length "$context_length" \
    --warmup_steps "$warmup_steps" \
    --time_execution_steps "$time_execution_steps"
done

echo "All runs are completed."
