#!/bin/bash

EXP_DIR="results_h100_cu126/qwen35-9b_rr_sweep"

XVAR="request_rate"
CURVE="random_input_len"

metrics=(
  output_throughput
  p99_ttft_ms
  p99_itl_ms
  p99_e2el_ms
  p99_tpot_ms
)

for metric in "${metrics[@]}"; do
  echo "Plotting $metric ..."
  
  vllm bench sweep plot "$EXP_DIR" \
    --var-x "$XVAR" \
    --var-y "$metric" \
    --curve-by "$CURVE" \
    --fig-name "${metric}_vs_rr"

done