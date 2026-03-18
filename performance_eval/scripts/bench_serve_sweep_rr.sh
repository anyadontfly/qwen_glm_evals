#!/usr/bin/env bash
set -euo pipefail

MODEL=$1
GPU_IDX=$2
PORT=$3

OUTDIR="results"
MODEL_TAG=$(echo "$MODEL" | sed 's|/|-|g')

CUDA_VISIBLE_DEVICES=${GPU_IDX} vllm bench sweep serve \
  --serve-cmd "vllm serve ${MODEL} --served-model-name ${MODEL} --port ${PORT} --max-model-len 200k --enable-prefix-caching" \
  --bench-cmd "vllm bench serve --model ${MODEL} --num-warmups 16 --host 127.0.0.1 --port ${PORT} --max-concurrency 16 --dataset-name random --num-prompts 128 --random-input-len 2048 --random-output-len 128 --random-range-ratio 0.2" \
  --bench-params setups/rr.json \
  --output-dir ${OUTDIR} \
  --experiment-name ${MODEL_TAG}_rr_sweep_random \
  --num-runs 1
