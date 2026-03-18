#!/usr/bin/env bash
set -euo pipefail

MODEL=$1
GPU_IDX=$2

OUTDIR="results_offline"
mkdir -p ${OUTDIR}

MODEL_TAG=$(echo "$MODEL" | sed 's|/|-|g')

# Sweep only input len + max_num_seqs
INPUT_LEN_LIST=(512 2048 8192)
MAX_NUM_SEQS_LIST=(2 8 32)

NUM_PROMPTS=128
OUTPUT_LEN=32   # keep decode minimal (good practice)

for INPUT_LEN in "${INPUT_LEN_LIST[@]}"; do
  for MAX_NUM_SEQS in "${MAX_NUM_SEQS_LIST[@]}"; do

    OUT_FILE="${OUTDIR}/${MODEL_TAG}_in${INPUT_LEN}_seq${MAX_NUM_SEQS}.json"

    echo "========================================="
    echo "Model: ${MODEL}"
    echo "input_len: ${INPUT_LEN}"
    echo "max_num_seqs: ${MAX_NUM_SEQS}"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=${GPU_IDX} vllm bench throughput \
      --model ${MODEL} \
      --dataset-name random \
      --num-prompts ${NUM_PROMPTS} \
      --random-input-len ${INPUT_LEN} \
      --random-output-len ${OUTPUT_LEN} \
      --max-num-seqs ${MAX_NUM_SEQS} \
      --block-size 16 \
      --output-json ${OUT_FILE}

    echo "Saved → ${OUT_FILE}"
    echo

  done
done

echo "Offline sweep finished."
