#!/usr/bin/env bash
set -euo pipefail

MODEL=$1
VAR_X=$2
OUTDIR="results"

MODEL_TAG=$(echo "$MODEL" | sed 's|/|-|g')

case "$VAR_X" in
  request_rate|rr)
    X_KEY="request_rate"
    EXP_DIR="${OUTDIR}/${MODEL_TAG}_rr_sweep_random"
    PREFIX="${MODEL_TAG}_rr"
    ;;
  max_concurrency|mc)
    X_KEY="max_concurrency"
    EXP_DIR="${OUTDIR}/${MODEL_TAG}_mc_sweep_random"
    PREFIX="${MODEL_TAG}_mc"
    ;;
  random_input_len|inp|input_len)
    X_KEY="random_input_len"
    EXP_DIR="${OUTDIR}/${MODEL_TAG}_inp_len_sweep_random"
    PREFIX="${MODEL_TAG}_inp_len"
    ;;
  random_output_len|out|output_len)
    X_KEY="random_output_len"
    EXP_DIR="${OUTDIR}/${MODEL_TAG}_out_len_sweep_random"
    PREFIX="${MODEL_TAG}_out_len"
    ;;
  burstiness|burst)
    X_KEY="burstiness"
    EXP_DIR="${OUTDIR}/${MODEL_TAG}_burst_sweep_random"
    PREFIX="${MODEL_TAG}_burst"
    ;;
  *)
    echo "Unsupported var-x: $VAR_X"
    echo "Use one of: rr, mc, inp, out, burst"
    exit 1
    ;;
esac

echo "Plotting from ${EXP_DIR} with x-axis ${X_KEY}"

vllm bench sweep plot "${EXP_DIR}" \
  --var-x "${X_KEY}" \
  --var-y output_throughput \
  --fig-name "${PREFIX}_vs_output_throughput"

vllm bench sweep plot "${EXP_DIR}" \
  --var-x "${X_KEY}" \
  --var-y p99_ttft_ms \
  --fig-name "${PREFIX}_vs_p99_ttft_ms"

vllm bench sweep plot "${EXP_DIR}" \
  --var-x "${X_KEY}" \
  --var-y p99_tpot_ms \
  --fig-name "${PREFIX}_vs_p99_tpot_ms"

vllm bench sweep plot "${EXP_DIR}" \
  --var-x "${X_KEY}" \
  --var-y p99_itl_ms \
  --fig-name "${PREFIX}_vs_p99_itl_ms"

vllm bench sweep plot "${EXP_DIR}" \
  --var-x "${X_KEY}" \
  --var-y p99_e2el_ms \
  --fig-name "${PREFIX}_vs_p99_e2el_ms"

echo "Done."
