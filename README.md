# LLM Screening Test Benchmark

This repo contains the code, configs, and presentation materials for a screening test on **model intelligence benchmarking** and **LLM serving performance evaluation**.

## Goal

The project studies two dimensions of model quality:

1. **Intelligence / capability**
   - Benchmark:
     - `Qwen3.5-9B`
     - `Qwen3.5-27B`
     - `GLM-4.7-Flash`
   - Focus on a small but representative set of tasks rather than exhaustive coverage.

2. **Runtime performance**
   - Evaluate:
     - `Qwen3.5-9B`
     - `GLM-4.7-Flash`
   - Serving backends:
     - `vLLM` / `SGLang` / `TensorRT-LLM` (depending on environment support)
   - Analyze major performance factors including:
     - batch size
     - input/output sequence length
     - sequence length variability
     - throughput / latency tradeoffs

## Repository Structure

```text
.
├── intelligence_eval/   # scripts for capability evaluation
├── performance_eval/    # scripts for runtime and throughput benchmarking
├── present/             # final presentation
└── README.md