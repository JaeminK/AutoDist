# AutoDist

## Description
AutoDist automatically converts a single-GPU HuggingFace model into a distributed setup using tensor parallelism and/or pipeline parallelism for efficient inference.

## Features
- Tensor Parallel (TP) shard creation and caching
- Pipeline Parallel (PP) stage creation and caching
- Optional CUDA graph capture via `ModelWrapper` for reduced kernel launch overhead
- Simple CLI to run single-GPU, TP, or PP inference

## Requirements
- Python 3.10+
- PyTorch 2.7+
- CUDA-capable GPU(s) with a compatible PyTorch build
- PyTorch and Transformers (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JaeminK/AutoDist.git
   cd AutoDist
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package (editable install recommended for development):
   ```bash
   pip install -e .
   ```

## Quick Start

- Single GPU:
  ```bash
  python main.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 512
  ```

- Tensor Parallel (example: 2-way TP on 2 GPUs):
  ```bash
  torchrun --nproc_per_node 2 main.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --tensor-parallel-size 2 \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 512
  ```

- Pipeline Parallel (example: 2 stages on 2 GPUs):
  ```bash
  torchrun --nproc_per_node 2 main.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --pipeline-parallel-size 2 \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 512
  ```

Benchmark scripts are also available:
- `benchmarks/test_single.sh`
- `benchmarks/test_tp_dist.sh`
- `benchmarks/test_pp_dist.sh`

## CLI Options
`main.py` supports the following options:
- `--model` (str): Hugging Face repo or local path (default: `facebook/opt-1.3b`).
- `--cache-dir` (str): HF cache directory (default: `/workspace/cache`).
- `--output-dir` (str): Directory to write results (default: `./results`).
- `--tensor-parallel-size` (int): Number of TP shards (default: `1`).
- `--pipeline-parallel-size` (int): Number of PP stages (default: `1`).
- `--batch-size` (int): Must be `1` for single-token inference (default: `1`).
- `--seed` (int): Random seed (default: `1234`).
- `--local-rank` (int): Local rank (auto-set by `torchrun`).
- `--world-size` (int): World size (auto-set by `torchrun`).
- `--min-output-length` (int): Minimum generated tokens (default: `1`).
- `--max-output-length` (int): Maximum generated tokens (default: `157`).

Notes:
- For TP/PP runs, `world_size` must equal `tensor_parallel_size * pipeline_parallel_size`.
- `local_rank` must be less than the product of TP and PP sizes.

## Benchmarks
See the scripts in `benchmarks/` for ready-to-run examples matching the commands above.
