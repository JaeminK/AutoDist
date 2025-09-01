#! /bin/bash
export CUDA_LAUNCH_BLOCKING=1

# nsys profile --trace=cuda,nvtx --cuda-graph=node -o ./pp_dist --force-overwrite true \
torchrun --nproc_per_node 2 ../main.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --pipeline-parallel-size 2 \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 512 \