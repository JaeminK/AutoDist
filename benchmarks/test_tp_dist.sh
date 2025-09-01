#! /bin/bash
export CUDA_LAUNCH_BLOCKING=1

# nsys profile --trace=cuda,nvtx --cuda-graph=node -o ./tp_dist --force-overwrite true \
torchrun --nproc_per_node 4 ../main.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir /home/work/cache \
    --output-dir ./results \
    --tensor-parallel-size 4 \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 512 \