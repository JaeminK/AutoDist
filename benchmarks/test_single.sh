#! /bin/bash
export CUDA_LAUNCH_BLOCKING=1

# nsys profile --trace=cuda,nvtx --cuda-graph=node -o ./single --force-overwrite true \
python ../main.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 512 \