import os, sys
import time
import argparse

import torch
import torch.distributed as dist

import transformers
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ModelRunner, DistributedSampler

                
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--cache-dir", type=str, default="/workspace/cache", help="Cache directory for models.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Result directory for models.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    parser.add_argument("--min-output-length", type=int, default=1)
    parser.add_argument("--max-output-length", type=int, default=157)
    return parser.parse_args()

def get_eos_token_ids(tokenizer, model_runner):
    if isinstance(tokenizer.eos_token_id, list):
        eos_token_ids = tokenizer.eos_token_id
    else:
        eos_token_ids = [tokenizer.eos_token_id]
    
    if isinstance(tokenizer.pad_token_id, list):
        pad_token_ids = tokenizer.pad_token_id
    else:
        pad_token_ids = [tokenizer.pad_token_id]
    
    gen_config_eos_id = model_runner.model.model.generation_config.eos_token_id
    if isinstance(gen_config_eos_id, list):
        eos_token_ids.extend(gen_config_eos_id)
    else:
        eos_token_ids.append(gen_config_eos_id)

    eos_token_ids.extend(pad_token_ids)
    eos_token_ids = list(set(eos_token_ids))
    return eos_token_ids

def main():
    args = parse_args()

    assert args.tensor_parallel_size <= args.world_size, f"tensor_parallel_size {args.tensor_parallel_size} must be less than or equal to world_size {args.world_size}"
    assert args.pipeline_parallel_size <= args.world_size, f"pipeline_parallel_size {args.pipeline_parallel_size} must be less than or equal to world_size {args.world_size}"
    assert args.local_rank < args.tensor_parallel_size*args.pipeline_parallel_size, f"local_rank {args.local_rank} must be less than parallel_size {args.tensor_parallel_size*args.pipeline_parallel_size}"
    assert args.world_size == args.tensor_parallel_size*args.pipeline_parallel_size, f"world_size {args.world_size} must be equal to tensor_parallel_size {args.tensor_parallel_size} * pipeline_parallel_size {args.pipeline_parallel_size}"
    assert args.batch_size == 1, f"batch_size {args.batch_size} must be 1 for single-token inference"

    # set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    transformers.set_seed(args.seed)
    # Device configuration
    device = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

    model_runner = ModelRunner(
        model_name_or_path=args.model,
        device=device,
        dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        local_rank=args.local_rank,
        world_size=args.world_size,
        capture_graph=True
    )

    eos_token_ids = get_eos_token_ids(tokenizer, model_runner)

    sampler_manager = DistributedSampler(
        vocab_size=model_runner.model_config.vocab_size,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        local_rank=args.local_rank,
        world_size=args.world_size,
        eos_token_ids=eos_token_ids,
        max_output_length=args.max_output_length
    )

    for _ in range(5):
        _ = model_runner.warmup()

    prompt = "Once upon a time,"
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_ids = prompt_ids.clone()
    output_ids = torch.tensor([], dtype=torch.long, device=device)
    stop_flag = False

    torch.cuda.synchronize()
    
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    record_time = []
    torch.cuda.synchronize()
    
    if dist.is_initialized():
        dist.barrier()
    
    print(f"start inference.. {args.local_rank}")
    if args.local_rank == 0:
        print(prompt, end='', flush=True)

    while not stop_flag:
        start_time.record()
        logits = model_runner.run_model(input_ids=input_ids, num_output_logits=1)
        
        tokens = sampler_manager.sample_and_distribute(logits)
        if sampler_manager.should_sample:
            output_ids, stop_flag = sampler_manager.post_process(tokens, output_ids)
        
        stop_flag = sampler_manager.broadcast_stop_flag(stop_flag, device)
        input_ids = tokens[:, -1:]
        
        end_time.record()
        torch.cuda.synchronize()
        record_time.append(start_time.elapsed_time(end_time))
        
        if args.local_rank == 0:
            new_token = tokens[0, -1].item()
            print(tokenizer.decode([new_token]), end='', flush=True)
    
    torch.cuda.synchronize()
    
    if dist.is_initialized():
        dist.barrier()
    
    if sampler_manager.should_output_results():
        final_output = torch.cat([prompt_ids, output_ids.view(1, -1)], dim=1)
        # print("============================================")
        # print(tokenizer.decode(final_output[0].tolist(), skip_special_tokens=True))
        print("============================================")
        record_time = record_time[10:] # remove first 10 records for graph capture warmup
        print(f"\ngeneration latency : {sum(record_time) / len(record_time):.2f} ms")
        print(f"throughput : {1 / (sum(record_time) / len(record_time))* 1e3:.2f} tokens/second")


if __name__ == '__main__':
    main()