import os, sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.distributed as dist


class Sampler(torch.nn.Module):
    def __init__(self,
                 max_vocab_size: int = None,
                 min_vocab_size: int = None):
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.min_vocab_size = min_vocab_size

    def forward(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        current_vocab_size = logits.shape[-1]
        if self.min_vocab_size is not None and current_vocab_size > self.min_vocab_size:
            logits = logits[:, :, :self.min_vocab_size].clone()

        sampled_tokens = logits.argmax(dim=-1)
        return sampled_tokens
    

class DistributedSampler:
    def __init__(self, vocab_size, tensor_parallel_size, pipeline_parallel_size, local_rank, world_size, eos_token_ids, max_output_length):
        self.vocab_size = vocab_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.local_rank = local_rank
        self.world_size = world_size
        self.eos_token_ids = eos_token_ids
        self.eos_token_ids_tensor = torch.tensor(self.eos_token_ids, device=torch.cuda.current_device())
        self.max_output_length = max_output_length
        
        self.should_sample = self._determine_sampling_rank()
        
        if self.should_sample:
            self.sampler = Sampler(max_vocab_size=self.vocab_size)
        else:
            self.sampler = None
    
    def _determine_sampling_rank(self):
        if self.tensor_parallel_size > 1:
            return self.local_rank == 0
        elif self.pipeline_parallel_size > 1:
            stage_num = self.local_rank
            return stage_num == self.pipeline_parallel_size - 1
        else:
            return True
    
    def sample_and_distribute(self, logits):
        if self.should_sample:
            tokens = self.sampler(logits=logits)
        else:
            tokens = torch.empty((1, 1), dtype=torch.long, device=torch.cuda.current_device())
            
        if self.tensor_parallel_size > 1:
            dist.broadcast(tokens, src=0)
        elif self.pipeline_parallel_size > 1:
            last_stage_rank = (self.pipeline_parallel_size - 1) * self.pipeline_parallel_size - 1
            dist.broadcast(tokens, src=last_stage_rank)
        return tokens
    
    def broadcast_stop_flag(self, stop_flag, device):
        if self.tensor_parallel_size > 1 or self.pipeline_parallel_size > 1:
            stop_flag_tensor = torch.tensor([stop_flag], dtype=torch.bool, device=device)
            
            if self.tensor_parallel_size > 1:
                dist.broadcast(stop_flag_tensor, src=0)
            else:
                last_stage_rank = (self.pipeline_parallel_size - 1) * self.pipeline_parallel_size - 1
                dist.broadcast(stop_flag_tensor, src=last_stage_rank)
            
            return stop_flag_tensor.item()
        
        return stop_flag
    
    def should_output_results(self):
        return self.should_sample
    
    def post_process(self, new_tokens, output_ids):
        stop_flag = False
        eos_index = -1
        
        eos_mask = torch.isin(new_tokens, self.eos_token_ids_tensor)
        if eos_mask.any():
            eos_index = torch.where(eos_mask)[0][0].item()

        if eos_index != -1:
            output_ids = torch.cat([output_ids, new_tokens[0][:eos_index+1]], dim=0)
            stop_flag = True
        else:
            output_ids = torch.cat([output_ids, new_tokens[0]], dim=0)
        
        stop_flag = stop_flag or output_ids.shape[-1] >= self.max_output_length
        return output_ids, stop_flag