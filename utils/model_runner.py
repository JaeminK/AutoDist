import os, sys
import time
import argparse

import torch

import transformers
from transformers import AutoModelForCausalLM, StaticCache

from autodist import load_or_create_tp_model, load_or_create_pp_model, load_or_create_tp_pp_model, ModelWrapper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelRunner:
    def __init__(
        self, 
        model_name_or_path, 
        device, 
        dtype, 
        cache_dir, 
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        local_rank=0, 
        world_size=1,
        capture_graph=True
    ):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype 
        self.device = device
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.local_rank = local_rank
        self.world_size = world_size
        self.execute_rank = 0  # Hardcoded to 0 for simplicity
        self.capture_graph = capture_graph
        
        if self.use_tp() and self.use_pp():
            self.model, self.model_config = load_or_create_tp_pp_model(
                model_name_or_path=self.model_name_or_path,
                cache_dir=self.cache_dir,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                execute_rank=self.execute_rank,
                dtype=self.dtype,
                device=self.device,
                local_rank=self.local_rank,
                world_size=self.world_size,
                capture_graph=self.capture_graph
            )
        
        elif self.use_tp():
            self.model, self.model_config = load_or_create_tp_model(
                model_name_or_path=self.model_name_or_path,
                cache_dir=self.cache_dir,
                tensor_parallel_size=self.tensor_parallel_size,
                execute_rank=self.execute_rank,
                dtype=self.dtype,
                device=self.device,
                local_rank=self.local_rank,
                world_size=self.world_size,
                capture_graph=self.capture_graph
            )
        elif self.use_pp():
            self.model, self.model_config = load_or_create_pp_model(
                model_name_or_path=self.model_name_or_path,
                cache_dir=self.cache_dir,
                pipeline_parallel_size=self.pipeline_parallel_size,
                execute_rank=self.execute_rank,
                dtype=self.dtype,
                device=self.device,
                local_rank=self.local_rank,
                world_size=self.world_size,
                capture_graph=self.capture_graph
            )
        else:
            if self.is_execute_rank():
                self.model, self.model_config = self.load_model(model_name_or_path)
            else:
                self.model = None
                self.model_config = None

        # Initialize past_key_values for all ranks that have a model
        if self.model is not None:
            if not "opt" in self.model_name_or_path.lower():
                self.past_key_values = StaticCache(
                    config=self.model_config,
                    batch_size=1,
                    max_cache_len=self.model_config.max_position_embeddings,
                    dtype=self.dtype,
                    device=self.device
                )
            else:
                self.past_key_values = None
        else:
            self.past_key_values = None

    def use_tp(self):
        return self.tensor_parallel_size > 1

    def use_pp(self):
        return self.pipeline_parallel_size > 1

    def is_execute_rank(self):
        return self.local_rank == self.execute_rank

    def get_stage_num(self):
        """Get the pipeline stage number for current rank."""
        if self.use_tp() and self.use_pp():
            return self.local_rank // self.tensor_parallel_size
        elif self.use_pp():
            return self.local_rank
        else:
            return 0

    def get_rank_in_stage(self):
        """Get the rank within the current pipeline stage."""
        if self.use_tp() and self.use_pp():
            return self.local_rank % self.tensor_parallel_size
        else:
            return 0

    def load_model(self, model_name_or_path: str):
        """
        Loads a HuggingFace model.
        """
        print(f"Loading model from {model_name_or_path}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=self.cache_dir,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )
        
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=self.cache_dir,
                torch_dtype=self.dtype
            )
        
        except Exception as e:
            print(f"Failed to load model '{model_name_or_path}': {e}")
            raise 

        return ModelWrapper(model.to(self.device), capture_graph=self.capture_graph), model.config

    def run_model(
        self, 
        input_ids,
        num_output_logits=1,
    ):
        with torch.no_grad():
            past_seen_tokens = self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + input_ids.shape[-1], device=input_ids.device)
            position_ids = cache_position.unsqueeze(0)
            
            outputs = self._run_model_forward(input_ids, position_ids, cache_position, num_output_logits)
        
        self.past_key_values = outputs.past_key_values
        if outputs.logits is None:
            return None
        
        logits = outputs.logits[:, -num_output_logits:, :].clone()
        return logits
    
    def _prepare_inputs_embeds(self, input_ids):
        if self.use_pp() and self.get_stage_num() > 0:
            batch_size, seq_len = input_ids.shape
            hidden_size = self.model_config.hidden_size
            return torch.empty(batch_size, seq_len, hidden_size, dtype=self.dtype, device=self.device)
        return None
    
    def _run_model_forward(self, input_ids, position_ids, cache_position, num_output_logits):
        model_kwargs = {
            'past_key_values': self.past_key_values,
            'use_cache': True,
            'position_ids': position_ids,
            'cache_position': cache_position,
            'input_ids': input_ids
        }
        
        try:
            model_kwargs['num_logits_to_keep'] = num_output_logits
            return self.model(**model_kwargs)
        except TypeError:
            model_kwargs.pop('num_logits_to_keep', None)
            return self.model(**model_kwargs)

    def warmup(self):
        with torch.no_grad():
            warmup_input_ids = torch.randint(0, self.model_config.vocab_size, (1, 100)).to(self.device)
            
            try:
                return self.model(input_ids=warmup_input_ids, num_logits_to_keep=1)
            except TypeError:
                return self.model(input_ids=warmup_input_ids)
