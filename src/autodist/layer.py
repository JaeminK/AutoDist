import torch

from .comm_utils import CommUtils

class TensorParallelEmbedding(torch.nn.Embedding):
    def __init__(self, weight: torch.Tensor, padding_idx: int, comm_utils: CommUtils):
        num_embeddings, embedding_dim = weight.shape
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=weight.dtype)
        self.comm_utils = comm_utils
        self.is_tp_module = True
        self.weight.copy_(weight)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output = self.comm_utils.all_gather(output)
        return output


class ColumnParallelLinear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, comm_utils: CommUtils, gather_output: bool = False):
        out_features, in_features = weight.shape
        super().__init__(in_features, out_features, bias=True if bias is not None else False, dtype=weight.dtype)
        self.comm_utils = comm_utils
        self.is_tp_module = True
        self.gather_output = gather_output
        self.weight.copy_(weight)
        if bias is not None:
            self.bias.copy_(bias)

    def forward(self, input: torch.Tensor):
        output = torch.nn.functional.linear(input, self.weight)
        if self.bias is not None:
            output += self.bias
        if self.gather_output:
            output = self.comm_utils.all_gather(output)
        return output


class RowParallelLinear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, comm_utils: CommUtils):
        out_features, in_features = weight.shape
        super().__init__(in_features, out_features, bias=True if bias is not None else False, dtype=weight.dtype)
        self.comm_utils = comm_utils
        self.is_tp_module = True
        self.weight.copy_(weight)
        if bias is not None:
            self.bias.copy_(bias)

    def forward(self, input: torch.Tensor):
        output = torch.nn.functional.linear(input, self.weight)
        output = self.comm_utils.all_reduce(output)
        if self.bias is not None:
            output += self.bias
        return output


class PipelineParallelEmbedding(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Module, dtype: torch.dtype, stage_num: int, is_first_stage: bool = False):
        super().__init__()
        self.stage_num = stage_num
        self.is_first_stage = is_first_stage
        self.dtype = dtype
        self.is_pp_module = True
        
        if is_first_stage:
            self.embedding = embedding
        else:
            self.embedding_dim = embedding.embedding_dim
            self.num_embeddings = embedding.num_embeddings
            self.padding_idx = getattr(embedding, 'padding_idx', None)
    
    def forward(self, input_ids):
        if self.is_first_stage:
            return self.embedding(input_ids)
        else:
            return torch.empty(input_ids.shape[0], input_ids.shape[1], self.embedding_dim, dtype=self.dtype, device=input_ids.device)


class PipelineParallelFinalLayerNorm(torch.nn.Module):
    def __init__(self, final_layer_norm: torch.nn.Module, stage_num: int, is_last_stage: bool = False):
        super().__init__()
        self.stage_num = stage_num
        self.is_last_stage = is_last_stage
        self.is_pp_module = True
        
        if is_last_stage:
            self.final_layer_norm = final_layer_norm
    
    def forward(self, hidden_states):
        if self.is_last_stage:
            return self.final_layer_norm(hidden_states)
        else:
            return hidden_states


class PipelineParallelLMHead(torch.nn.Module):
    def __init__(self, lm_head: torch.nn.Module, stage_num: int, is_last_stage: bool = False):
        super().__init__()
        self.stage_num = stage_num
        self.is_last_stage = is_last_stage
        self.is_pp_module = True
        
        if is_last_stage:
            self.lm_head = lm_head
        else:
            self.in_features = lm_head.in_features
            self.out_features = lm_head.out_features
            self.bias = lm_head.bias is not None
    
    def forward(self, hidden_states):
        if self.is_last_stage:
            return self.lm_head(hidden_states)
        else:
            return None


class PipelineParallelTransformerLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, comm_utils: CommUtils, stage_num: int, 
                 is_first_layer_in_stage: bool = False, is_last_layer_in_stage: bool = False, layer_idx: int = 0):
        super().__init__()
        self.layer = layer
        self.comm_utils = comm_utils
        self.stage_num = stage_num
        self.is_first_layer_in_stage = is_first_layer_in_stage
        self.is_last_layer_in_stage = is_last_layer_in_stage
        self.layer_idx = layer_idx
        self.is_pp_module = True
        
        self._adjust_layer_idx_for_attention()
    
    def _adjust_layer_idx_for_attention(self):
        for name, module in self.layer.named_modules():
            if hasattr(module, 'layer_idx'):
                setattr(module, 'layer_idx', self.layer_idx)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            layer = super().__getattr__("layer")
        except AttributeError:
            raise
        return getattr(layer, name)

    def forward(self, *args, **kwargs):
        if self.is_first_layer_in_stage and self.stage_num > 0:
            if 'hidden_states' in kwargs:
                kwargs['hidden_states'] = self.comm_utils.recv(kwargs['hidden_states'], self.stage_num - 1)
            else:
                recv_args = self.comm_utils.recv(args[0], self.stage_num - 1)
                args = (recv_args,) + args[1:]

        output = self.layer(*args, **kwargs)
        
        if self.is_last_layer_in_stage and self.stage_num < self.comm_utils.world_size - 1:
            if not isinstance(output, tuple):
                self.comm_utils.send(output, self.stage_num + 1)
            else:
                self.comm_utils.send(output[0], self.stage_num + 1)
        
        return output


class TensorPipelineParallelEmbedding(torch.nn.Embedding):
    def __init__(self, weight: torch.Tensor, padding_idx: int, tp_comm_utils: CommUtils, 
                 stage_num: int, is_first_stage: bool = False):
        num_embeddings, embedding_dim = weight.shape
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=weight.dtype)
        self.tp_comm_utils = tp_comm_utils
        self.stage_num = stage_num
        self.is_first_stage = is_first_stage
        self.is_tp_pp_module = True
        self.weight.copy_(weight)

    def forward(self, input_ids):
        if not self.is_first_stage:
            return torch.empty(input_ids.shape[0], input_ids.shape[1], self.embedding_dim, 
                             dtype=self.weight.dtype, device=input_ids.device)
        
        output = super().forward(input_ids)
        output = self.tp_comm_utils.all_gather(output)
        return output


class TensorPipelineParallelLMHead(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, tp_comm_utils: CommUtils,
                 stage_num: int, is_last_stage: bool = False):
        out_features, in_features = weight.shape
        super().__init__(in_features, out_features, bias=True if bias is not None else False, dtype=weight.dtype)
        self.tp_comm_utils = tp_comm_utils
        self.stage_num = stage_num
        self.is_last_stage = is_last_stage
        self.is_tp_pp_module = True
        self.weight.copy_(weight)
        if bias is not None:
            self.bias.copy_(bias)

    def forward(self, hidden_states):
        if not self.is_last_stage:
            return None
        
        output = torch.nn.functional.linear(hidden_states, self.weight)
        if self.bias is not None:
            output += self.bias
        output = self.tp_comm_utils.all_gather(output)
        return output