import torch
import torch.distributed as dist

class CommUtils:
    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)

    def broadcast(self, tensor: torch.Tensor, source_rank: int = 0):
        dist.broadcast(tensor, src=source_rank, group=self.group)
        return tensor

    def gather(self, tensor: torch.Tensor, dst_rank: int = 0):
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.gather(tensor, tensor_list, dst=dst_rank, group=self.group)
        return torch.cat(tensor_list, dim=-1)

    def all_reduce(self, tensor: torch.Tensor):
        dist.all_reduce(tensor, group=self.group)
        return tensor

    def all_gather(self, tensor: torch.Tensor):
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor, group=self.group)
        return torch.cat(tensor_list, dim=-1)

    def send(self, tensor: torch.Tensor, dst_rank: int):
        dist.send(tensor, dst=dst_rank, group=self.group)
        return tensor

    def recv(self, tensor: torch.Tensor, src_rank: int):
        dist.recv(tensor, src=src_rank, group=self.group)
        return tensor