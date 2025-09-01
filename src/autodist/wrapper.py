import torch

from .comm_utils import CommUtils


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, capture_graph: bool = False):
        super().__init__()
        self.model = model
        
        self.capture_graph = capture_graph
        if self.capture_graph:
            self.prepare_graph()

    def prepare_graph(self):
        self.warmup_count = {}
        self.graphs = {}
        self.static_input = {}
        self.static_output = {}

    def is_graph_capture_needed(self, config):
        if (count := self.warmup_count.get(config, 0)) < 5:
            self.warmup_count[config] = count + 1
            return False
        if not self.is_graph_capture_done(config):
            self.graphs[config] = torch.cuda.CUDAGraph()
            return True
        return False

    def is_graph_capture_done(self, config):
        return self.graphs.get(config, None) is not None

    def get_config(self, *args, **kwargs):
        batch_size, sequence_length = kwargs.get("input_ids", args[0]).shape[:2] if len(args) > 0 else kwargs["input_ids"].shape[:2]
        if kwargs.get("num_logits_to_keep", None) is not None:
            num_logits_to_keep = kwargs["num_logits_to_keep"]
            return f"{batch_size},{sequence_length},{num_logits_to_keep}"
        return f"{batch_size},{sequence_length}"

    def get_all_tensors_in_args(self, *args, **kwargs):
        tensors = {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors[arg] = arg
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
        return tensors

    def set_all_tensors_to_static_input(self, *args, **kwargs):
        config = self.get_config(*args, **kwargs)
        input_tensors = self.get_all_tensors_in_args(*args, **kwargs)
        for key, tensor in input_tensors.items():
            self.static_input[config][key].copy_(tensor)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if self.capture_graph:
            config = self.get_config(*args, **kwargs)
            if self.is_graph_capture_needed(config):
                self.static_input[config] = self.get_all_tensors_in_args(*args, **kwargs)
                with torch.cuda.graph(self.graphs[config]):
                    output = self.model(*args, **kwargs)
                self.static_output[config] = output
            elif self.is_graph_capture_done(config):
                self.set_all_tensors_to_static_input(*args, **kwargs)
                self.graphs[config].replay()
                return self.static_output[config]
        return self.model(*args, **kwargs)


class TensorParallelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, comm_utils: CommUtils, capture_graph: bool = False):
        super().__init__()
        self.model = model
        self.comm_utils = comm_utils
        
        for name, module in self.model.named_modules():
            if hasattr(module, "comm_utils"):
                setattr(module, "comm_utils", self.comm_utils)
        
        self.capture_graph = capture_graph
        if self.capture_graph:
            self.prepare_graph()

    def prepare_graph(self):
        self.warmup_count = {}
        self.graphs = {}
        self.static_input = {}
        self.static_output = {}

    def is_graph_capture_needed(self, config):
        if (count := self.warmup_count.get(config, 0)) < 5:
            self.warmup_count[config] = count + 1
            return False
        if not self.is_graph_capture_done(config):
            self.graphs[config] = torch.cuda.CUDAGraph()
            return True
        return False

    def is_graph_capture_done(self, config):
        return self.graphs.get(config, None) is not None

    def get_config(self, *args, **kwargs):
        batch_size, sequence_length = kwargs.get("input_ids", args[0]).shape[:2] if len(args) > 0 else kwargs["input_ids"].shape[:2]
        if kwargs.get("num_logits_to_keep", None) is not None:
            num_logits_to_keep = kwargs["num_logits_to_keep"]
            return f"{batch_size},{sequence_length},{num_logits_to_keep}"
        return f"{batch_size},{sequence_length}"

    def get_all_tensors_in_args(self, *args, **kwargs):
        tensors = {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors[arg] = arg
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
        return tensors

    def set_all_tensors_to_static_input(self, *args, **kwargs):
        config = self.get_config(*args, **kwargs)
        input_tensors = self.get_all_tensors_in_args(*args, **kwargs)
        for key, tensor in input_tensors.items():
            self.static_input[config][key].copy_(tensor)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if self.capture_graph:
            config = self.get_config(*args, **kwargs)
            if self.is_graph_capture_needed(config):
                self.static_input[config] = self.get_all_tensors_in_args(*args, **kwargs)
                with torch.cuda.graph(self.graphs[config]):
                    output = self.model(*args, **kwargs)
                self.static_output[config] = output
            elif self.is_graph_capture_done(config):
                self.set_all_tensors_to_static_input(*args, **kwargs)
                self.graphs[config].replay()
                return self.static_output[config]
        return self.model(*args, **kwargs)


class PipelineParallelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, comm_utils: CommUtils, capture_graph: bool = False, stage_num: int = 0, rank_in_stage: int = 0):
        super().__init__()
        self.model = model
        self.comm_utils = comm_utils
        self.stage_num = stage_num
        self.rank_in_stage = rank_in_stage
        
        for name, module in self.model.named_modules():
            if hasattr(module, "comm_utils"):
                setattr(module, "comm_utils", self.comm_utils)
        
        self.capture_graph = capture_graph
        if self.capture_graph:
            self.prepare_graph()
            
    def prepare_graph(self):
        self.warmup_count = {}
        self.graphs = {}
        self.static_input = {}
        self.static_output = {}

    def is_graph_capture_needed(self, config):
        if (count := self.warmup_count.get(config, 0)) < 5:
            self.warmup_count[config] = count + 1
            return False
        if not self.is_graph_capture_done(config):
            self.graphs[config] = torch.cuda.CUDAGraph()
            return True
        return False

    def is_graph_capture_done(self, config):
        return self.graphs.get(config, None) is not None

    def get_config(self, *args, **kwargs):
        # Determine which input tensor to use for config
        if kwargs.get("input_ids") is not None:
            input_tensor = kwargs["input_ids"]
        elif len(args) > 0:
            input_tensor = args[0]
        else:
            raise ValueError("No input tensor found in args or kwargs")
        
        batch_size, sequence_length = input_tensor.shape[:2]
        
        if kwargs.get("num_logits_to_keep", None) is not None:
            num_logits_to_keep = kwargs["num_logits_to_keep"]
            return f"{batch_size},{sequence_length},{num_logits_to_keep}"
        return f"{batch_size},{sequence_length}"

    def get_all_tensors_in_args(self, *args, **kwargs):
        tensors = {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors[arg] = arg
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
        return tensors

    def set_all_tensors_to_static_input(self, *args, **kwargs):
        config = self.get_config(*args, **kwargs)
        input_tensors = self.get_all_tensors_in_args(*args, **kwargs)
        for key, tensor in input_tensors.items():
            self.static_input[config][key].copy_(tensor)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if self.capture_graph:
            config = self.get_config(*args, **kwargs)
            if self.is_graph_capture_needed(config):
                self.static_input[config] = self.get_all_tensors_in_args(*args, **kwargs)
                with torch.cuda.graph(self.graphs[config]):
                    output = self.model(*args, **kwargs)
                self.static_output[config] = output
            elif self.is_graph_capture_done(config):
                self.set_all_tensors_to_static_input(*args, **kwargs)
                self.graphs[config].replay()
                return self.static_output[config]
        return self.model(*args, **kwargs)


class TensorPipelineParallelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, tp_comm_utils: CommUtils, pp_comm_utils: CommUtils, 
                 capture_graph: bool = False, stage_num: int = 0, rank_in_stage: int = 0):
        super().__init__()
        self.model = model
        self.tp_comm_utils = tp_comm_utils
        self.pp_comm_utils = pp_comm_utils
        self.stage_num = stage_num
        self.rank_in_stage = rank_in_stage
        
        for name, module in self.model.named_modules():
            if hasattr(module, "comm_utils"):
                if hasattr(module, "is_tp_pp_module") and module.is_tp_pp_module:
                    setattr(module, "comm_utils", self.tp_comm_utils)
                elif hasattr(module, "is_tp_module") and module.is_tp_module:
                    setattr(module, "comm_utils", self.tp_comm_utils)
                elif hasattr(module, "is_pp_module") and module.is_pp_module:
                    setattr(module, "comm_utils", self.pp_comm_utils)
                    if hasattr(module, "tp_comm_utils"):
                        setattr(module, "tp_comm_utils", self.tp_comm_utils)
                else:
                    setattr(module, "comm_utils", self.tp_comm_utils)
        
        self.capture_graph = capture_graph
        if self.capture_graph:
            self.prepare_graph()
            
    def prepare_graph(self):
        self.warmup_count = {}
        self.graphs = {}
        self.static_input = {}
        self.static_output = {}

    def is_graph_capture_needed(self, config):
        if (count := self.warmup_count.get(config, 0)) < 5:
            self.warmup_count[config] = count + 1
            return False
        if not self.is_graph_capture_done(config):
            self.graphs[config] = torch.cuda.CUDAGraph()
            return True
        return False

    def is_graph_capture_done(self, config):
        return self.graphs.get(config, None) is not None

    def get_config(self, *args, **kwargs):
        if kwargs.get("input_ids") is not None:
            input_tensor = kwargs["input_ids"]
        elif len(args) > 0:
            input_tensor = args[0]
        else:
            raise ValueError("No input tensor found in args or kwargs")
        
        batch_size, sequence_length = input_tensor.shape[:2]
        
        if kwargs.get("num_logits_to_keep", None) is not None:
            num_logits_to_keep = kwargs["num_logits_to_keep"]
            return f"{batch_size},{sequence_length},{num_logits_to_keep}"
        return f"{batch_size},{sequence_length}"

    def get_all_tensors_in_args(self, *args, **kwargs):
        tensors = {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensors[arg] = arg
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
        return tensors

    def set_all_tensors_to_static_input(self, *args, **kwargs):
        config = self.get_config(*args, **kwargs)
        input_tensors = self.get_all_tensors_in_args(*args, **kwargs)
        for key, tensor in input_tensors.items():
            self.static_input[config][key].copy_(tensor)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        if self.capture_graph:
            config = self.get_config(*args, **kwargs)
            if self.is_graph_capture_needed(config):
                self.static_input[config] = self.get_all_tensors_in_args(*args, **kwargs)
                with torch.cuda.graph(self.graphs[config]):
                    output = self.model(*args, **kwargs)
                self.static_output[config] = output
            elif self.is_graph_capture_done(config):
                self.set_all_tensors_to_static_input(*args, **kwargs)
                self.graphs[config].replay()
                return self.static_output[config]
        return self.model(*args, **kwargs)

