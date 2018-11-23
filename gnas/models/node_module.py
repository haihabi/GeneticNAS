import torch
import torch.nn as nn
from gnas.search_space.space_config import OperationConfig


class NodeModule(nn.Module):
    def __init__(self, max_inputs: int, oc: OperationConfig):
        super(NodeModule, self).__init__()
        self.max_inputs = max_inputs
        self.oc = oc

        self._build_operators()
        self.weight = None
        self.merge = None
        self.non_linear = None

    def _build_operators(self):
        self.nl_module = self.oc.get_non_linear_modules()
        self.merge_module = self.oc.get_merge_op_modules()
        self.weight_modules = self.oc.get_weight_modules(self.max_inputs)

    def forward(self, inputs):
        x = [w(i) for i, w in zip(inputs, self.weight)]
        x = self.merge(x)
        if isinstance(x, int):
            print("a")
        return self.non_linear(x)

    def set_current_node_config(self, node_config):
        self.node_config = node_config
        self.weight = [self.weight_modules[i][wc] for i, wc in enumerate(node_config.connection_index)]
        self.merge = self.merge_module[node_config.mo_index]
        self.non_linear = self.nl_module[node_config.nl_index]
