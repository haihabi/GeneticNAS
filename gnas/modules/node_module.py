import torch
import torch.nn as nn
from gnas.modules.operation_modules import get_module


class NodeModule(nn.Module):
    def __init__(self, max_inputs: int, n_channels: int, operation_config):
        super(NodeModule, self).__init__()
        self.max_inputs = max_inputs
        self.oc = operation_config
        self.n_channels = n_channels
        self.nl_module = self.oc.get_non_linear_modules()
        self.merge_module = self.oc.get_merge_op_modules()
        self.weight_modules = self.oc.get_weight_modules(max_inputs, n_channels)
        [[self.add_module(str(i) + str(j), op) for j, op in enumerate(input_w)] for i, input_w in
         enumerate(self.weight_modules)]

        self.weight = None
        self.merge = None
        self.non_linear = None

    def forward(self, inputs):
        x = [w(i) for i, w in zip(inputs, self.weight)]
        x = self.merge(x)
        return self.non_linear(x)

    def set_current_node_config(self, node_config):
        self.node_config = node_config
        self.weight = [self.weight_modules[i][wc] for i, wc in enumerate(node_config.connection_index)]
        self.merge = self.merge_module[node_config.mo_index]
        self.non_linear = self.nl_module[node_config.nl_index]
