import torch
import numpy as np
import torch.nn as nn
from gnas.modules.operation_modules import MLinear


class NodeModule(nn.Module):
    def __init__(self, max_inputs: int, n_channels: int, operation_config):
        super(NodeModule, self).__init__()
        self.max_inputs = max_inputs
        self.oc = operation_config
        self.n_channels = n_channels
        self.nl_module = self.oc.get_non_linear_modules()
        self.m_linear = MLinear(max_inputs, n_channels)
        self.add_module('m_linear', self.m_linear)
        if not operation_config.is_linear_mode():
            raise Exception('This node only work in linear mode')
        # else:
        #     self.merge_module = self.oc.get_merge_op_modules()
        #     self.weight_modules = self.oc.get_weight_modules(max_inputs, n_channels)
        #
        #     [[self.add_module(str(i) + str(j), op) for j, op in enumerate(input_w)] for i, input_w in
        #     enumerate(self.weight_modules)]

        # self.weight = None
        # self.merge = None
        self.non_linear = None
        self.node_config = None
        # self.index_selection = None

    def forward(self, inputs):
        x = self.m_linear(inputs)
        return self.non_linear(x)

    def set_current_node_config(self, node_config):
        self.node_config = node_config
        index_selection = np.where(np.asarray(node_config.connection_index) > 0)[0]
        self.m_linear.set_input_index(index_selection)
        self.non_linear = self.nl_module[node_config.nl_index]
