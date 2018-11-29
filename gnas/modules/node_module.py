import torch
import numpy as np
import torch.nn as nn
from gnas.modules.module_generator import generate_non_linear


class RnnInputNodeModule(nn.Module):
    def __init__(self, node_config):
        super(RnnInputNodeModule, self).__init__()
        if node_config.get_n_inputs() != 2: raise Exception('aaa')
        self.nc = node_config

        self.in_channels = self.nc.x_size
        self.n_channels = self.nc.recurrent_size
        self.nl_module = generate_non_linear(self.nc.non_linear_list)

        self.x_linear_list = [nn.Linear(self.in_channels, self.n_channels) for _ in range(2)]
        [self.add_module('c_linear' + str(i), m) for i, m in enumerate(self.x_linear_list)]
        self.h_linear_list = [nn.Linear(self.n_channels, self.n_channels) for _ in range(2)]
        [self.add_module('h_linear' + str(i), m) for i, m in enumerate(self.h_linear_list)]
        self.sigmoid = nn.Sigmoid()

        self.non_linear = None
        self.node_config = None

    def forward(self, inputs):
        c = self.sigmoid(self.x_linear_list[0](inputs[0]) + self.h_linear_list[0](inputs[1]))
        return c * self.non_linear(self.x_linear_list[1](inputs[0]) + self.h_linear_list[1](inputs[1])) + (1 - c) * \
               inputs[1]

    def set_current_node_config(self, current_config):
        nl_index = self.nc.parse_config(current_config)
        self.cc = current_config
        self.non_linear = self.nl_module[nl_index]


class RnnNodeModule(nn.Module):
    def __init__(self, node_config):
        super(RnnNodeModule, self).__init__()
        self.nc = node_config
        if node_config.get_n_inputs() < 1: raise Exception('aaa')

        self.n_channels = self.nc.recurrent_size
        self.nl_module = generate_non_linear(self.nc.non_linear_list)

        self.x_linear_list = [nn.Linear(self.n_channels, self.n_channels) for _ in range(node_config.get_n_inputs())]
        [self.add_module('c_linear' + str(i), m) for i, m in enumerate(self.x_linear_list)]
        self.h_linear_list = [nn.Linear(self.n_channels, self.n_channels) for _ in range(node_config.get_n_inputs())]
        [self.add_module('h_linear' + str(i), m) for i, m in enumerate(self.h_linear_list)]
        self.sigmoid = nn.Sigmoid()

        self.non_linear = None
        self.node_config = None

    def forward(self, inputs):
        x = inputs[self.select_index]
        c = self.sigmoid(self.x_linear(x))
        return c * self.non_linear(self.h_linear(x)) + (1 - c) * x

    def set_current_node_config(self, current_config):
        self.select_index, op_index, nl_index = self.nc.parse_config(current_config)
        self.cc = current_config
        self.non_linear = self.nl_module[nl_index]
        self.x_linear = self.x_linear_list[op_index]
        self.h_linear = self.h_linear_list[op_index]
