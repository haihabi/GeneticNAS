import torch.nn as nn
from gnas.modules.module_generator import generate_non_linear, generate_op
from modules.weight_drop import WeightDrop
from modules.drop_module import DropModule


class RnnInputNodeModule(nn.Module):
    def __init__(self, node_config, config_dict):
        super(RnnInputNodeModule, self).__init__()
        if node_config.get_n_inputs() != 2: raise Exception('aaa')
        self.nc = node_config
        dropout = 0.5  # TODO:change to input config
        self.in_channels = config_dict.get('in_channels')
        self.n_channels = config_dict.get('n_channels')
        self.nl_module = generate_non_linear(self.nc.non_linear_list)

        self.x_linear_list = [nn.Linear(self.in_channels, self.n_channels) for _ in range(2)]
        [self.add_module('c_linear' + str(i), m) for i, m in enumerate(self.x_linear_list)]
        self.h_linear_list = [WeightDrop(nn.Linear(self.n_channels, self.n_channels), ['weight'], dropout, True) for _
                              in
                              range(2)]
        [self.add_module('h_linear' + str(i), m) for i, m in enumerate(self.h_linear_list)]
        self.sigmoid = nn.Sigmoid()

        self.non_linear = None
        self.node_config = None

    def forward(self, inputs):
        c = self.sigmoid(self.x_linear_list[0](inputs[0]) + self.h_linear_list[0](inputs[1]))
        output = c * self.non_linear(self.x_linear_list[1](inputs[0]) + self.h_linear_list[1](inputs[1])) + (1 - c) * \
                 inputs[1]
        return output

    def set_current_node_config(self, current_config):
        nl_index = current_config
        self.cc = current_config
        self.non_linear = self.nl_module[nl_index]


class RnnNodeModule(nn.Module):
    def __init__(self, node_config, config_dict):
        super(RnnNodeModule, self).__init__()
        self.nc = node_config
        if node_config.get_n_inputs() < 1: raise Exception('aaa')

        self.n_channels = config_dict.get('n_channels')
        self.nl_module = generate_non_linear(self.nc.non_linear_list)
        # self.bn = nn.BatchNorm1d(self.n_channels)

        self.x_linear_list = [nn.Linear(self.n_channels, self.n_channels) for _ in range(node_config.get_n_inputs())]
        [self.add_module('c_linear' + str(i), m) for i, m in enumerate(self.x_linear_list)]
        self.h_linear_list = [nn.Linear(self.n_channels, self.n_channels) for _ in range(node_config.get_n_inputs())]
        [self.add_module('h_linear' + str(i), m) for i, m in enumerate(self.h_linear_list)]
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(self.n_channels)
        self.non_linear = None
        self.node_config = None

    def forward(self, inputs):
        x = inputs[self.select_index]
        c = self.sigmoid(self.x_linear(x))
        return c * self.non_linear(self.h_linear(x)) + (1 - c) * x

    def set_current_node_config(self, current_config):
        self.select_index, op_index, nl_index = current_config
        self.cc = current_config
        self.non_linear = self.nl_module[nl_index]
        self.x_linear = self.x_linear_list[op_index]
        self.h_linear = self.h_linear_list[op_index]


class ConvNodeModule(nn.Module):
    def __init__(self, node_config, config_dict):
        super(ConvNodeModule, self).__init__()
        self.nc = node_config

        self.n_channels = config_dict.get('n_channels')
        self.conv_module = []
        for j in range(node_config.get_n_inputs()):
            op_list = [DropModule(op, node_config.drop_path_control) for op in
                       generate_op(self.nc.op_list, self.n_channels, self.n_channels)]
            self.conv_module.append(op_list)
            [self.add_module('conv_op_' + str(i) + '_in_' + str(j), m) for i, m in enumerate(self.conv_module[-1])]

        self.non_linear_a = None
        self.non_linear_b = None
        self.input_a = None
        self.input_b = None
        self.cc = None
        self.op_a = None
        self.op_b = None

    def forward(self, inputs):
        net_a = inputs[self.input_a]
        net_b = inputs[self.input_b]
        return self.op_a(net_a) + self.op_b(net_b)

    def set_current_node_config(self, current_config):
        input_a, input_b, input_index_a, input_index_b, op_a, op_b = current_config
        self.select_index = [input_a, input_b]
        self.cc = current_config
        self.input_a = input_a
        self.input_b = input_b
        self.op_a = self.conv_module[input_index_a][op_a]
        self.op_b = self.conv_module[input_index_b][op_b]
        #### set grad false
        for p in self.parameters():
            p.requires_grad = False
        #### set grad true
        for p in self.op_b.parameters():
            p.requires_grad = True
        for p in self.op_a.parameters():
            p.requires_grad = True
