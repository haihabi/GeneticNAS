import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from gnas.search_space.individual import Individual
from gnas.modules.sub_graph_module import SubGraphModule
from torch.nn import functional as F
from modules.se_block import SEBlock
from modules.identity import Identity


class CnnSearchModule(nn.Module):
    def __init__(self, n_channels, ss, individual_index=0, se_block=True):
        super(CnnSearchModule, self).__init__()

        self.ss = ss
        self.n_channels = n_channels
        self.config_dict = {'n_channels': n_channels}
        self.sub_graph_module = SubGraphModule(ss, self.config_dict,
                                               individual_index=individual_index)
        if ss.single_block:
            self.n_inputs = len(ss.ocl[0].inputs)
        else:
            self.n_inputs = len(ss.ocl[individual_index][0].inputs)
        if se_block:
            self.se_block = SEBlock(n_channels, 8)
        else:
            self.se_block = Identity()

        self.bn = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU()
        if self.ss.single_block:
            self.weights = [Parameter(torch.Tensor(n_channels, n_channels, 1, 1)) for _ in range(len(ss.ocl))]
        else:
            self.weights = [Parameter(torch.Tensor(n_channels, n_channels, 1, 1)) for _ in
                            range(len(ss.ocl[individual_index]))]
        [self.register_parameter('w_' + str(i), w) for i, w in enumerate(self.weights)]
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.n_channels * len(self.weights)
        stdv = 1. / math.sqrt(n)
        for w in self.weights:
            w.data.uniform_(-stdv, stdv)

    def forward(self, inputs_tensor, bypass_input):
        if self.n_inputs == 1:
            net = self.sub_graph_module(inputs_tensor)
        elif self.n_inputs == 2:
            net = self.sub_graph_module(inputs_tensor, bypass_input)

        net = torch.cat([net[i] for i in self.sub_graph_module.avg_index if i > 1], dim=1)
        w = torch.cat([self.weights[i - 2] for i in self.sub_graph_module.avg_index if i > 1], dim=1)
        net = self.bn(F.conv2d(self.relu(net), w, self.bias, 1, 0, 1, 1))
        return self.se_block(net) + inputs_tensor

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param
