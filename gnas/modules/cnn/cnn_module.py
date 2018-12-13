import torch
import torch.nn as nn
from gnas.search_space.individual import Individual
from gnas.modules.sub_graph_module import SubGraphModule


class CnnSearchModule(nn.Module):
    def __init__(self, n_channels, working_device, ss):
        super(CnnSearchModule, self).__init__()

        self.ss = ss
        # self.in_channels = in_channels
        self.n_channels = n_channels
        self.working_device = working_device
        self.config_dict = {'n_channels': n_channels}
        self.sub_graph_module = SubGraphModule(ss, self.config_dict)

    def forward(self, inputs_tensor, bypass_input):
        net = self.sub_graph_module(inputs_tensor, bypass_input)
        output = torch.mean(torch.stack([net[i] for i in self.sub_graph_module.avg_index], dim=-1), dim=-1)
        return output

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param
