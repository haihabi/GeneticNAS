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
        self.config_dict = {'channels': n_channels}
        self.sub_graph_module1 = SubGraphModule(ss, self.config_dic)

        self.reset_parameters()

    def forward(self, inputs_tensor, bypass_input):
        # input size [Time step,Batch,features]
        return self.sub_graph_module(inputs_tensor, bypass_input)

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param
