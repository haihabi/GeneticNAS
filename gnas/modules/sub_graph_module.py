import torch
import torch.nn as nn
from gnas.search_space.individual import Individual
from gnas.modules.alignment_module import AlignmentModule
from gnas.modules.node_module import NodeModule


class SubGraphModule(nn.Module):
    def __init__(self, input_size, n_channels, search_space):
        super(SubGraphModule, self).__init__()
        self.ss = search_space
        self.node_modules = [
            NodeModule(min(i + search_space.n_inputs, search_space.n_inputs + search_space.n_nodes), n_channels,
                       search_space.oc) for i
            in
            range(search_space.n_outputs + search_space.n_nodes)]
        [self.add_module('node' + str(i), n) for i, n in enumerate(self.node_modules)]
        self.alignment_modules = [AlignmentModule(input_size[i], n_channels, search_space.ac) for i in
                                  range(search_space.n_inputs)]
        [self.add_module('alignment' + str(i), n) for i, n in enumerate(self.alignment_modules)]

    def forward(self, *inputs):
        net = [a(i) for i, a in zip(inputs, self.alignment_modules)]
        for nm in self.node_modules:
            net.append(nm(net))
        outputs = net[-self.ss.n_outputs:]
        if self.ss.n_outputs == 1:
            return outputs[0]
        else:
            raise NotImplemented

    def set_individual(self, individual: Individual):
        for nc, nm in zip(individual.generate_node_config(), self.node_modules):
            nm.set_current_node_config(nc)
