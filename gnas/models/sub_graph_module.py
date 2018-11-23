import torch
import torch.nn as nn
from gnas.search_space.individual import Individual
from gnas.search_space.search_space import SearchSpace


class SubGraphModule(nn.Module):
    def __init__(self, ss: SearchSpace):
        super(SubGraphModule, self).__init__()
        self.ss = ss
        self.node_modules = ss.generate_nodes_modules()
        self.alignment_modules = ss.generate_alignment_modules()

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
