import torch
import torch.nn as nn
import numpy as np
from gnas.search_space.individual import Individual
from gnas.modules.operation_factory import get_module
from cnn_utils import DropPath


class SubGraphModule(nn.Module):
    def __init__(self, search_space, config_dict, individual_index=0):
        super(SubGraphModule, self).__init__()
        self.ss = search_space
        self.config_dict = config_dict
        self.individual_index = individual_index
        if self.ss.single_block:
            self.node_modules = [get_module(oc, config_dict)for oc in self.ss.ocl]
        else:
            self.node_modules = [get_module(oc, config_dict)for oc in
                                 self.ss.ocl[individual_index]]
        # print(len(self.node_modules))
        [self.add_module('Node' + str(i), n) for i, n in enumerate(self.node_modules)]

    def forward(self, *input_list):
        net = list(input_list)
        for nm in self.node_modules:
            net.append(nm(net))
        return net

    def set_individual(self, individual: Individual):
        if not self.ss.single_block:
            individual = individual.get_individual(self.individual_index)
        si_list = []
        for nc, nm in zip(individual.generate_node_config(), self.node_modules):
            nm.set_current_node_config(nc)
            if nm.__dict__.get('select_index') is not None:
                si_list.append(nm.select_index)

        current_node_list = np.unique(si_list)
        if self.ss.single_block:
            self.avg_index = np.asarray([n.node_id for n in self.ss.ocl if n.node_id not in current_node_list]).astype(
                'int')
        else:
            self.avg_index = np.asarray(
                [n.node_id for n in self.ss.ocl[self.individual_index] if n.node_id not in current_node_list]).astype(
                'int')
