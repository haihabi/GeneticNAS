import torch
import torch.nn as nn
import numpy as np
from gnas.search_space.individual import Individual
from gnas.modules.operation_factory import get_module


class SubGraphModule(nn.Module):
    def __init__(self, search_space, config_dict, individual_index=0):
        super(SubGraphModule, self).__init__()
        self.ss = search_space
        self.config_dict = config_dict
        self.individual_index = individual_index
        if self.ss.single_block:
            self.block_modules = [get_module(oc, config_dict) for oc in self.ss.ocl]
        else:
            self.block_modules = [get_module(oc, config_dict) for oc in
                                  self.ss.ocl[individual_index]]
        #
        [self.add_module('Node' + str(i), n) for i, n in enumerate(self.block_modules)]

    def forward(self, *input_list):
        # input list at start is h_n and h_(n-1)
        net = list(input_list)
        for nm in self.block_modules:  # loop over all blocks
            net.append(nm(net))  # call each block in the sub graph
        return net  # output list of all block in the sub graph

    def set_individual(self, individual: Individual):
        if not self.ss.single_block:
            individual = individual.get_individual(self.individual_index)
        si_list = []
        for nc, nm in zip(individual.generate_node_config(), self.block_modules):
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
