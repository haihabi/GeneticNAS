import torch
import torch.nn as nn
from gnas.search_space.individual import Individual
from gnas.modules.sub_graph_module import SubGraphModule


class RnnSearchModule(nn.Module):
    def __init__(self, in_channels, n_channels, working_device, ss):
        super(RnnSearchModule, self).__init__()
        if ss.n_inputs != 2: raise Exception('')
        if ss.n_outputs != 1: raise Exception('')
        self.ss = ss
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.working_device = working_device
        self.sub_graph_module = SubGraphModule([in_channels, n_channels], n_channels, ss)

    def forward(self, inputs_tensor, state_tensor):
        # input size [Batch,Time step,features]
        state = state_tensor[0, :, :]
        outputs = []
        for i in torch.split(inputs_tensor, split_size_or_sections=1, dim=1):  # Loop over time steps
            state = self.sub_graph_module(i.squeeze(dim=1), state)
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def init_state(self, batch_size=1):  # model init state
        return torch.zeros(1, batch_size, self.n_channels, device=self.working_device)
