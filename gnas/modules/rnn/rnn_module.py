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
        # input size [Time step,Batch,features]
        state = state_tensor[0, :, :]
        outputs = []

        for i in torch.split(inputs_tensor, split_size_or_sections=1, dim=0):  # Loop over time steps

            state = self.sub_graph_module(i.squeeze(dim=0), state)

            outputs.append(state)
        output = torch.stack(outputs, dim=0)

        return output, output

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def init_state(self, batch_size=1):  # model init state
        weight = next(self.parameters())
        return weight.new_zeros(1, batch_size, self.n_channels)
        # return torch.zeros(1, batch_size, self.n_channels, device=self.working_device)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param
