import torch
import torch.nn as nn
from gnas.search_space.individual import Individual
from gnas.modules.sub_graph_module import SubGraphModule


class RnnSearchModule(nn.Module):
    def __init__(self, in_channels, n_channels, working_device, ss):
        super(RnnSearchModule, self).__init__()

        self.ss = ss
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.working_device = working_device

        self.sub_graph_module = SubGraphModule(ss)

        self.bn = nn.BatchNorm1d(n_channels, affine=False)

    def forward(self, inputs_tensor, state_tensor):
        # input size [Time step,Batch,features]

        state = state_tensor[0, :, :]
        outputs = []

        for i in torch.split(inputs_tensor, split_size_or_sections=1, dim=0):  # Loop over time steps
            state = self.cell(i, state)

            outputs.append(state)
        output = torch.stack(outputs, dim=0)

        return output, output

    def cell(self, x, state):
        state = self.sub_graph_module(x.squeeze(dim=0), state)
        return self.bn(state)

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def init_state(self, batch_size=1):  # model init state
        weight = next(self.parameters())
        return weight.new_zeros(1, batch_size, self.n_channels)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param
