import torch
import torch.nn as nn
from gnas.search_space.individual import Individual
from gnas.search_space.search_space import SearchSpace
from gnas.models.sub_graph_module import SubGraphModule


class RnnModule(nn.Module):
    def __init__(self, ss: SearchSpace):
        super(RnnModule, self).__init__()
        if ss.n_inputs != 2: raise Exception('')
        if ss.n_outputs != 2: raise Exception('')
        self.ss = ss
        self.sub_graph_module = SubGraphModule(ss)

    def forward(self, inputs_tensor, state_tensor):
        # input size [Batch,Time step,features]
        state = state_tensor[0, :, :]
        outputs = []
        for i in range(inputs_tensor.shape[1]):  # Loop over time steps
            state = self.sub_graph_module(inputs_tensor[:, i, :], state)
            outputs.append(state)
        return outputs

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def init_state(self, batch_size=1):  # model init state
        return torch.zeros(1, batch_size, self.ss.get_n_channels(), device=self.config.device)
