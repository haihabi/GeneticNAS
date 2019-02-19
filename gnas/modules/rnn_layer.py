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
        self.config_dict = {'in_channels': self.in_channels,
                            'n_channels': self.n_channels}
        self.sub_graph_module = SubGraphModule(ss, self.config_dict)

        self.reset_parameters()

    def forward(self, inputs_tensor, state_tensor):
        # input size [Time step,Batch,features]

        state = state_tensor[0, :, :]
        outputs = []

        for i in torch.split(inputs_tensor, split_size_or_sections=1, dim=0):  # Loop over time steps
            output, state = self.cell(i, state)
            # state_norm = state.norm(dim=-1)
            # max_norm = 25.0
            # if torch.any(state_norm > max_norm).item():
            #     clip_select = state_norm > max_norm
            #     clip_norms = state_norm[clip_select]
            #
            #     mask = torch.ones(state.size(), device=self.working_device)
            #     normalizer = max_norm / clip_norms
            #     mask[clip_select, :] = normalizer.unsqueeze(dim=-1)
            #     mask = mask.detach()
            #     state *= mask
                # print(np.max(state.norm(dim=-1).detach().cpu().numpy()))
                # print("Max Norm pass")
            # state = state / state.norm(dim=-1)
            outputs.append(output)
        output = torch.stack(outputs, dim=0)

        return output, state.unsqueeze(dim=0)

    def cell(self, x, state):
        net = self.sub_graph_module(x.squeeze(dim=0), state)
        output, state = torch.mean(torch.stack([net[i] for i in self.sub_graph_module.avg_index], dim=-1), dim=-1), net[
            -1]
        return output, output

    def set_individual(self, individual: Individual):
        self.sub_graph_module.set_individual(individual)

    def init_state(self, batch_size=1):  # model init state
        weight = next(self.parameters())
        return weight.new_zeros(1, batch_size, self.n_channels)

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def reset_parameters(self):
        init_range = 0.025
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
