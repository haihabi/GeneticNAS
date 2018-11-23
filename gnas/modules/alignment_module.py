import torch
import torch.nn as nn


class AlignmentModule(nn.Module):
    def __init__(self, in_channels, out_channels, alignment_config):
        super(AlignmentModule, self).__init__()
        self.ac = alignment_config
        self.weight = self.ac.get_weight_modules(in_channels, out_channels)
        self.add_module('weight_alignment', self.weight)

    def forward(self, inputs):
        return self.weight(inputs)
