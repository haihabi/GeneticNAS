import torch
import torch.nn as nn
from gnas.modules.operation_modules import get_module


class AlignmentModule(nn.Module):
    def __init__(self, in_channels, out_channels, alignment_config):
        super(AlignmentModule, self).__init__()
        self.ac = alignment_config
        self.weight = get_module(self.ac.alignment_operator)(in_channels, out_channels)
        self.add_module('weight_alignment', self.weight)

    def forward(self, inputs):
        return self.weight(inputs)
