import torch
import torch.nn as nn
from gnas.search_space.space_config import AlignmentConfig


class AlignmentModule(nn.Module):
    def __init__(self, ac: AlignmentConfig):
        super(AlignmentModule, self).__init__()
        self.ac = ac
        self._build_operators()

    def _build_operators(self):
        self.weight = self.ac.get_weight_modules()

    def forward(self, inputs):
        return self.weight(inputs)
