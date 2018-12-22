import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, n_channels, rate):
        super(SEBlock, self).__init__()
        self.se_block = nn.Sequential(nn.Linear(n_channels, int(n_channels / rate)),
                                      nn.ReLU(),
                                      nn.Linear(int(n_channels / rate), n_channels),
                                      nn.Sigmoid())

    def forward(self, x):
        x_gp = torch.mean(torch.mean(x, dim=-1), dim=-1)
        att = self.se_block(x_gp).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return x * att
