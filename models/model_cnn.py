import torch.nn as nn
import torch.nn.functional as F
import gnas
import torch


class RepeatBlock(nn.Module):
    def __init__(self, n_blocks, n_channels, ss):
        super(RepeatBlock, self).__init__()
        self.block_list = [gnas.modules.CnnSearchModule(n_channels, ss) for i in range(n_blocks)]
        [self.add_module('block_' + str(i), n) for i, n in enumerate(self.block_list)]

    def forward(self, x):
        x_prev = x
        for b in self.block_list:
            x_new = b(x, x_prev)
            x_prev = x
            x = x_new
        return x

    def set_individual(self, individual):
        [b.set_individual(individual) for b in self.block_list]


class Net(nn.Module):
    def __init__(self, n_blocks, n_channels, n_classes, dropout, ss):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, n_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.block_repeat_1 = RepeatBlock(1, n_channels, ss)

        self.conv2 = nn.Conv2d(n_channels, 2 * n_channels, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * n_channels)
        self.block_repeat_2 = RepeatBlock(n_blocks, 2 * n_channels, ss)

        self.conv3 = nn.Conv2d(2 * n_channels, 4 * n_channels, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * n_channels)
        self.block_repeat_3 = RepeatBlock(n_blocks, 4 * n_channels, ss)

        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)
        self.fc1 = nn.Sequential(nn.ReLU(),
                                 nn.Linear(4 * n_channels, n_classes))
        self.reset_param()

    def reset_param(self):
        for p in self.parameters():
            if len(p.shape) == 4:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.block_repeat_1(x)
        x = self.bn2(self.conv2(self.relu(x)))
        x = self.block_repeat_2(x)
        x = self.bn3(self.conv3(self.relu(x)))
        x = self.block_repeat_3(x)

        x = torch.mean(torch.mean(x, dim=-1), dim=-1)
        return self.fc1(self.dp(x))

    def set_individual(self, individual):
        self.block_repeat_1.set_individual(individual)
        self.block_repeat_2.set_individual(individual)
        self.block_repeat_3.set_individual(individual)
