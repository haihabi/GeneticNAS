import torch.nn as nn
import gnas
import torch


class RepeatBlock(nn.Module):
    def __init__(self, n_blocks, n_channels, ss, individual_index=0, first_block=None):
        super(RepeatBlock, self).__init__()
        if first_block is None: first_block = individual_index
        self.block_list = [gnas.modules.CnnSearchModule(n_channels, ss,
                                                        individual_index=first_block if i == 0 else individual_index)
                           for i in
                           range(n_blocks)]
        [self.add_module('block_' + str(i), n) for i, n in enumerate(self.block_list)]

    def forward(self, x, x_prev):
        for b in self.block_list:
            x_new = b(x, x_prev)
            x_prev = x
            x = x_new
        return x, x_prev

    def set_individual(self, individual):
        [b.set_individual(individual) for b in self.block_list]


class Net(nn.Module):
    def __init__(self, n_blocks, n_channels, n_classes, dropout, ss, aux=False):
        n_block_types = len(ss.ocl)
        normal_block_index = 0
        reduce_block_index = 0
        first_block_index = 0
        if n_block_types >= 2:
            normal_block_index = 1
            first_block_index = 1
        if n_block_types == 3: first_block_index = 2
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, n_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)

        self.block_1 = RepeatBlock(n_blocks, n_channels, ss,
                                   individual_index=normal_block_index, first_block=first_block_index)

        self.avg = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(n_channels, 2 * n_channels, 1, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * n_channels)

        self.conv2_prev = nn.Conv2d(n_channels, 2 * n_channels, 1, stride=1, padding=1, bias=False)
        self.bn2_prev = nn.BatchNorm2d(2 * n_channels)
        # self

        self.block_2_reduce = gnas.modules.CnnSearchModule(2 * n_channels, ss, individual_index=reduce_block_index)
        self.block_2 = RepeatBlock(n_blocks, 2 * n_channels, ss,
                                   individual_index=normal_block_index)

        self.conv3 = nn.Conv2d(2 * n_channels, 4 * n_channels, 1, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * n_channels)

        self.conv3_prev = nn.Conv2d(2 * n_channels, 4 * n_channels, 1, stride=1, padding=1, bias=False)
        self.bn3_prev = nn.BatchNorm2d(4 * n_channels)

        self.block_3_reduce = gnas.modules.CnnSearchModule(4 * n_channels, ss, individual_index=reduce_block_index)
        self.block_3 = RepeatBlock(n_blocks, 4 * n_channels, ss,
                                   individual_index=normal_block_index)

        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=dropout)
        self.fc1 = nn.Sequential(nn.ReLU(),
                                 nn.Linear(4 * n_channels, n_classes))
        self.aux = aux
        if aux:
            self.fc2 = nn.Sequential(nn.ReLU(),
                                     nn.Linear(2 * n_channels, n_classes))
        self.reset_param()

    def reset_param(self):
        for p in self.parameters():
            if len(p.shape) == 4:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        x_prev = self.bn1(self.conv1(x))

        x, x_prev = self.block_1(x_prev, x_prev)

        # reduce dim
        x = self.bn2(self.conv2(self.avg(x)))
        x_prev = self.bn2_prev(self.conv2_prev(self.avg(x_prev)))

        x, x_prev = self.block_2(self.block_2_reduce(x, x_prev), x)


        if self.aux: x2 = torch.mean(torch.mean(x, dim=-1), dim=-1)


        x = self.bn3(self.conv3(self.avg(x)))
        x_prev = self.bn3_prev(self.conv3_prev(self.avg(x_prev)))

        x, x_prev = self.block_3(self.block_3_reduce(x, x_prev), x)


        # Global pooling
        x = torch.mean(torch.mean(x, dim=-1), dim=-1)
        if self.aux:
            return [self.fc1(self.dp(x)), self.fc2(self.dp(x2))]
        else:
            return [self.fc1(self.dp(x))]

    def set_individual(self, individual):
        self.block_1.set_individual(individual)
        self.block_2.set_individual(individual)
        self.block_2_reduce.set_individual(individual)
        self.block_3_reduce.set_individual(individual)
        self.block_3.set_individual(individual)
