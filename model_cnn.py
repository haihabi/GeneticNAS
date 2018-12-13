import torch.nn as nn
import torch.nn.functional as F
import gnas
import torch


class Net(nn.Module):
    def __init__(self, working_device, ss):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = gnas.modules.CnnSearchModule(64, working_device, ss)
        self.block1_1 = gnas.modules.CnnSearchModule(64, working_device, ss)
        self.block1_2 = gnas.modules.CnnSearchModule(64, working_device, ss)
        self.block1_3 = gnas.modules.CnnSearchModule(64, working_device, ss)
        self.block1_4 = gnas.modules.CnnSearchModule(64, working_device, ss)
        self.block1_5 = gnas.modules.CnnSearchModule(64, working_device, ss)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_by_pass = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3_bp = nn.BatchNorm2d(128)
        self.block2 = gnas.modules.CnnSearchModule(128, working_device, ss)
        self.block2_1 = gnas.modules.CnnSearchModule(128, working_device, ss)
        self.block2_2 = gnas.modules.CnnSearchModule(128, working_device, ss)
        self.block2_3 = gnas.modules.CnnSearchModule(128, working_device, ss)
        self.block2_4 = gnas.modules.CnnSearchModule(128, working_device, ss)
        self.block2_5 = gnas.modules.CnnSearchModule(128, working_device, ss)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_by_pass = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn4_pb = nn.BatchNorm2d(256)
        self.block3 = gnas.modules.CnnSearchModule(256, working_device, ss)
        self.block3_1 = gnas.modules.CnnSearchModule(256, working_device, ss)
        self.block3_2 = gnas.modules.CnnSearchModule(256, working_device, ss)
        self.block3_3 = gnas.modules.CnnSearchModule(256, working_device, ss)
        self.block3_4 = gnas.modules.CnnSearchModule(256, working_device, ss)
        self.block3_5 = gnas.modules.CnnSearchModule(256, working_device, ss)
        self.fc1 = nn.Linear(256, 10)
        self.reset_param()
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def reset_param(self):
        for p in self.parameters():
            if len(p.shape) == 4:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x_bypass = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x_bypass)))
        x_b1 = self.block1(x, x_bypass)
        x_b2 = self.block1_1(x_b1, x)
        x_b3 = self.block1_2(x_b2, x_b1)
        x_b4 = self.block1_3(x_b3, x_b2)
        x_b5 = self.block1_4(x_b4, x_b3)
        x = self.block1_5(x_b5, x_b4)

        x = F.relu(self.bn3(self.conv3(x)))
        x_bypass = self.pool(x)
        x = F.relu(self.bn3_bp(self.conv3_by_pass(x_bypass)))

        x_b2_1 = self.block2(x, x_bypass)
        x_b2_2 = self.block2_1(x_b2_1, x)
        x_b2_3 = self.block2_2(x_b2_2, x_b2_1)
        x_b2_4 = self.block2_3(x_b2_3, x_b2_2)
        x_b2_5 = self.block2_4(x_b2_4, x_b2_3)
        x = self.block2_5(x_b2_5, x_b2_4)

        x = F.relu(self.bn4(self.conv4(x)))
        x_bypass = self.pool(x)
        x = F.relu(self.bn4_pb(self.conv4_by_pass(x_bypass)))

        x_b3 = self.block3(x, x_bypass)
        x_b3_1 = self.block3_1(x_b3, x)
        x_b3_2 = self.block3_2(x_b3_1, x_b3)
        x_b3_3 = self.block3_3(x_b3_2, x_b3_1)
        x_b3_4 = self.block3_4(x_b3_3, x_b3_2)
        x = self.block3_5(x_b3_4, x_b3_3)

        x = torch.mean(torch.mean(x, dim=-1), dim=-1)
        return self.fc1(x)

    def set_individual(self, individual):
        self.block1.set_individual(individual)
        self.block1_1.set_individual(individual)
        self.block1_2.set_individual(individual)
        self.block1_3.set_individual(individual)
        self.block1_4.set_individual(individual)
        self.block1_5.set_individual(individual)

        self.block2.set_individual(individual)
        self.block2_1.set_individual(individual)
        self.block2_2.set_individual(individual)
        self.block2_3.set_individual(individual)
        self.block2_4.set_individual(individual)
        self.block2_5.set_individual(individual)

        self.block3.set_individual(individual)
        self.block3_1.set_individual(individual)
        self.block3_2.set_individual(individual)
        self.block3_3.set_individual(individual)
        self.block3_4.set_individual(individual)
        self.block3_5.set_individual(individual)
