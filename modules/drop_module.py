import torch
import torch.cuda
import torch.nn as nn
from random import random
from torch.autograd import Variable


class DropModuleControl(object):
    def __init__(self, drop_prob=0.9):
        self.drop_prob = drop_prob
        self.status = False

    def enable(self):
        self.status = True


class DropModule(nn.Module):
    def __init__(self, module, drop_control: DropModuleControl):
        super(DropModule, self).__init__()
        self.module = module
        self.shape = None
        self.drop_control = drop_control
        self.tensor_init = torch.FloatTensor

    def update_tensor_shape(self, *input):
        if self.shape is None:
            output_tensor = self.module(*input)
            self.shape = output_tensor.size()  # fetch tensor shape
            if output_tensor.data.is_cuda: self.tensor_init = torch.cuda.FloatTensor

    def forward(self, *input):
        self.update_tensor_shape(*input)
        if self.training and self.drop_control.status:
            if random() <= self.drop_control.drop_prob:  # forward module tensor
                return self.module(*input) / self.drop_control.drop_prob  # Apply scaling
            else:  # forward zero tensor
                return Variable(self.tensor_init(torch.Size([input[0].shape[0], *list(self.shape[1:])])).zero_())
        else:  # Inference
            return self.module(*input)
