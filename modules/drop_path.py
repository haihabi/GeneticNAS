import torch
import torch.cuda
import torch.nn as nn

from random import random
from torch.autograd import Variable


# Currently there is a risk of dropping all paths...
# We should create a version that take all paths into account to make sure one stays alive
# But then keep_prob is meaningless and we have to copute/keep track of the conditional


class DropPathControl(object):
    def __init__(self, keep_prob=0.9):
        self.keep_prob = keep_prob
        self.status = False

    def enable(self):
        self.status = True


class DropPath(nn.Module):
    def __init__(self, module, drop_control: DropPathControl):
        super(DropPath, self).__init__()
        self.module = module
        self.keep_prob = drop_control.keep_prob
        self.shape = None
        self.training = True
        self.drop_control = drop_control
        self.dtype = torch.FloatTensor

    def forward(self, *input):
        if self.training and self.drop_control.status:
            # If we don't now the shape we run the forward path once and store the output shape
            if self.shape is None:
                temp = self.module(*input)
                self.shape = temp.size()
                if temp.data.is_cuda:
                    self.dtype = torch.cuda.FloatTensor
                del temp
            p = random()
            if p <= self.keep_prob:
                return self.module(*input) / self.keep_prob  # Inverted scaling
            else:
                return Variable(self.dtype(torch.Size([input[0].shape[0], *list(self.shape[1:])])).zero_())

        else:
            return self.module(*input)
