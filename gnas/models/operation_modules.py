import torch
import torch.nn as nn


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, inputs):
        outputs = inputs[0]
        for i in inputs[1:]:
            outputs += i
        return outputs


class NC(nn.Module):
    def __init__(self, in_channel=0, out_channels=0):
        super(NC, self).__init__()

    def forward(self, inputs):
        return 0


class Identity(nn.Module):
    def __init__(self, in_channel=0, out_channels=0):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


__module_dict__ = {'Tanh': nn.Tanh,
                   'ReLU': nn.ReLU,
                   'Sigmoid': nn.Sigmoid,
                   'ReLU6': nn.ReLU6,
                   'Add': Add,
                   'Linear': nn.Linear,
                   'NC': NC,
                   'Identity': Identity}


def get_module(module_name):
    m = __module_dict__.get(module_name)
    if m is None:
        raise Exception('Can\'t find module named:' + module_name)
    return m
