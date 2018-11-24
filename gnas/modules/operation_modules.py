import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class MLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        # >>> m = nn.MLinear(3, 30)
        # >>> input = torch.randn(128, 30)
        # >>> output = m(input)
        # >>> print(output.size())
    """

    def __init__(self, n_in, out_features):
        super(MLinear, self).__init__()
        self.n_in = n_in
        self.out_features = out_features
        self.weight_list = []
        self.bias_list = []
        self.index_list = list(range(n_in))
        for i in range(n_in):
            self.weight_list.append(Parameter(torch.Tensor(out_features, out_features)))
            self.bias_list.append(Parameter(torch.Tensor(out_features)))
            self.register_parameter('weight_' + str(i), self.weight_list[i])
            self.register_parameter('bias_' + str(i), self.bias_list[i])
        self.weight = torch.cat(self.weight_list, -1)
        self.bias = torch.sum(torch.stack(self.bias_list, dim=-1), dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        for w in self.weight_list:
            w.data.uniform_(-stdv, stdv)
        for b in self.bias_list:
            b.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        return F.linear(torch.cat([inputs[i] for i in self.index_list], -1), self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.out_features, self.out_features, self.bias is not None
        )

    def set_input_index(self, index_list):
        self.index_list = index_list
        w_list = [self.weight_list[i] for i in self.index_list]
        b_list = [self.bias_list[i] for i in self.index_list]
        self.weight = torch.cat(w_list, -1)
        self.bias = torch.sum(torch.stack(b_list, dim=-1), dim=-1)


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, inputs):
        return torch.sum(torch.stack(inputs, dim=-1), dim=-1)


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
