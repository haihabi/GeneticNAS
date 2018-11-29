from torch import nn

__nl_dict__ = {'Tanh': nn.Tanh,
               'ReLU': nn.ReLU6,
               'ReLU6': nn.ReLU,
               'Sigmoid': nn.Sigmoid}


def generate_non_linear(non_linear_list):
    return [__nl_dict__.get(nl)() for nl in non_linear_list]
