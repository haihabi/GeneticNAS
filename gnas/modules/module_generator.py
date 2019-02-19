from torch import nn
from modules.identity import Identity

__nl_dict__ = {'Tanh': nn.Tanh,
               'ReLU': nn.ReLU6,
               'ReLU6': nn.ReLU,
               'SELU': nn.SELU,
               'LeakyReLU': nn.LeakyReLU,
               'Sigmoid': nn.Sigmoid}


def generate_dw_conv(in_channels, out_channels, kernel):
    padding = int((kernel - 1) / 2)
    conv1 = nn.Sequential(nn.ReLU(),
                          nn.Conv2d(in_channels, in_channels, kernel, padding=padding, groups=in_channels, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(),
                          nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                          nn.BatchNorm2d(out_channels))
    conv2 = nn.Sequential(nn.ReLU(),
                          nn.Conv2d(in_channels, in_channels, kernel, padding=padding, groups=in_channels, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(),
                          nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                          nn.BatchNorm2d(out_channels))
    return nn.Sequential(conv1, conv2)


def max_pool3x3(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.MaxPool2d(3, stride=1, padding=1))


def avg_pool3x3(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.AvgPool2d(3, stride=1, padding=1))


def conv3x3(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels))


def dw_conv3x3(in_channels, out_channels):
    return generate_dw_conv(in_channels, out_channels, 3)


def dw_conv1x3(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), groups=out_channels, bias=False),
                         nn.BatchNorm2d(out_channels))


def dw_conv3x1(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, (3, 1), padding=(1, 0), groups=out_channels, bias=False),
                         nn.BatchNorm2d(out_channels))


def conv5x5(in_channels, out_channels):
    return nn.Sequential(nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, 5, padding=2, bias=False),
                         nn.BatchNorm2d(out_channels))


def dw_conv5x5(in_channels, out_channels):
    return generate_dw_conv(in_channels, out_channels, 5)


def identity(in_channels, out_channels):
    return Identity()


__op_dict__ = {'Conv3x3': conv3x3,
               'Dw3x3': dw_conv3x3,
               'Dw3x1': dw_conv3x1,
               'Dw1x3': dw_conv1x3,
               'Conv5x5': conv5x5,
               'Dw5x5': dw_conv5x5,
               'Identity': identity,
               'Max3x3': max_pool3x3,
               'Avg3x3': avg_pool3x3, }


def generate_non_linear(non_linear_list):
    return [__nl_dict__.get(nl)() for nl in non_linear_list]


def generate_op(op_list, in_channels, out_channels):
    return [__op_dict__.get(nl)(in_channels, out_channels) for nl in op_list]
