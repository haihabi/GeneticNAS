import numpy as np
from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig, CnnNodeConfig
from gnas.search_space.search_space import SearchSpace
from modules.drop_module import DropModuleControl
import gnas


def generate_ss():
    nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
    node_config_list = [RnnInputNodeConfig(2, [0, 1], nll)]
    for i in range(12):
        node_config_list.append(RnnNodeConfig(3 + i, list(np.linspace(2, 2 + i, 1 + i).astype('int')), nll))
    ss = SearchSpace(node_config_list)
    return ss


def generate_ss_cnn():
    nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
    op = ['Conv3x3', 'Dw3x3', 'Conv5x5', 'Dw5x5']
    dp_control = DropModuleControl(1)
    node_config_list = [CnnNodeConfig(2, [0, 1], op, dp_control)]
    for i in range(3):
        node_config_list.append(CnnNodeConfig(3 + i, list(np.linspace(0, 2 + i, 3 + i).astype('int')), op, dp_control))
    ss = SearchSpace(node_config_list)
    return ss
