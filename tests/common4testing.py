import numpy as np
from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig
from gnas.search_space.search_space import SearchSpace


def generate_ss():
    nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
    node_config_list = [RnnInputNodeConfig(2, [0, 1], 32, 128, nll)]
    for i in range(12):
        node_config_list.append(RnnNodeConfig(3 + i, np.linspace(2, 2 + i, 1 + i).astype('int'), 128, nll))
    ss = SearchSpace(node_config_list)
    return ss
