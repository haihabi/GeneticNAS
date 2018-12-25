import numpy as np
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig, CnnNodeConfig


def get_enas_rnn_search_space(n_nodes) -> SearchSpace:
    nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
    node_config_list = [RnnInputNodeConfig(2, [0, 1], nll)]
    for i in range(n_nodes - 1):
        node_config_list.append(RnnNodeConfig(3 + i, np.linspace(2, 2 + i, 1 + i).astype('int'), nll))
    return SearchSpace(node_config_list)


def get_enas_cnn_search_space(n_nodes) -> SearchSpace:
    nll = ['SELU', 'ReLU', 'ReLU6', 'LeakyReLU']
    op = ['Dw3x3', 'Identity', 'Dw5x5', 'Avg3x3', 'Max3x3']
    node_config_list = [CnnNodeConfig(2, [0, 1], nll, op)]
    for i in range(n_nodes - 1):
        node_config_list.append(CnnNodeConfig(3 + i, np.linspace(0, 2 + i, 3 + i).astype('int'), nll, op))
    return SearchSpace(node_config_list)
