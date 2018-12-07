import numpy as np
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig


def get_enas_rnn_search_space(n_nodes) -> SearchSpace:
    nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
    node_config_list = [RnnInputNodeConfig(2, [0, 1], nll)]
    for i in range(n_nodes):
        node_config_list.append(RnnNodeConfig(3 + i, np.linspace(2, 2 + i, 1 + i).astype('int'), nll))
    return SearchSpace(node_config_list)
