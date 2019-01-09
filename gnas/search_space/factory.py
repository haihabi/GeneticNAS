import numpy as np
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig, CnnNodeConfig
from enum import Enum

class SearchSpaceType(Enum):
    CNNSingleCell = 0
    CNNDualCell = 1
    CNNTripleCell = 2

def _two_input_cell(n_nodes,drop_path):
    op = ['Dw3x3', 'Identity', 'Dw5x5', 'Avg3x3', 'Max3x3']
    node_config_list = [CnnNodeConfig(2, [0, 1], op,drop_path=drop_path)]
    for i in range(n_nodes - 1):
        node_config_list.append(CnnNodeConfig(3 + i, np.linspace(0, 2 + i, 3 + i).astype('int'), op,drop_path=drop_path))
    return node_config_list

def _one_input_cell(n_nodes,drop_path):
    op = ['Dw3x3', 'Identity', 'Dw5x5', 'Avg3x3', 'Max3x3']
    node_config_list = [CnnNodeConfig(1, [0], op,drop_path=drop_path)]
    for i in range(n_nodes - 1):
        node_config_list.append(CnnNodeConfig(2 + i, np.linspace(0, 2 + i, 3 + i).astype('int'), op,drop_path=drop_path))
    return node_config_list



def get_enas_cnn_search_space(n_nodes,drop_path,n_cell_type:SearchSpaceType) -> SearchSpace:
    node_config_list_a = _two_input_cell(n_nodes,drop_path)
    if n_cell_type==SearchSpaceType.CNNSingleCell:
        return SearchSpace(node_config_list_a)
    elif n_cell_type==SearchSpaceType.CNNDualCell:
        node_config_list_b = _two_input_cell(n_nodes,drop_path)
        return SearchSpace([node_config_list_a, node_config_list_b], single_block=False)
    elif n_cell_type == SearchSpaceType.CNNTripleCell:
        node_config_list_b = _two_input_cell(n_nodes, drop_path)
        node_config_list_c=_one_input_cell(n_nodes,drop_path)
        return SearchSpace([node_config_list_a, node_config_list_b,node_config_list_c], single_block=False)

# def get_enas_cnn_search_space_dual(n_nodes,drop_path) -> SearchSpace:
#     nll = ['SELU', 'ReLU', 'ReLU6', 'LeakyReLU']
#     op = ['Dw3x3', 'Identity', 'Dw5x5', 'Avg3x3', 'Max3x3']
#     node_config_list_a = [CnnNodeConfig(2, [0, 1], nll, op,drop_path=drop_path)]
#     for i in range(n_nodes - 1):
#         node_config_list_a.append(CnnNodeConfig(3 + i, np.linspace(0, 2 + i, 3 + i).astype('int'), nll, op,drop_path=drop_path))
#
#     nll = ['SELU', 'ReLU', 'ReLU6', 'LeakyReLU']
#     op = ['Dw3x3', 'Identity', 'Dw5x5', 'Avg3x3', 'Max3x3']
#     node_config_list_b = [CnnNodeConfig(2, [0, 1], nll, op,drop_path=drop_path)]
#     for i in range(n_nodes - 1):
#         node_config_list_b.append(CnnNodeConfig(3 + i, np.linspace(0, 2 + i, 3 + i).astype('int'), nll, op,drop_path=drop_path))
#     return SearchSpace([node_config_list_a,node_config_list_b],single_block=False)



# def get_enas_rnn_search_space(n_nodes) -> SearchSpace:
#     nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
#     node_config_list = [RnnInputNodeConfig(2, [0, 1], nll)]
#     for i in range(n_nodes - 1):
#         node_config_list.append(RnnNodeConfig(3 + i, np.linspace(2, 2 + i, 1 + i).astype('int'), nll))
#     return SearchSpace(node_config_list)