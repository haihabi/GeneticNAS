from gnas.search_space.search_space import SearchSpace
from gnas.search_space.space_config import OperationConfig, AlignmentConfig

__operation_config_dict__ = {'ENAS-RNN': OperationConfig(['Tanh', 'ReLU', 'ReLU6', 'Sigmoid'], ['Linear'], ['Add'])}


def get_search_space(alignment_operator: str, operation_space: str, n_inputs: int, n_nodes: int,
                     n_outputs: int) -> SearchSpace:
    ac = AlignmentConfig(alignment_operator)
    oc = __operation_config_dict__.get(operation_space)
    if oc is None: raise Exception('Unkown operation config type')
    return SearchSpace(ac, oc, n_inputs, n_nodes, n_outputs)
