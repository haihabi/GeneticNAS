from gnas.search_space.space_config import OperationConfig
from gnas.search_space.individual import Individual
from gnas.models.node_module import NodeModule
from gnas.models.alignment_module import AlignmentModule


class SearchSpace(object):
    def __init__(self, ac_list: list, oc: OperationConfig, n_inputs: int, n_nodes: int,
                 n_outputs: int):
        self.ac_list = ac_list
        self.oc = oc
        self.n_inputs = n_inputs
        if len(ac_list) != n_inputs:
            raise Exception('The number of alignment configs is not equal to the number of inputs')
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes

    def get_n_channels(self) -> int:
        return self.oc.n_channels

    def generate_nodes_modules(self):
        return [NodeModule(min(i + self.n_inputs, self.n_inputs + self.n_nodes), self.oc) for i in
                range(self.n_outputs + self.n_nodes)]

    def generate_alignment_modules(self):
        return [AlignmentModule(self.ac_list[i]) for i in range(self.n_inputs)]

    def generate_individual(self):
        connection_vector = []
        operation_vector = []
        for i in range(self.n_inputs, self.n_inputs + self.n_nodes + self.n_outputs):
            cv, ov = self.oc.generate_operation_vector(
                max_inputs=min(i, self.n_inputs + self.n_nodes))
            connection_vector.append(cv)
            operation_vector.append(ov)
        return Individual(connection_vector, operation_vector, self.oc)

    def generate_population(self, size):
        return [self.generate_individual() for _ in range(size)]
