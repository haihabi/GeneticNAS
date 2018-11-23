from gnas.search_space.space_config import OperationConfig, AlignmentConfig
from gnas.search_space.individual import Individual
from gnas.modules.node_module import NodeModule
from gnas.modules.alignment_module import AlignmentModule


class SearchSpace(object):
    def __init__(self, ac: AlignmentConfig, oc: OperationConfig, n_inputs: int, n_nodes: int,
                 n_outputs: int):
        self.ac = ac
        self.oc = oc
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes

    def generate_nodes_modules(self, n_channels):
        return [NodeModule(min(i + self.n_inputs, self.n_inputs + self.n_nodes), n_channels, self.oc) for i in
                range(self.n_outputs + self.n_nodes)]

    def generate_alignment_modules(self, input_size, n_channels):
        return [AlignmentModule(input_size[i], n_channels, self.ac) for i in range(self.n_inputs)]

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
