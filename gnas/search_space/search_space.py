import numpy as np
# from gnas.search_space.space_config import OperationConfig, AlignmentConfig
from gnas.search_space.individual import Individual


# from gnas.modules.node_module import NodeModule
# from gnas.modules.alignment_module import AlignmentModule


class SearchSpace(object):
    def __init__(self, operation_config_list: list):
        self.ocl = operation_config_list

    def get_operation_configs(self):
        return self.ocl

    def get_n_nodes(self):
        return len(self.ocl)

    def get_max_values_vector(self):
        return [o.max_values_vector(i) for i, o in enumerate(self.ocl)]

    @staticmethod
    def generate_vector(max_values):
        return np.asarray([np.random.randint(0, mv) for mv in max_values])

    def generate_individual(self):
        operation_vector = [self.generate_vector(o.max_values_vector(i)) for i, o in enumerate(self.ocl)]
        max_inputs = [i for i, _ in enumerate(self.ocl)]
        return Individual(operation_vector, max_inputs, self)

    def generate_population(self, size):
        return [self.generate_individual() for _ in range(size)]
