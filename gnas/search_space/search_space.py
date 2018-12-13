import numpy as np
from gnas.search_space.individual import Individual, MultipleBlockIndividual


class SearchSpace(object):
    def __init__(self, operation_config_list: list, single_block=True):
        self.single_block = single_block
        self.ocl = operation_config_list
        self.n_elements = sum([len(self.generate_vector(o.max_values_vector(i))) for i, o in enumerate(self.ocl)])

    def get_operation_configs(self):
        return self.ocl

    def get_n_nodes(self):
        if self.single_block:
            return len(self.ocl)
        else:
            return [len(ocl) for ocl in self.ocl]

    def get_max_values_vector(self):
        if self.single_block:
            return [o.max_values_vector(i) for i, o in enumerate(self.ocl)]
        else:
            raise NotImplemented

    def generate_vector(self, max_values):
        return np.asarray([np.random.randint(0, mv + 1) for mv in max_values])

    def _generate_individual_single(self, ocl):
        operation_vector = [self.generate_vector(o.max_values_vector(i)) for i, o in enumerate(ocl)]
        max_inputs = [i for i, _ in enumerate(ocl)]
        return Individual(operation_vector, max_inputs, self)

    def generate_individual(self):
        if self.single_block:
            return self._generate_individual_single(self.ocl)
        else:
            return MultipleBlockIndividual([self._generate_individual_single(ocl) for ocl in self.ocl])

    def generate_population(self, size):
        if self.single_block:
            return [self.generate_individual() for _ in range(size)]
        else:
            raise NotImplemented
