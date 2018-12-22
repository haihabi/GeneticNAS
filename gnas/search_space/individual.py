import numpy as np


class Individual(object):
    def __init__(self, individual_vector, max_inputs, search_space):
        self.iv = individual_vector
        self.mi = max_inputs
        self.ss = search_space
        # Generate config when generating individual
        self.config_list = [oc.parse_config(iv) for iv, oc in zip(self.iv, self.ss.ocl)]
        self.code = self.get_array()

    def copy(self):
        return Individual(self.iv, self.mi, self.ss)

    def generate_node_config(self):
        return self.config_list

    def update_individual(self, individual_vector):
        return Individual(individual_vector, self.mi, self.ss)

    def n_elements(self):
        return sum([len(i) for i in self.iv])

    def get_array(self):
        return np.concatenate(self.iv,axis=0)

    def __eq__(self, other):
        return np.array_equal(self.code, other.code)

    def __str__(self):
        return "code:" + str(self.code)

    def __hash__(self):
        return hash(str(self))


class MultipleBlockIndividual(object):
    def __init__(self, individual_list):
        self.individual_list = individual_list

    def generate_node_config(self):
        raise NotImplemented

    def update_individual(self, individual_vector):
        raise NotImplemented
