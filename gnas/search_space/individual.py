import numpy as np


class Individual(object):
    def __init__(self, individual_vector, max_inputs, search_space):
        self.iv = individual_vector
        self.mi = max_inputs
        self.ss = search_space
        # Generate config when generating individual
        self.config_list = [oc.parse_config(iv) for iv, oc in zip(self.iv, self.ss.ocl)]

    def generate_node_config(self):
        return self.config_list

    def update_individual(self, individual_vector):
        return Individual(individual_vector, self.mi, self.ss)

    def n_elements(self):
        return sum([len(i) for i in self.iv])

    def get_array(self):
        return np.asarray(self.iv).flatten()


class MultipleBlockIndividual(object):
    def __init__(self, individual_list):
        self.individual_list = individual_list

    def generate_node_config(self):
        raise NotImplemented

    def update_individual(self, individual_vector):
        raise NotImplemented
