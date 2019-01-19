import numpy as np


class Individual(object):
    def __init__(self, individual_vector, max_inputs, search_space, index=0):
        self.iv = individual_vector
        self.mi = max_inputs
        self.ss = search_space
        # Generate config when generating individual
        self.index = index
        self.config_list = [oc.parse_config(iv) for iv, oc in zip(self.iv, self.ss.get_opeartion_config(self.index))]
        self.code = np.concatenate(self.iv, axis=0)

    def get_length(self):
        return len(self.code)

    def get_n_op(self):
        return len(self.iv)

    def copy(self):
        return Individual(self.iv, self.mi, self.ss, index=self.index)

    def generate_node_config(self):
        return self.config_list

    def update_individual(self, individual_vector):
        return Individual(individual_vector, self.mi, self.ss, index=self.index)

    def __eq__(self, other):
        return np.array_equal(self.code, other.code)

    def __str__(self):
        return "code:" + str(self.code)

    def __hash__(self):
        return hash(str(self))


class MultipleBlockIndividual(object):
    def __init__(self, individual_list):
        self.individual_list = individual_list
        self.code = np.concatenate([i.code for i in self.individual_list])

    def get_individual(self, index):
        return self.individual_list[index]

    def generate_node_config(self, index):
        return self.individual_list[index].generate_node_config()

    def update_individual(self, individual_vector):
        raise NotImplemented

    def __eq__(self, other):
        return np.array_equal(self.code, other.code)

    def __str__(self):
        return "code:" + str(self.code)

    def __hash__(self):
        return hash(str(self))
