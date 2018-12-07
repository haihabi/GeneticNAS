class NodeConfig(object):
    def __init__(self, operation_config):
        self.oc = operation_config

    def __str__(self):
        return str(self.oc.non_linear_list[self.nl_index])


class Individual(object):
    def __init__(self, individual_vector, max_inputs, search_space):
        self.iv = individual_vector
        self.mi = max_inputs
        self.ss = search_space

    def generate_node_config(self):
        return [NodeConfig(iv) for iv in self.iv]

    def update_individual(self, individual_vector):
        return Individual(individual_vector, self.mi, self.ss)

    def n_elements(self):
        return sum([len(i) for i in self.iv])


class MultipleBlockIndividual(object):
    def __init__(self, individual_list):
        self.individual_list = individual_list

    def generate_node_config(self):
        raise NotImplemented

    def update_individual(self, individual_vector):
        raise NotImplemented
