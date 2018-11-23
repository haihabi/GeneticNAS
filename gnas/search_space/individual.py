import pygraphviz as pgv


class NodeConfig(object):
    def __init__(self, connection_vector, operation_vector, operation_config):
        self.oc = operation_config
        self.connection_index = operation_config.calculate_connection_index(connection_vector)
        self.nl_index, self.mo_index = operation_config.calculate_operation_index(operation_vector)
        self.n_inputs = len(self.connection_index)

    def __str__(self):
        return str(self.oc.non_linear_list[self.nl_index]) + '-' + str(self.oc.merge_op_list[self.mo_index])


class Individual(object):
    def __init__(self, connection_vector, operation_vector, operation_config):
        self.connection_vector = connection_vector
        self.operation_vector = operation_vector
        self.oc = operation_config
        self.n_iterations = len(operation_vector)

    def generate_node_config(self):
        node_config_list = []
        for i, d in enumerate(zip(self.connection_vector, self.operation_vector)):
            cv, ov = d
            node_config_list.append(NodeConfig(cv, ov, self.oc))
        return node_config_list
