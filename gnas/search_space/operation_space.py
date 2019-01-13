import numpy as np
from gnas.common.bit_utils import vector_bits2int


class RnnInputNodeConfig(object):
    def __init__(self, node_id, inputs: list, non_linear_list):
        self.node_id = node_id
        self.inputs = inputs
        self.non_linear_list = non_linear_list

    def get_n_bits(self, max_inputs):
        return np.log2(len(self.non_linear_list)).astype('int')

    def max_values_vector(self, max_inputs):
        return np.ones(self.get_n_bits(0))

    def get_n_inputs(self):
        return len(self.inputs)

    def parse_config(self, oc):
        return vector_bits2int(oc)


class RnnNodeConfig(object):
    def __init__(self, node_id, inputs: list, non_linear_list):
        self.node_id = node_id
        self.inputs = inputs
        self.non_linear_list = non_linear_list

    def get_n_bits(self, max_inputs):
        return np.log2(len(self.non_linear_list)).astype('int') + (max_inputs > 1)

    def max_values_vector(self, max_inputs):
        op_bits = np.ones(self.get_n_bits(0))
        if max_inputs > 1:
            return np.concatenate([np.asarray(max_inputs - 1).reshape(1), op_bits])
        return op_bits

    def get_n_inputs(self):
        return len(self.inputs)

    def parse_config(self, oc):
        if len(self.inputs) == 1:
            return self.inputs[0], 0, vector_bits2int(oc)
        else:
            return self.inputs[oc[0]], oc[0], vector_bits2int(oc[1:])


class CnnNodeConfig(object):
    def __init__(self, node_id, inputs: list, op_list, drop_path_control):
        self.node_id = node_id
        self.inputs = inputs
        self.op_list = op_list
        self.drop_path_control = drop_path_control

    def max_values_vector(self, max_inputs):
        max_inputs = len(self.inputs)
        if max_inputs > 1:
            return np.asarray([max_inputs - 1, max_inputs - 1, len(self.op_list) - 1, len(self.op_list) - 1])
        return np.asarray([len(self.op_list) - 1, len(self.op_list) - 1])

    def get_n_inputs(self):
        return len(self.inputs)

    def parse_config(self, oc):
        if len(self.inputs) == 1:
            op_a = oc[0]
            op_b = oc[1]
            input_index_a = 0
            input_index_b = 0
        else:
            input_index_a = oc[0]
            input_index_b = oc[1]
            op_a = oc[2]
            op_b = oc[3]
        input_a = self.inputs[input_index_a]
        input_b = self.inputs[input_index_b]
        return input_a, input_b, input_index_a, input_index_b, op_a, op_b
