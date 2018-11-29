import numpy as np
from gnas.common.bit_utils import vector_bits2int


class RnnInputNodeConfig(object):
    def __init__(self, node_id, inputs: list, x_size, recurrent_size, non_linear_list):
        self.node_id = node_id
        self.inputs = inputs
        self.x_size = x_size
        self.recurrent_size = recurrent_size
        self.non_linear_list = non_linear_list

    def get_n_bits(self, max_inputs):
        return np.log2(len(self.non_linear_list)).astype('int')

    def max_values_vector(self, max_inputs):
        return np.ones(self.get_n_bits(0))

    def get_n_inputs(self):
        return len(self.inputs)

    def parse_config(self, config):
        return vector_bits2int(config.oc)


class RnnNodeConfig(object):
    def __init__(self, node_id, inputs: list, recurrent_size, non_linear_list):
        self.node_id = node_id
        self.inputs = inputs
        self.recurrent_size = recurrent_size
        self.non_linear_list = non_linear_list

    def get_n_bits(self, max_inputs):
        return np.log2(len(self.non_linear_list)).astype('int') + (max_inputs > 1)

    def max_values_vector(self, max_inputs):
        op_bits = np.ones(self.get_n_bits(0))
        if (max_inputs > 1):
            return np.concatenate([np.asarray(max_inputs - 1).reshape(1), op_bits])
        return op_bits

    def get_n_inputs(self):
        return len(self.inputs)

    def parse_config(self, config):
        if len(self.inputs) == 1:
            return self.inputs[0], 0, vector_bits2int(config.oc)
        else:
            return self.inputs[config.oc[0]], config.oc[0], vector_bits2int(config.oc[1:])
