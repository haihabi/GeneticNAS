import numpy as np
from enum import Enum
from gnas.common.bit_utils import vector_bits2int
from gnas.modules.operation_modules import get_module


def is_size_power_of_two(input_list: list):
    n = len(input_list)
    if n == 0: raise Exception('Input list is empty')
    n_gal = np.power(2, np.floor(np.log2(n)))
    return not (n != 1 and n_gal != n)


class AlignmentConfig(object):
    def __init__(self, alignment_operator: str):
        self.alignment_operator = alignment_operator

    def get_weight_modules(self, in_channels, out_channels):
        return get_module(self.alignment_operator)(in_channels, out_channels)


class OperationConfig(object):
    def __init__(self, non_linear_list: list, weight_op_list: list, merge_op_list: list):
        weight_op_list = ['NC', *weight_op_list]
        if not is_size_power_of_two(non_linear_list): raise Exception('Non Linear list is not power of two')
        if not is_size_power_of_two(weight_op_list): raise Exception('Weight Operation list is not power of two')
        if not is_size_power_of_two(merge_op_list): raise Exception(
            'Merge Operation list is not power of two')
        self.non_linear_list = non_linear_list
        self.weight_op_list = weight_op_list
        self.merge_op_list = merge_op_list

        self.nl_bits = np.floor(np.log2(len(non_linear_list))).astype('int')
        self.wo_bits = np.floor(np.log2(len(weight_op_list))).astype('int')  # the plus one
        self.mo_bits = np.floor(np.log2(len(merge_op_list))).astype('int')

    def get_non_linear_modules(self):
        return [get_module(nl)() for nl in self.non_linear_list]

    def get_merge_op_modules(self):
        return [get_module(mo)() for mo in self.merge_op_list]

    def get_weight_modules(self, max_inputs, n_channels):
        return [[get_module(wo)(n_channels, n_channels) for wo in self.weight_op_list] for _ in
                range(max_inputs)]

    def calculate_operation_bits(self):
        return self.nl_bits + self.mo_bits

    def calculate_connection_bits(self, max_inputs):
        return max_inputs * self.wo_bits

    def generate_operation_vector(self, max_inputs):
        connection = np.round(np.random.rand(self.calculate_connection_bits(max_inputs))).astype('int')
        if np.sum(connection) == 0:
            connection[np.random.randint(0, len(connection))] = 1
        return connection, np.round(
            np.random.rand(self.calculate_operation_bits())).astype('int')

    def calculate_connection_index(self, connection_vector):
        if connection_vector.shape[0] % self.wo_bits != 0: raise Exception('connection vector size is illegal')
        connection_index = []
        while len(connection_vector) != 0:
            ci, connection_vector = self._get_current_index(connection_vector, self.wo_bits)
            connection_index.append(ci)
        return connection_index

    def calculate_operation_index(self, operation_vector):
        if len(operation_vector) != self.calculate_operation_bits(): raise Exception(
            'Operation vector size don\'t fit the current config')
        nl_index, operation_vector = self._get_current_index(operation_vector, self.nl_bits)
        mo_index, _ = self._get_current_index(operation_vector, self.mo_bits)
        return nl_index, mo_index

    @staticmethod
    def _get_current_index(operation_vector, n_bits):
        if n_bits == 0:
            index = 0
        else:
            index = vector_bits2int(operation_vector[:n_bits])

        return index, operation_vector[n_bits:]
