# import numpy as np
# from gnas.common.bit_utils import vector_bits2int
# from gnas.modules.operation_factory import get_module
#
#
# def is_size_power_of_two(input_list: list):
#     n = len(input_list)
#     if n == 0: raise Exception('Input list is empty')
#     n_gal = np.power(2, np.floor(np.log2(n)))
#     return not (n != 1 and n_gal != n)
#
#
# class OperationConfig(object):
#     def __init__(self, non_linear_list: list):
#         if not is_size_power_of_two(non_linear_list): raise Exception('Non Linear list is not power of two')
#         self.non_linear_list = non_linear_list
#         self.nl_bits = np.floor(np.log2(len(non_linear_list))).astype('int')
#
#     def get_non_linear_modules(self):
#         return [get_module(nl)() for nl in self.non_linear_list]
#
#     def calculate_operation_bits(self):
#         return self.nl_bits
#
#     def generate_operation_vector(self, max_inputs):
#         operation = np.round(np.random.rand(self.calculate_operation_bits())).astype('int')
#         if max_inputs != 0:
#             connection = np.asarray(np.random.randint(0, max_inputs)).reshape([1])
#             return np.concatenate([connection, operation])
#         else:
#             return operation
#
#     def calculate_operation_index(self, operation_vector):
#         if len(operation_vector) != self.calculate_operation_bits(): raise Exception(
#             'Operation vector size don\'t fit the current config')
#         nl_index, operation_vector = self._get_current_index(operation_vector, self.nl_bits)
#         return nl_index
#
#     @staticmethod
#     def _get_current_index(operation_vector, n_bits):
#         if n_bits == 0:
#             index = 0
#         else:
#             index = vector_bits2int(operation_vector[:n_bits])
#
#         return index, operation_vector[n_bits:]
