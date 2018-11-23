import unittest
import numpy as np
from gnas.search_space.space_config import OperationConfig, SpaceType


class TestSpaceConfig(unittest.TestCase):
    def test_operation_config(self):
        oc = OperationConfig(128, [0, 1, 2, 3], [0], [0])
        self.assertTrue(oc.mo_bits == 0)
        self.assertTrue(oc.nl_bits == 2)
        self.assertTrue(oc.wo_bits == 1)
        connection, operation_vector = oc.generate_operation_vector(1)
        self.assertTrue(len(operation_vector) == 2)
        nl, m = oc.calculate_operation_index(operation_vector)
        self.assertTrue(nl == (operation_vector[0] * 2 + operation_vector[1]))
        self.assertTrue(m == 0)


if __name__ == '__main__':
    unittest.main()
