import unittest
import numpy as np
import os
import inspect
from gnas.search_space.space_config import OperationConfig, AlignmentConfig
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.cross_over import individual_uniform_crossover
from gnas.common.graph_draw import draw_network


class TestSearchSpace(unittest.TestCase):
    def test_basic(self):
        oc = OperationConfig(128, [0, 1, 2, 3], [0], [0])
        ac = AlignmentConfig('Linear', 16, 128)
        n_nodes = 4
        n_output = 1
        ss = SearchSpace([ac, ac], oc, 2, n_nodes, n_output)

        individual = ss.generate_individual()
        self._test_individual(individual, n_nodes, n_output)

    def test_cross_over(self):
        oc = OperationConfig(128, [0, 1, 2, 3], [0], [0])
        ac = AlignmentConfig('Linear', 16, 128)
        n_nodes = 4
        n_output = 1
        ss = SearchSpace([ac, ac], oc, 2, n_nodes, n_output)
        for i in range(100):
            individual_a = ss.generate_individual()
            individual_b = ss.generate_individual()
            individual_c = individual_uniform_crossover(individual_a, individual_b)
            self._test_individual(individual_c, n_nodes, n_output)

    def test_plot_individual(self):
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        oc = OperationConfig(128, ['Tanh', 'ReLU', 'Sigmoid', 'ReLU6'], [0], [0])
        ac = AlignmentConfig('Linear', 16, 128)
        n_nodes = 6
        n_output = 1
        ss = SearchSpace([ac, ac], oc, 2, n_nodes, n_output)
        if os.path.isfile(os.path.join(current_path, 'graph.png')):
            os.remove(os.path.join(current_path, 'graph.png'))
        individual = ss.generate_individual()
        draw_network(ss, individual, os.path.join(current_path, 'graph.png'))

    def _test_individual(self, individual, n_nodes, n_output):
        self.assertTrue(len(individual.connection_vector) == (n_nodes + n_output))
        self.assertTrue(len(individual.operation_vector) == (n_nodes + n_output))
        for i in individual.connection_vector:
            self.assertTrue(np.sum(i) != 0)
        individual.generate_node_config()


if __name__ == '__main__':
    unittest.main()
