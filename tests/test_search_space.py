import unittest
import numpy as np
import os
import inspect
# from gnas.search_space.space_config import OperationConfig, AlignmentConfig
from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.cross_over import individual_uniform_crossover
from gnas.common.graph_draw import draw_network


class TestSearchSpace(unittest.TestCase):
    @staticmethod
    def generate_ss():
        nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
        node_config_list = [RnnInputNodeConfig(32, 128, nll)]
        for i in range(12):
            node_config_list.append(RnnNodeConfig(128, nll))
        ss = SearchSpace(node_config_list)
        return ss

    def test_basic(self):
        ss = self.generate_ss()
        individual = ss.generate_individual()
        # self._test_individual(individual, n_nodes, n_output)

    def test_cross_over(self):
        ss = self.generate_ss()
        for i in range(100):
            individual_a = ss.generate_individual()
            individual_b = ss.generate_individual()
            individual_c = individual_uniform_crossover(individual_a, individual_b)
            self._test_individual(individual_c, ss.get_n_nodes())

    def test_plot_individual(self):
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        ss = self.generate_ss()
        if os.path.isfile(os.path.join(current_path, 'graph.png')):
            os.remove(os.path.join(current_path, 'graph.png'))
        individual = ss.generate_individual()
        draw_network(ss, individual, os.path.join(current_path, 'graph.png'))

    def _test_individual(self, individual, n_nodes):
        # self.assertTrue(len(individual.connection_vector) == (n_nodes + n_output))
        self.assertTrue(len(individual.iv) == n_nodes)
        # for i in individual.connection_vector:
        #     self.assertTrue(np.sum(i) != 0)
        individual.generate_node_config()


if __name__ == '__main__':
    unittest.main()
