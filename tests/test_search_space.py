import unittest
import numpy as np
import os
import inspect

from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.individual import Individual
from gnas.search_space.cross_over import individual_uniform_crossover
from gnas.common.graph_draw import draw_network


class TestSearchSpace(unittest.TestCase):
    @staticmethod
    def generate_block():
        nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
        node_config_list = [RnnInputNodeConfig(0, [], 32, 128, nll)]
        for i in range(12):
            node_config_list.append(RnnNodeConfig(i + 1, [0], 128, nll))
        return node_config_list

    @staticmethod
    def generate_ss():

        ss = SearchSpace(TestSearchSpace.generate_block())
        return ss

    @staticmethod
    def generate_ss_multiple_blocks():
        block_list = [TestSearchSpace.generate_block(), TestSearchSpace.generate_block()]
        ss = SearchSpace(block_list, single_block=False)
        return ss

    def test_basic(self):
        ss = self.generate_ss()
        individual = ss.generate_individual()
        self._test_individual(individual, ss.get_n_nodes())

    def test_basic_multiple(self):
        ss = self.generate_ss_multiple_blocks()
        individual = ss.generate_individual()
        self._test_individual(individual, ss.get_n_nodes())

    def test_cross_over(self):
        ss = self.generate_ss()
        for i in range(100):
            individual_a = ss.generate_individual()
            individual_b = ss.generate_individual()
            individual_c = individual_uniform_crossover(individual_a, individual_b)
            self._test_individual(individual_c, ss.get_n_nodes())

    # def test_plot_individual(self):
    #     current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    #     ss = self.generate_ss()
    #     if os.path.isfile(os.path.join(current_path, 'graph.png')):
    #         os.remove(os.path.join(current_path, 'graph.png'))
    #     individual = ss.generate_individual()
    #     draw_network(ss, individual, os.path.join(current_path, 'graph.png'))

    def _test_individual(self, individual, n_nodes):
        if isinstance(individual, Individual):
            self.assertTrue(len(individual.iv) == n_nodes)
            individual.generate_node_config()
        else:
            [self.assertTrue(len(ind.iv) == n_nodes[i]) for i, ind in enumerate(individual.individual_list)]
            individual.generate_node_config()


if __name__ == '__main__':
    unittest.main()
