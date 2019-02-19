import unittest
import numpy as np
import os
import inspect

from gnas.search_space.operation_space import RnnInputNodeConfig, RnnNodeConfig
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.individual import Individual
from gnas.search_space.cross_over import individual_uniform_crossover
from gnas.common.graph_draw import draw_network
from gnas.search_space.mutation import individual_flip_mutation
import gnas


class TestSearchSpace(unittest.TestCase):
    @staticmethod
    def generate_block():
        nll = ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid']
        node_config_list = [RnnInputNodeConfig(0, [], nll)]
        for i in range(12):
            node_config_list.append(RnnNodeConfig(i + 1, [0], nll))
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

    def test_individual(self):
        ss = self.generate_ss()
        individual_a = ss.generate_individual()
        individual_b = ss.generate_individual()
        individual_a_tag = individual_a.copy()
        dict2test = dict()
        self.assertFalse(individual_a == individual_b)
        self.assertTrue(individual_a_tag == individual_a)
        dict2test.update({individual_a: 40})
        dict2test.update({individual_b: 80})
        dict2test.update({individual_a_tag: 90})
        self.assertTrue(dict2test.get(individual_a) == 90)
        self.assertTrue(dict2test.get(individual_a_tag) == 90)
        self.assertTrue(dict2test.get(individual_b) == 80)
        self.assertTrue(len(dict2test) == 2)
        # res_dict

    def test_basic_multiple(self):
        ss = self.generate_ss_multiple_blocks()
        individual = ss.generate_individual()
        self._test_individual(individual, ss.get_n_nodes())

    def test_mutation(self):
        ss = self.generate_ss()
        for i in range(100):
            individual_a = ss.generate_individual()
            individual_c = individual_flip_mutation(individual_a, 1 / 10)
            ce = 0
            te = 0
            for a, c in zip(individual_a.iv, individual_c.iv):
                for ia, ic in zip(a, c):
                    te += 1
                    ce += ia != ic

            self._test_individual(individual_c, ss.get_n_nodes())
        self.assertTrue(ce != te)

    def test_cross_over(self):
        ss = self.generate_ss()
        ca = 0
        cb = 0
        cc = 0
        for i in range(100):
            individual_a = ss.generate_individual()
            individual_b = ss.generate_individual()
            individual_c, individual_d = individual_uniform_crossover(individual_a, individual_b, 1)

            for a, b, c in zip(individual_a.iv, individual_b.iv, individual_c.iv):
                for ia, ib, ic in zip(a, b, c):
                    cc += 1
                    ca += ia == ic
                    cb += ib == ic
                    self.assertTrue(ia == ic or ib == ic)

            self._test_individual(individual_c, ss.get_n_nodes())
        self.assertTrue(cc != cb)
        self.assertTrue(cc != ca)

    def test_plot_individual_rnn(self):
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        ss = gnas.get_gnas_rnn_search_space(12)
        ind = ss.generate_individual()
        draw_network(ss, ind, os.path.join(current_path, 'graph'))

    def test_plot_individual(self):
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        ss = gnas.get_gnas_cnn_search_space(5, 1, gnas.SearchSpaceType.CNNSingleCell)
        ind = ss.generate_individual()
        draw_network(ss, ind, os.path.join(current_path, 'graph'))

    def test_plot_individual_dual(self):
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        ss = gnas.get_gnas_cnn_search_space(5, 1, gnas.SearchSpaceType.CNNDualCell)
        ind = ss.generate_individual()
        draw_network(ss, ind, os.path.join(current_path, 'graph'))

    def test_plot_individual_triple(self):
        current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        ss = gnas.get_gnas_cnn_search_space(5, 1, gnas.SearchSpaceType.CNNTripleCell)
        ind = ss.generate_individual()
        draw_network(ss, ind, os.path.join(current_path, 'graph'))

    def _test_individual(self, individual, n_nodes):
        individual_flip_mutation(individual, 0.2)
        if isinstance(individual, Individual):
            self.assertTrue(len(individual.iv) == n_nodes)
            for c in individual.iv:
                if len(c) == 2:
                    self.assertFalse(np.any(c > 1))
                else:
                    self.assertFalse(np.any(c[1:] > 1))
            individual.generate_node_config()
        else:
            [self.assertTrue(len(ind.iv) == n_nodes[i]) for i, ind in enumerate(individual.individual_list)]
            individual.generate_node_config(0)


if __name__ == '__main__':
    unittest.main()
