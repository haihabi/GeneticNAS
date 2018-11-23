import unittest
import torch
import numpy as np
from gnas.search_space.space_config import OperationConfig, AlignmentConfig
from gnas.search_space.search_space import SearchSpace
from gnas.models.sub_graph_module import SubGraphModule


class TestModules(unittest.TestCase):
    def test_sub_graph_build_rnn(self):
        oc = OperationConfig(128, ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid'], ['Linear'], ['Add'])
        ac = AlignmentConfig('Linear', 16, 128)

        n_nodes = 4
        n_output = 1
        ss = SearchSpace([ac, ac], oc, 2, n_nodes, n_output)
        sgm = SubGraphModule(ss)

        sgm.set_individual(ss.generate_individual())

    def test_run_sub_module(self):
        oc = OperationConfig(128, ['Tanh', 'ReLU', 'ReLU6', 'Sigmoid'], ['Linear'], ['Add'])
        x_ac = AlignmentConfig('Linear', 16, 128)
        y_ac = AlignmentConfig('Identity', 128, 128)

        n_nodes = 4
        n_output = 1
        n_input = 2
        ss = SearchSpace([x_ac, y_ac], oc, n_input, n_nodes, n_output)
        sgm = SubGraphModule(ss)
        y = torch.randn(25, 128, dtype=torch.float)
        for i in range(100):
            sgm.set_individual(ss.generate_individual())
            x = torch.randn(25, 16, dtype=torch.float)
            y = sgm(x, y)


if __name__ == '__main__':
    unittest.main()
