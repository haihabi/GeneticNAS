import unittest
import torch
import gnas
import time
from gnas.search_space.space_config import OperationConfig, AlignmentConfig
from gnas.search_space.search_space import SearchSpace
from gnas.modules.sub_graph_module import SubGraphModule


class TestModules(unittest.TestCase):
    def test_sub_graph_build_rnn(self):
        oc = OperationConfig(['Tanh', 'ReLU', 'ReLU6', 'Sigmoid'], ['Linear'], ['Add'])
        ac = AlignmentConfig('Linear')

        n_nodes = 4
        n_output = 1
        ss = SearchSpace(ac, oc, 2, n_nodes, n_output)
        sgm = SubGraphModule([16, 128], 128, ss)

        sgm.set_individual(ss.generate_individual())

    def test_run_sub_module(self):
        oc = OperationConfig(['Tanh', 'ReLU', 'ReLU6', 'Sigmoid'], ['Linear'], ['Add'])
        ac = AlignmentConfig('Linear')

        n_nodes = 4
        n_output = 1
        n_input = 2
        ss = SearchSpace(ac, oc, n_input, n_nodes, n_output)
        sgm = SubGraphModule([16, 128], 128, ss)
        y = torch.randn(25, 128, dtype=torch.float)
        for i in range(100):
            sgm.set_individual(ss.generate_individual())
            x = torch.randn(25, 16, dtype=torch.float)
            y = sgm(x, y)

    def test_rnn_module(self):
        batch_size = 64
        in_channels = 300
        out_channels = 128
        time_steps = 35
        input = torch.randn(time_steps, batch_size, in_channels, dtype=torch.float)

        ss = gnas.get_search_space('Linear', 'ENAS-RNN', n_inputs=2, n_nodes=11, n_outputs=1)
        rnn = gnas.modules.RnnSearchModule(in_channels=in_channels, n_channels=out_channels, working_device='cpu',
                                           ss=ss)
        rnn.set_individual(ss.generate_individual())

        state = rnn.init_state(batch_size)
        s = time.time()
        output, state = rnn(input, state)
        print(time.time() - s)
        self.assertTrue(output.shape[1] == batch_size)
        self.assertTrue(output.shape[0] == time_steps)
        self.assertTrue(output.shape[2] == out_channels)


if __name__ == '__main__':
    unittest.main()
