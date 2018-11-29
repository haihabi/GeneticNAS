import unittest
import torch
import gnas
import time
# from gnas.search_space.space_config import OperationConfig, AlignmentConfig
# from gnas.search_space.search_space import SearchSpace
from tests.common4testing import generate_ss
from gnas.modules.sub_graph_module import SubGraphModule


class TestModules(unittest.TestCase):
    def test_sub_graph_build_rnn(self):
        ss = generate_ss()
        sgm = SubGraphModule(ss)

        sgm.set_individual(ss.generate_individual())

    def test_run_sub_module(self):
        ss = generate_ss()
        sgm = SubGraphModule(ss)
        y = torch.randn(25, 128, dtype=torch.float)
        for i in range(100):
            sgm.set_individual(ss.generate_individual())
            x = torch.randn(25, 32, dtype=torch.float)
            y = sgm(x, y)

    def test_rnn_module(self):
        batch_size = 64
        in_channels = 300
        out_channels = 128
        time_steps = 35
        input = torch.randn(time_steps, batch_size, in_channels, dtype=torch.float)

        ss = gnas.get_enas_rnn_search_space(in_channels, out_channels, 12)
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
