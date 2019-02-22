import unittest
import torch
import gnas
import time
from tests.common4testing import generate_ss, generate_ss_cnn
from gnas.modules.sub_graph_module import SubGraphModule
from modules.drop_module import DropModuleControl

class TestModules(unittest.TestCase):
    def test_sub_graph_build_rnn(self):
        ss = generate_ss()
        sgm = SubGraphModule(ss, {'in_channels': 32, 'n_channels': 128})

        sgm.set_individual(ss.generate_individual())

    def test_run_sub_module(self):
        ss = generate_ss()
        sgm = SubGraphModule(ss, {'in_channels': 32, 'n_channels': 128})
        y = torch.randn(25, 128, dtype=torch.float)
        for i in range(100):
            sgm.set_individual(ss.generate_individual())
            x = torch.randn(25, 32, dtype=torch.float)
            y = sgm(x, y)
            y = y[-1]

    def test_cnn_sub_module(self):
        ss = generate_ss_cnn()
        sgm = SubGraphModule(ss, {'n_channels': 64})

        for i in range(100):
            sgm.set_individual(ss.generate_individual())
            y = torch.randn(32, 64, 16, 16, dtype=torch.float)
            x = torch.randn(32, 64, 16, 16, dtype=torch.float)
            res = sgm(x, y)

    def test_cnn_module(self):
        batch_size = 64
        h, w = 16, 16
        channels = 64
        input = torch.randn(batch_size, channels, h, w, dtype=torch.float)
        input_b = torch.randn(batch_size, channels, h, w, dtype=torch.float)
        dp_control = DropModuleControl(1)
        ss = gnas.get_gnas_cnn_search_space(4, dp_control, gnas.SearchSpaceType.CNNSingleCell)
        rnn = gnas.modules.CnnSearchModule(n_channels=channels,
                                           ss=ss)
        rnn.set_individual(ss.generate_individual())

        s = time.time()
        output = rnn(input, input_b)
        print(time.time() - s)
        self.assertTrue(output.shape[0] == batch_size)
        self.assertTrue(output.shape[1] == channels)
        self.assertTrue(output.shape[2] == h)
        self.assertTrue(output.shape[3] == w)

    def test_rnn_module(self):
        batch_size = 64
        in_channels = 300
        out_channels = 128
        time_steps = 35
        input = torch.randn(time_steps, batch_size, in_channels, dtype=torch.float)

        ss = gnas.get_gnas_rnn_search_space(12)
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
