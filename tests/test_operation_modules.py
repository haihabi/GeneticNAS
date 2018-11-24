import unittest
import torch
from gnas.modules.operation_modules import MLinear
from torch.nn import functional as  F
from torch.nn.parameter import Parameter
import time
from torch import nn


class TestOperation(unittest.TestCase):
    def test_speed(self):
        out_features = 128
        n_inputs = 10
        batch_size = 32

        input_list = []
        weight_list = []
        bias_list = []
        output_vector = torch.rand(batch_size, out_features)
        for _ in range(n_inputs):
            tensor_input = torch.rand(batch_size, out_features)
            tensor_weight = Parameter(torch.Tensor(out_features, out_features))
            tensor_bias = Parameter(torch.Tensor(out_features))
            input_list.append(tensor_input)
            weight_list.append(tensor_weight)
            bias_list.append(tensor_bias)

        s = time.time()
        res = None
        for i in range(n_inputs):
            if res is None:
                res = F.linear(input_list[i], weight_list[i], bias_list[i])
            else:
                res += F.linear(input_list[i], weight_list[i], bias_list[i])
        loss = torch.mean(torch.pow(res - output_vector, 2))
        loss.backward()
        print(time.time() - s)
        ml = MLinear(n_inputs, out_features)

        # w = torch.cat(weight_list, -1)
        # b = torch.cat(bias_list, -1)
        # i = torch.cat(input_list, -1)
        # print(list(ml.parameters()))
        ml.set_input_index([0, 1, 2])
        s = time.time()
        res = ml(input_list)
        loss = torch.mean(torch.pow(res - output_vector, 2))
        loss.backward()
        print(time.time() - s)


if __name__ == '__main__':
    unittest.main()
