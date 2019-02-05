import unittest
import time
import torch.nn as nn
from models import model_cnn
import gnas
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import os
import pickle
import datetime
from config import default_config_cnn, save_config, load_config
import argparse
from cnn_utils import CosineAnnealingLR, Cutout, evaluate_individual_list, evaluate_single

from matplotlib import pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        train_transform = transforms.Compose([])
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                  shuffle=True, num_workers=4)
        d, l = next(iter(trainloader))
        for i, (m, s) in enumerate(zip([125.3, 123.0, 113.9], [63.0, 62.1, 66.7])):
            d[:, i, :, :] = s * d[:, i, :, :] + m
        d = d.cpu().numpy()
        d = np.transpose(d, [0, 2, 3, 1])[0, :, :, :].astype('uint8')
        plt.imshow(d)
        plt.show()
        print("a")


if __name__ == '__main__':
    unittest.main()
