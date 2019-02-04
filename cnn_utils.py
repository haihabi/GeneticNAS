import numpy as np
import math
import torch.optim as optim

import torch
import torch.cuda
import torch.nn as nn
import torch.functional as F
from random import random
from torch.autograd import Variable


def evaluate_single(input_individual, input_model, data_loader, device):
    correct = 0
    total = 0
    input_model = input_model.eval()
    input_model.set_individual(input_individual)
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = input_model(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def evaluate_individual_list(input_individual_list, ga, input_model, data_loader, device):
    correct = 0
    total = 0
    input_model = input_model.eval()
    i = 0
    with torch.no_grad():
        while len(input_individual_list) > i:
            for data in data_loader:
                if len(input_individual_list) <= i:
                    pass
                else:
                    ind = input_individual_list[i]
                    input_model.set_individual(ind)
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = input_model(images)
                    _, predicted = torch.max(outputs[0].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    ga.update_current_individual_fitness(ind, acc)
                    i += 1


# Currently there is a risk of dropping all paths...
# We should create a version that take all paths into account to make sure one stays alive
# But then keep_prob is meaningless and we have to copute/keep track of the conditional probability
class DropPath(nn.Module):
    def __init__(self, module, keep_prob=0.9):
        super(DropPath, self).__init__()
        self.module = module
        self.keep_prob = keep_prob
        self.shape = None
        self.training = True
        self.dtype = torch.FloatTensor

    def forward(self, *input):
        if self.training:
            # If we don't now the shape we run the forward path once and store the output shape
            if self.shape is None:
                temp = self.module(*input)
                self.shape = temp.size()
                if temp.data.is_cuda:
                    self.dtype = torch.cuda.FloatTensor
                del temp
            p = random()
            if p <= self.keep_prob:
                return Variable(self.dtype(self.shape).zero_())
            else:
                return self.module(*input) / self.keep_prob  # Inverted scaling
        else:
            return self.module(*input)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class CosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mul, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mul = T_mul
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = [self.eta_min + (base_lr - self.eta_min) *
              (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
              for base_lr in self.base_lrs]
        if self.last_epoch != 0 and self.last_epoch % self.T_max == 0:
            self.T_max = self.T_mul * self.T_max
            self.last_epoch = 0
        return lr
