import time
import torch.nn as nn
import torch.onnx
from models import model_cnn
import gnas
from gnas.genetic_algorithm.annealing_functions import cosine_annealing
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import math
import os
import pickle
import datetime
from config import default_config, save_config, load_config
import argparse


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


parser = argparse.ArgumentParser(description='PyTorch GNAS')
parser.add_argument('--config_file', type=str, help='location of the config file')
args = parser.parse_args()
#######################################
# Parameters
#######################################
config = default_config()
if args.config_file is not None:
    print("Loading config file:" + args.config_file)
    config.update(load_config(args.config_file))
print(config)
#######################################
# Search Working Device
#######################################
working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(working_device)
######################################
# Read dataset and set augmentation
######################################
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop([32, 32], padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.get('batch_size'),
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.get('batch_size'),
                                         shuffle=False, num_workers=4)
######################################
# Config model and search space
######################################
ss = gnas.get_enas_cnn_search_space_dual(config.get('n_nodes'))
ga = gnas.genetic_algorithm_searcher(ss, generation_size=config.get('generation_size'),
                                     population_size=config.get('population_size'), min_objective=False)
net = model_cnn.Net(config.get('n_blocks'), config.get('n_channels'), config.get('num_class'), config.get('dropout'),
                    ss)
net.to(working_device)
######################################
# Build Optimizer and Loss function
#####################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config.get('learning_rate'), momentum=config.get('momentum'), nesterov=True,
                      weight_decay=config.get('weight_decay'))
######################################
# Select Learning schedule
#####################################
if config.get('LRType') == 'CosineAnnealingLR':
    scheduler = CosineAnnealingLR(optimizer, 10, 2, 0.005)
else:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [int(config.get('n_epochs') / 2), int(3 * config.get('n_epochs') / 4)])
#
##################################################
# Generate log dir and Save Params
##################################################
log_dir = os.path.join('.', 'logs', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.makedirs(log_dir, exist_ok=True)
save_config(log_dir, config)


def evaulte_single(input_individual, input_model, data_loader, device):
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


##################################################
# Start Epochs
##################################################
ra = gnas.ResultAppender()
best = 0
for epoch in range(config.get('n_epochs')):  # loop over the dataset multiple times
    # print(epoch)
    running_loss = 0.0
    correct = 0
    total = 0
    scheduler.step()
    s = time.time()
    net = net.train()
    p = cosine_annealing(epoch, 1, delay=15, end=25)
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs
        net.set_individual(ga.sample_child(p))

        inputs = inputs.to(working_device)
        labels = labels.to(working_device)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = net(inputs)  # forward

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)

        loss.backward()  # backward

        optimizer.step()  # optimize

        # print statistics
        running_loss += loss.item()

    for ind in ga.get_current_generation():
        acc = evaulte_single(ind, net, testloader, working_device)
        ga.update_current_individual_fitness(ind, acc)
    _, _, f_max, _ = ga.update_population()
    if f_max > best:
        print("Update Best")
        best = f_max
        torch.save(net.state_dict(), os.path.join(log_dir, 'best_model.pt'))
        gnas.draw_network(ss, ga.best_individual, os.path.join(log_dir, 'best_graph_' + str(epoch) + '_'))
        pickle.dump(ga.best_individual, open(os.path.join(log_dir, 'best_individual.pickle'), "wb"))
    print(
        '|Epoch: {:2d}|Time: {:2.3f}|Loss:{:2.3f}|Accuracy: {:2.3f}%|LR: {:2.3f}|'.format(epoch, (time.time() - s) / 60,
                                                                                          running_loss / i,
                                                                                          100 * correct / total,
                                                                                          scheduler.get_lr()[-1]))
    ra.add_epoch_result('Annealing', p)
    ra.add_epoch_result('LR', scheduler.get_lr()[-1])
    ra.add_epoch_result('Training Loss', running_loss / i)
    ra.add_epoch_result('Training Accuracy', 100 * correct / total)
    ra.add_result('Fitness', ga.ga_result.fitness_list)
    ra.add_result('Fitness-Population', ga.ga_result.fitness_full_list)
    ra.save_result(log_dir)

print('Finished Training')
