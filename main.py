import time
import torch.nn as nn

import torch
import torch.optim as optim
import os
import pickle
import argparse

import gnas
from models import model_cnn, model_rnn
from cnn_utils import CosineAnnealingLR, evaluate_single, evaluate_individual_list
from data import get_cifar10
from common import load_final, make_log_dir
from config import default_config_cnn, load_config

parser = argparse.ArgumentParser(description='PyTorch GNAS')
parser.add_argument('--config_file', type=str, help='location of the config file')
parser.add_argument('--search_dir', type=str, help='the log dir of the search')
parser.add_argument('--final', type=bool, help='location of the config file', default=False)
args = parser.parse_args()

#######################################
# Parameters
#######################################
config = default_config_cnn()
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
trainloader, testloader = get_cifar10(config)

######################################
# Config model and search space
######################################
n_cell_type = gnas.SearchSpaceType(config.get('n_block_type') - 1)
dp_control = gnas.DropPathControl(config.get('drop_path_keep_prob'))
ss = gnas.get_enas_cnn_search_space(config.get('n_nodes'), dp_control, n_cell_type)

ga = gnas.genetic_algorithm_searcher(ss, generation_size=config.get('generation_size'),
                                     population_size=config.get('population_size'), delay=config.get('delay'),
                                     keep_size=config.get('keep_size'), mutation_p=config.get('mutation_p'),
                                     p_cross_over=config.get('p_cross_over'),
                                     cross_over_type=config.get('cross_over_type'),
                                     min_objective=False)

net = model_cnn.Net(config.get('n_blocks'), config.get('n_channels'), config.get('num_class'), config.get('dropout'),
                    ss, aux=config.get('aux_loss'))
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
    scheduler = CosineAnnealingLR(optimizer, 10, 2, config.get('lr_min'))
elif config.get('LRType') == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [int(config.get('n_epochs') / 2), int(3 * config.get('n_epochs') / 4)])
##################################################
# Generate log dir and Save Params
##################################################
log_dir = make_log_dir(config)
#######################################
# Load Indvidual
#######################################
if args.final: ind = load_final(net, args.search_dir)
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
    if epoch == config.get('drop_path_start_epoch'):
        dp_control.enable()
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs
        if not args.final: net.set_individual(ga.sample_child())

        inputs = inputs.to(working_device)
        labels = labels.to(working_device)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = net(inputs)  # forward

        _, predicted = torch.max(outputs[0], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs[0], labels)
        if config.get('aux_loss'): loss += config.get('aux_scale') * criterion(outputs[1], labels)
        loss.backward()  # backward

        optimizer.step()  # optimize

        # print statistics
        running_loss += loss.item()

    if args.final:
        f_max = evaluate_single(ind, net, testloader, working_device)
        n_diff = 0
    else:
        if config.get('full_dataset'):
            for ind in ga.get_current_generation():
                acc = evaluate_single(ind, net, testloader, working_device)
                ga.update_current_individual_fitness(ind, acc)
            _, _, f_max, _, n_diff = ga.update_population()
            best_individual = ga.best_individual
        else:
            f_max = 0
            n_diff = 0
            for _ in range(config.get('generation_per_epoch')):
                evaluate_individual_list(ga.get_current_generation(), ga, net, testloader, working_device)
                _, _, v_max, _, n_d = ga.update_population()
                n_diff += n_d
                if v_max > f_max:
                    f_max = v_max
                    best_individual = ga.best_individual
            f_max = evaluate_single(best_individual, net, testloader, working_device)  # evalute best
    if f_max > best:
        print("Update Best")
        best = f_max
        torch.save(net.state_dict(), os.path.join(log_dir, 'best_model.pt'))
        if not args.final:
            gnas.draw_network(ss, ga.best_individual, os.path.join(log_dir, 'best_graph_' + str(epoch) + '_'))
            pickle.dump(ga.best_individual, open(os.path.join(log_dir, 'best_individual.pickle'), "wb"))
    print(
        '|Epoch: {:2d}|Time: {:2.3f}|Loss:{:2.3f}|Accuracy: {:2.3f}%|Validation Accuracy: {:2.3f}%|LR: {:2.3f}|N Change : {:2d}|'.format(
            epoch, (
                           time.time() - s) / 60,
                   running_loss / i,
                   100 * correct / total, f_max,
            scheduler.get_lr()[
                -1],
            n_diff))
    ra.add_epoch_result('N', n_diff)
    ra.add_epoch_result('Best', best)
    ra.add_epoch_result('Validation Accuracy', f_max)
    ra.add_epoch_result('LR', scheduler.get_lr()[-1])
    ra.add_epoch_result('Training Loss', running_loss / i)
    ra.add_epoch_result('Training Accuracy', 100 * correct / total)
    if not args.final:
        ra.add_result('Fitness', ga.ga_result.fitness_list)
        ra.add_result('Fitness-Population', ga.ga_result.fitness_full_list)
    ra.save_result(log_dir)

print('Finished Training')
