import time
import torch.nn as nn

import torch
import torch.optim as optim
import os
import pickle
import argparse

import gnas
from models import model_cnn, model_rnn
from cnn_utils import evaluate_single, evaluate_individual_list
from rnn_utils import train_genetic_rnn, rnn_genetic_evaluate, rnn_evaluate
from data import get_dataset
from common import load_final, make_log_dir, get_model_type, ModelType
from config import get_config, load_config, save_config
from modules.drop_module import DropModuleControl
from modules.cosine_annealing import CosineAnnealingLR

#######################################
# Constants
#######################################
log_interval = 200
#######################################
# User input
#######################################
parser = argparse.ArgumentParser(description='PyTorch GNAS')
parser.add_argument('--dataset_name', type=str, choices=['CIFAR10', 'CIFAR100', 'PTB'], help='the working data',
                    default='CIFAR10')
parser.add_argument('--config_file', type=str, help='location of the config file')
parser.add_argument('--search_dir', type=str, help='the log dir of the search')
parser.add_argument('--final', type=bool, help='location of the config file', default=False)
parser.add_argument('--data_path', type=str, default='./dataset/', help='location of the dataset')
args = parser.parse_args()
#######################################
# Search Working Device
#######################################
working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(working_device)
#######################################
# Set seed
#######################################
model_type = get_model_type(dataset_name=args.dataset_name)
print("Selected mode type:" + str(model_type))
#######################################
# Parameters
#######################################
config = get_config(model_type)
if args.config_file is not None:
    print("Loading config file:" + args.config_file)
    config.update(load_config(args.config_file))
config.update({'data_path': args.data_path, 'dataset_name': args.dataset_name, 'working_device': str(working_device)})
print(config)
######################################
# Read dataset and set augmentation
######################################
trainloader, testloader, n_param = get_dataset(config)
######################################
# Config model and search space
######################################
if model_type == ModelType.CNN:
    min_objective = False
    n_cell_type = gnas.SearchSpaceType(config.get('n_block_type') - 1)
    dp_control = DropModuleControl(config.get('drop_path_keep_prob'))
    ss = gnas.get_gnas_cnn_search_space(config.get('n_nodes'), dp_control, n_cell_type)

    net = model_cnn.Net(config.get('n_blocks'), config.get('n_channels'), n_param,
                        config.get('dropout'),
                        ss, aux=config.get('aux_loss')).to(working_device)
    ######################################
    # Build Optimizer and Loss function
    #####################################
    optimizer = optim.SGD(net.parameters(), lr=config.get('learning_rate'), momentum=config.get('momentum'),
                          nesterov=True,
                          weight_decay=config.get('weight_decay'))
elif model_type == ModelType.RNN:
    min_objective = True
    ntokens = n_param
    ss = gnas.get_gnas_rnn_search_space(config.get('n_nodes'))
    net = model_rnn.RNNModel(ntokens, config.get('n_channels'), config.get('n_channels'), config.get('n_blocks'),
                             config.get('dropout'),
                             tie_weights=True,
                             ss=ss).to(
        working_device)
    ######################################
    # Build Optimizer and Loss function
    #####################################
    optimizer = optim.SGD(net.parameters(), lr=config.get('learning_rate'),
                          weight_decay=config.get('weight_decay'))
######################################
# Build genetic_algorithm_searcher
#####################################
ga = gnas.genetic_algorithm_searcher(ss, generation_size=config.get('generation_size'),
                                     population_size=config.get('population_size'),
                                     keep_size=config.get('keep_size'), mutation_p=config.get('mutation_p'),
                                     p_cross_over=config.get('p_cross_over'),
                                     cross_over_type=config.get('cross_over_type'),
                                     min_objective=min_objective)
######################################
# Loss function
######################################
criterion = nn.CrossEntropyLoss()
######################################
# Select Learning schedule
#####################################
if config.get('LRType') == 'CosineAnnealingLR':
    scheduler = CosineAnnealingLR(optimizer, 10, 2, config.get('lr_min'))
elif config.get('LRType') == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [int(config.get('n_epochs') / 2), int(3 * config.get('n_epochs') / 4)])
elif config.get('LRType') == 'ExponentialLR':
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.get('gamma'))
else:
    raise Exception('unkown LRType:' + config.get('LRType'))

##################################################
# Generate log dir and Save Params
##################################################
log_dir = make_log_dir(config)
save_config(log_dir, config)
#######################################
# Load Indvidual
#######################################
if args.final: ind = load_final(net, args.search_dir)
##################################################
# Start Epochs
##################################################
ra = gnas.ResultAppender()
if model_type == ModelType.CNN:
    best = 0
    print("Starting Traing with CNN Model")
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
        ############################################
        # Loop over batchs update weights
        ############################################
        for i, (inputs, labels) in enumerate(trainloader, 0):  # Loop over batchs
            # get the inputs
            # sample child from population
            if not args.final:
                net.set_individual(ga.sample_child())

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
        ############################################
        # Update GA population
        ############################################
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
                    evaluate_individual_list(ga.get_current_generation(), ga, net, testloader,
                                             working_device)  # evaluate next generation on the validation set
                    _, _, v_max, _, n_d = ga.update_population()  # replacement
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
elif model_type == ModelType.RNN:
    best = 1000
    for epoch in range(1, config.get('n_epochs') + 1):
        if epoch > 15:
            scheduler.step()
        epoch_start_time = time.time()
        eval_batch_size = config.get('batch_size_val')
        train_loss = train_genetic_rnn(ga, trainloader, net, optimizer, criterion, ntokens, config.get('batch_size'),
                                       config.get('bptt'), config.get('clip'),
                                       log_interval, args.final)
        if args.final:
            min_loss = rnn_evaluate(net, criterion, testloader, ntokens, config.get('batch_size_val'),
                                    config.get('bptt'))
        else:
            val_loss, loss_var, max_loss, min_loss, n_diff = rnn_genetic_evaluate(ga, net, criterion, testloader,
                                                                                  ntokens,
                                                                                  config.get('batch_size_val'),
                                                                                  config.get('bptt'))

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | lr {:02.2f} |  '
              ''.format(epoch, (time.time() - epoch_start_time),
                        min_loss, scheduler.get_lr()[-1]))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if min_loss < best:
            print("Update Best")
            torch.save(net.state_dict(), os.path.join(log_dir, 'best_model.pt'))
            if not args.final:
                gnas.draw_network(ss, ga.best_individual, os.path.join(log_dir, 'best_graph_' + str(epoch) + '_'))
                pickle.dump(ga.best_individual, open(os.path.join(log_dir, 'best_individual.pickle'), "wb"))

            best = min_loss

        ra.add_epoch_result('Loss', train_loss)
        ra.add_epoch_result('LR', scheduler.get_lr()[-1])
        ra.add_epoch_result('Best', best)
        if not args.final: ra.add_result('Fitness', ga.ga_result.fitness_list)
        ra.save_result(log_dir)
print('Finished Training')
