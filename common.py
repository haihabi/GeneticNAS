import os
import pickle
import datetime
from enum import Enum


class ModelType(Enum):
    CNN = 0
    RNN = 1


def make_log_dir(config):
    log_dir = os.path.join('.', 'logs', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def load_final(model, search_dir):
    ind_file = os.path.join(search_dir, 'best_individual.pickle')
    ind = pickle.load(open(ind_file, "rb"))
    model.set_individual(ind)
    return ind


def get_model_type(dataset_name):
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        return ModelType.CNN
    elif dataset_name == 'PTB':
        return ModelType.RNN
    else:
        raise Exception('unkown model for dataset:' + dataset_name)
