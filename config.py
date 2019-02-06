import json
import os
from common import ModelType


def save_config(path_dir, config):
    with open(os.path.join(path_dir, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)


def load_config(path_dir):
    with open(path_dir, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_config(model_type):
    if ModelType.CNN == model_type:
        return default_config_cnn()
    elif ModelType.RNN == model_type:
        return default_config_rnn()
    else:
        raise Exception('unkown model type:' + str(model_type))


def default_config_rnn():
    return {'batch_size': 64,
            'batch_size_val': 10,
            'bptt': 35,
            'n_epochs': 310,
            'n_blocks': 2,
            'n_nodes': 12,
            'n_channels': 200,
            'clip': 0.25,
            'generation_size': 20,
            'population_size': 20,
            'keep_size': 0,
            'mutation_p': 0.02,
            'p_cross_over': 1.0,
            'cross_over_type': 'Block',
            'learning_rate': 20.0,
            'weight_decay': 0.0001,
            'dropout': 0.2,
            'LRType': 'ExponentialLR',
            'gamma': 0.96}


def default_config_cnn():
    return {'batch_size': 128,
            'batch_size_val': 1000,
            'n_epochs': 310,
            'n_blocks': 2,
            'n_block_type': 3,
            'n_nodes': 5,
            'n_channels': 20,
            'generation_size': 20,
            'generation_per_epoch': 2,
            'full_dataset': False,
            'population_size': 20,
            'keep_size': 0,
            'mutation_p': 0.02,
            'p_cross_over': 1.0,
            'cross_over_type': 'Block',
            'learning_rate': 0.1,
            'lr_min': 0.0001,
            'weight_decay': 0.0001,
            'dropout': 0.2,
            'drop_path_keep_prob': 1.0,
            'drop_path_start_epoch': 50,
            'cutout': True,
            'n_holes': 1,
            'length': 16,
            'LRType': 'MultiStepLR',
            'num_class': 10,
            'momentum': 0.9,
            'aux_loss': False,
            'aux_scale': 0.4}
