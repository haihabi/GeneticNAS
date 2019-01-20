import json
import os


def save_config(path_dir, config):
    with open(os.path.join(path_dir, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)


def load_config(path_dir):
    with open(path_dir, 'r') as json_file:
        data = json.load(json_file)
    return data


def default_config():
    return {'batch_size': 256,
            'batch_size_val': 1000,
            'n_epochs': 310,
            'n_blocks': 1,
            'n_block_type': 3,
            'n_nodes': 5,
            'n_channels': 20,
            'generation_size': 20,
            'generation_per_epoch': 2,
            'full_dataset': False,
            'population_size': 20,
            'keep_size': 0,
            'mutation_p': None,
            'p_cross_over': 1.0,
            'cross_over_type': 'Block',
            'learning_rate': 0.2,
            'lr_min': 0.0002,
            'weight_decay': 0.0001,
            'delay': 10,
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
