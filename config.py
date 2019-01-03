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
    return {'batch_size': 128,
            'batch_size_val': 1024,
            'n_epochs': 310,
            'n_blocks': 2,
            'n_nodes': 5,
            'n_channels': 20,
            'generation_size': 20,
            'population_size': 60,
            'learning_rate': 0.1,
            'lr_min': 0.0001,
            'weight_decay': 0.0001,
            'delay': 10,
            'dropout': 0.0,
            'drop_path_keep_prob': 1.0,
            'LRType': 'CosineAnnealingLR',
            'num_class': 10,
            'momentum': 0.9}
