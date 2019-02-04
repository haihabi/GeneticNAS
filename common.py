import os
import pickle
import datetime
from config import save_config


def make_log_dir(config):
    log_dir = os.path.join('.', 'logs', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(log_dir, exist_ok=True)
    save_config(log_dir, config)
    return log_dir


def load_final(model, search_dir):
    ind_file = os.path.join(search_dir, 'best_individual.pickle')
    ind = pickle.load(open(ind_file, "rb"))
    model.set_individual(ind)
    return ind
