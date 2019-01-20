import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from config import load_config
import pandas as pd
from scipy.ndimage.filters import maximum_filter1d

file_list = ["/data/projects/GNAS/logs/2019_01_20_13_47_02", "/data/projects/GNAS/logs/2019_01_20_20_40_02",
             "/data/projects/GNAS/logs/2019_01_20_21_17_15"]


def read_config(file_path):
    pass


if len(file_list) == 1 and False:
    data = pickle.load(open(os.path.join(file_list[0], 'ga_result.pickle'), "rb"))
    config = load_config(os.path.join(file_list[0], 'config.json'))
    fitness = np.stack(data.result_dict.get('Fitness'))
    fitness_p = np.stack(data.result_dict.get('Fitness-Population'))

    epochs = np.linspace(0, fitness.shape[0] - 1, fitness.shape[0])
    plt.subplot(2, 3, 1)
    plt.errorbar(epochs, np.mean(fitness, axis=1), np.std(fitness, axis=1), label='population mean fitness')
    plt.plot(epochs, np.min(fitness, axis=1), '*--', label='min fitness')
    plt.plot(epochs, np.max(fitness, axis=1), '--', label='max fitness')
    plt.grid()
    plt.legend()
    plt.title('Population fitne ss over validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(2, 3, 2)
    plt.errorbar(epochs, np.mean(fitness_p, axis=1), np.std(fitness_p, axis=1), label='population mean fitness')
    plt.plot(epochs, np.min(fitness_p, axis=1), '*--', label='min fitness')
    plt.plot(epochs, np.max(fitness_p, axis=1), '--', label='max fitness')
    plt.grid()
    plt.legend()
    plt.title('Population fitne ss over validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 3)
    plt.plot(np.asarray(data.result_dict.get('Training Accuracy')), label='Training')
    plt.plot(np.asarray(data.result_dict.get('Validation Accuracy')), '--', label='Validation')
    plt.plot(np.asarray(data.result_dict.get('Best')), '*-', label='Best')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylabel('Accuracy[%]')
    plt.grid()
    # plt.subplot(2, 3, 3)
    # plt.plot(np.asarray(data.result_dict.get('Training Loss')))
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.subplot(2, 3, 4)
    # plt.plot(np.asarray(data.result_dict.get('LR')))
    # plt.title('Learning Rate Progress')
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.grid()
    # plt.subplot(2, 3, 5)
    # plt.plot(np.asarray(data.result_dict.get('N')))
    # plt.title('N')
    # plt.xlabel('Epoch')
    # plt.ylabel('Annealing')
    # plt.grid()
    plt.show()
    print("a")
else:
    ################
    # Build legend
    ################
    config_list = []
    param_list = []
    for f in file_list:
        config_list.append(load_config(os.path.join(f, 'config.json')))
        for k in config_list[-1].keys():
            param_list.append(k)
    param_list = np.unique(param_list)
    str_list = ['' for c in config_list]
    for p in param_list:
        if len(np.unique([c.get(p) for c in config_list if c.get(p) is not None])) > 1:
            for i, c in enumerate(config_list):
                str_list[i] = str_list[i] + ' ' + p + '=' + str(c.get(p))
        elif len(np.unique([c.get(p) for c in config_list if c.get(p) is not None])) == 1:
            if len([c.get(p) for c in config_list if c.get(p) is None]) != 0:
                for i, c in enumerate(config_list):
                    str_list[i] = str_list[i] + ' ' + p + '=' + str(c.get(p))
    #########################
    # Plot Validation
    #########################
    plt.subplot(2, 2, 1)
    for i, f in enumerate(file_list):
        data = pickle.load(open(os.path.join(f, 'ga_result.pickle'), "rb"))
        plt.plot(np.asarray(data.result_dict.get('Best')), label=str_list[i])
    # plt.title()
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 2)
    for i, f in enumerate(file_list):
        data = pickle.load(open(os.path.join(f, 'ga_result.pickle'), "rb"))
        config = load_config(os.path.join(f, 'config.json'))
        plt.plot(np.asarray(data.result_dict.get('Training Accuracy')), label=str_list[i])
        # plt.plot(np.asarray(data.result_dict.get('Validation Accuracy')), '*--', label='Validation ' + str_list[i])
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 3)
    for i, f in enumerate(file_list):
        data = pickle.load(open(os.path.join(f, 'ga_result.pickle'), "rb"))
        config = load_config(os.path.join(f, 'config.json'))
        plt.plot(np.asarray(data.result_dict.get('Training Loss')), label=str_list[i])
        # plt.plot(np.asarray(data.result_dict.get('Validation Accuracy')), '*--', label='Validation ' + str_list[i])
    plt.legend()
    plt.grid()
    plt.show()
