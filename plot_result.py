import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from config import load_config
import pandas as pd
from scipy.ndimage.filters import maximum_filter1d

file_list = ["/data/projects/GNAS/logs/2019_01_07_22_09_57","/data/projects/GNAS/logs/2019_01_08_21_42_45","/data/projects/GNAS/logs/2019_01_01_08_25_37"]


def read_config(file_path):
    pass

if len(file_list) == 1 and False:
    data = pickle.load(open(os.path.join(file_list[0], 'ga_result.pickle'), "rb"))
    fitness = np.stack(data.result_dict.get('Fitness'))
    mode = False
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
    if mode:

        plt.subplot(2, 3, 2)
        plt.plot(epochs, np.asarray(data.result_dict.get('Loss')), label='Loss')
        plt.plot(epochs, np.min(fitness, axis=1), '*--', label='min fitness')
        plt.title('Loss over training set')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylabel('Loss')
        plt.grid()
        plt.subplot(2, 3, 3)
        plt.plot(epochs, np.asarray(data.result_dict.get('Annealing')))
        plt.title('Annealing Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Annealing')
        plt.grid()
        plt.subplot(2, 2, 4)
        plt.plot(epochs, np.asarray(data.result_dict.get('LR')))
        plt.title('Learning Rate Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid()
        plt.show()
    else:
        plt.subplot(2, 3, 2)
        plt.plot( np.asarray(data.result_dict.get('Training Accuracy')), label='Loss')
        plt.plot(epochs, np.max(fitness, axis=1), '--', label='max fitness')
        plt.title('Loss over training set')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylabel('Accuracy[%]')
        plt.grid()
        plt.subplot(2, 3, 3)
        plt.plot( np.asarray(data.result_dict.get('Training Loss')))
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.subplot(2, 3, 4)
        plt.plot( np.asarray(data.result_dict.get('LR')))
        plt.title('Learning Rate Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid()
        plt.subplot(2, 3, 5)
        plt.plot( np.asarray(data.result_dict.get('N')))
        plt.title('N')
        plt.xlabel('Epoch')
        plt.ylabel('Annealing')
        plt.grid()
        plt.show()
    print("a")
else:
    for f in file_list:
        data = pickle.load(open(os.path.join(f, 'ga_result.pickle'), "rb"))
        config=load_config(os.path.join(f,'config.json'))
        if data.result_dict.get('Fitness') is not None:
            fitness = np.stack(data.result_dict.get('Fitness'))
            epochs = np.linspace(0, fitness.shape[0] - 1, fitness.shape[0])
            if config.get('full_dataset') is None or config.get('full_dataset')==True:
                plt.plot(epochs, np.max(fitness, axis=1), '*--', label='min fitness')
                plt.plot(epochs, np.asarray(data.result_dict.get('Training Accuracy')), label='Accuracy')
            else:
                window=config.get('generation_per_epoch')
                a=np.max(fitness, axis=1)
                n=int(a.shape[0]/window)
                b=np.zeros(int(a.shape[0]/window))
                for i in range(0, int(a.shape[0]/window)):
                    b[i] = np.amax(a[window*i:window*i + window])
                epochs = np.linspace(0, n - 1, n)
                plt.plot(epochs, b, '*--', label='min fitness')
                plt.plot(epochs, np.asarray(data.result_dict.get('Training Accuracy')), label='Accuracy')
                plt.plot(epochs,np.asarray(data.result_dict.get('Best')),label='best')
        else:
            plt.plot(np.asarray(data.result_dict.get('Training Accuracy')), label='Accuracy')
            plt.plot(np.asarray(data.result_dict.get('Validation Accuracy')), '*--', label='Validation')
    plt.legend()
    plt.grid()
    plt.show()
