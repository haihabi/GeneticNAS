import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# "/data/projects/GNAS/logs/2018_12_26_08_42_53",
# "/data/projects/GNAS/logs/2018_12_26_20_58_39",
#              "/data/projects/GNAS/logs/2018_12_27_07_41_06",
#              "/data/projects/GNAS/logs/2018_12_27_15_23_01",
file_list = ["/data/projects/GNAS/logs/2018_12_28_02_03_51",
             "/data/projects/GNAS/logs/2018_12_27_15_23_01"]
if len(file_list) == 1:
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
        plt.plot(epochs, np.asarray(data.result_dict.get('Training Accuracy')), label='Loss')
        plt.plot(epochs, np.max(fitness, axis=1), '--', label='max fitness')
        plt.title('Loss over training set')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylabel('Accuracy[%]')
        plt.grid()
        plt.subplot(2, 3, 3)
        plt.plot(epochs, np.asarray(data.result_dict.get('Training Loss')))
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.subplot(2, 3, 4)
        plt.plot(epochs, np.asarray(data.result_dict.get('LR')))
        plt.title('Learning Rate Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid()
        plt.subplot(2, 3, 5)
        plt.plot(epochs, np.asarray(data.result_dict.get('Annealing')))
        plt.title('Annealing Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Annealing')
        plt.grid()
        plt.show()
    print("a")
else:
    for f in file_list:
        data = pickle.load(open(os.path.join(f, 'ga_result.pickle'), "rb"))
        fitness = np.stack(data.result_dict.get('Fitness'))
        epochs = np.linspace(0, fitness.shape[0] - 1, fitness.shape[0])
        plt.plot(epochs, np.max(fitness, axis=1), '*--', label='min fitness')
        plt.plot(epochs, np.asarray(data.result_dict.get('Training Accuracy')), label='Accuracy')
    plt.grid()
    plt.show()
