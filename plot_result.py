import pickle
import numpy as np
from matplotlib import pyplot as plt

data = pickle.load(open("/data/projects/GNAS/ga_result_without_bn.pickle", "rb"))
fitness = np.stack(data.result_dict.get('Fitness'))

epochs = np.linspace(0, fitness.shape[0] - 1, fitness.shape[0])
plt.subplot(2, 2, 1)
plt.errorbar(epochs, np.mean(fitness, axis=1), np.std(fitness, axis=1), label='population mean fitness')
plt.plot(epochs, np.min(fitness, axis=1), '*--', label='min fitness')
plt.plot(epochs, np.max(fitness, axis=1), '--', label='max fitness')
plt.grid()
plt.legend()
plt.title('Population fitness over validation set')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(2, 2, 2)
plt.plot(epochs, np.asarray(data.result_dict.get('Loss')))
plt.title('Loss over training set')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()
print("a")
