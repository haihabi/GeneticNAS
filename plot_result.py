import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from common import load_final, make_log_dir, get_model_type, ModelType
from config import get_config, load_config, save_config
import gnas
from modules.drop_module import DropModuleControl
from gnas.common.graph_draw import draw_cell, draw_network
import matplotlib.image as mpimg

# Popultation size compare
file_list = ["/data/projects/gnas_results/p_mutation/2019_01_24_19_06_00",
             '/data/projects/gnas_results/population_size/2019_01_31_15_45_42',
             '/data/projects/gnas_results/population_size/2019_02_01_03_26_01',
             '/data/projects/gnas_results/population_size/2019_02_01_04_23_06',
             '/data/projects/gnas_results/population_size/2019_02_01_04_44_45',
             '/data/projects/gnas_results/population_size/2019_02_01_15_39_37',
             '/data/projects/gnas_results/population_size/2019_02_03_17_25_25',
             # '/data/projects/gnas_results/population_size/2019_02_03_17_25_27',
             '/data/projects/gnas_results/population_size/2019_02_03_17_25_28']

# p mutation
file_list = ["/data/projects/gnas_results/p_mutation/2019_01_23_21_20_33",
             "/data/projects/gnas_results/p_mutation/2019_01_24_19_06_00",
             "/data/projects/gnas_results/p_mutation/2019_01_25_08_46_39",
             "/data/projects/gnas_results/p_mutation/2019_01_26_13_18_17"]

# # LR Compare
file_list = ["/data/projects/gnas_results/p_mutation/2019_01_24_19_06_00",
             "/data/projects/gnas_results/lr_compare/2019_02_04_19_17_59",
             "/data/projects/gnas_results/lr_compare/2019_02_04_19_18_00"]

# # Bit Vs Block
file_list = ["/data/projects/gnas_results/p_mutation/2019_01_24_19_06_00",
             "/data/projects/GNAS/logs/2019_02_11_06_15_10"]
#
# # Plot CIFAR10 - Search Result
file_list = ["/data/projects/gnas_results/p_mutation/2019_01_24_19_06_00"]
# # Plot CIFAR100 - Search Result
file_list = ["/data/projects/GNAS/logs/2019_02_17_20_25_42"]

# CIFAR10 Final
file_list = ['/data/projects/gnas_results/new_log/2019_02_09_16_43_02']
# file_list=['/data/projects/gnas_results/new_log/2019_02_07_18_34_45',
#            '/data/projects/gnas_results/new_log/2019_02_09_02_23_52',
#            '/data/projects/gnas_results/new_log/2019_02_09_16_43_02',
#            '/data/projects/gnas_results/new_log/2019_02_14_18_15_48']


plot_arc = False
# file_list = ["/data/projects/gnas_results/p_mutation/2019_01_24_19_06_00", ]
if plot_arc:
    ind_file = os.path.join(file_list[0], 'best_individual.pickle')
    config_file = os.path.join(file_list[0], 'config.json')
    ind = pickle.load(open(ind_file, "rb"))

    config = get_config(ModelType.CNN)
    print("Loading config file:" + config_file)
    config.update(load_config(config_file))

    dp_control = DropModuleControl(config.get('drop_path_keep_prob'))
    n_cell_type = gnas.SearchSpaceType(config.get('n_block_type') - 1)
    ss = gnas.get_gnas_cnn_search_space(config.get('n_nodes'), dp_control, n_cell_type)
    draw_network(ss, ind, './')
    title_list = ['Reduce Cell', ' Normal Cell', ' Input Cell']
    for i in range(len(ss.ocl)):
        plt.subplot(1, len(ss.ocl), i + 1)
        img = mpimg.imread(os.path.join('./', str(i) + '.png'))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title_list[i])
    plt.show()
    # draw_cell(ss.ocl[0], ind.individual_list[0])
    # plt.show()
    # print("a")

if len(file_list) == 1 and True:
    data = pickle.load(open(os.path.join(file_list[0], 'ga_result.pickle'), "rb"))
    config = load_config(os.path.join(file_list[0], 'config.json'))
    if data.result_dict.get('Fitness') is None:
        plt.plot(np.asarray(data.result_dict.get('Training Accuracy')), label='Training Accuracy')
        plt.plot(np.asarray(data.result_dict.get('Validation Accuracy')), label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylabel('Accuracy[%]')
        plt.grid()
        plt.show()
    else:
        fitness = np.stack(data.result_dict.get('Fitness'))
        fitness_p = np.stack(data.result_dict.get('Fitness-Population'))
        fitness_p = fitness_p[0:-1:2, :]

        epochs = np.linspace(0, fitness_p.shape[0] - 1, fitness_p.shape[0])
        plt.plot(epochs, np.mean(fitness_p, axis=1), '*--',
                 label='Population mean accuracy')
        plt.plot(epochs, np.max(fitness_p, axis=1), label='Max accuracy')
        plt.plot(np.asarray(data.result_dict.get('Best')), '--', label='Best')
        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

        plt.errorbar(epochs, np.mean(fitness_p, axis=1), np.std(fitness_p, axis=1), fmt='*--',
                     label='Population mean accuracy')
        plt.plot(epochs, np.min(fitness_p, axis=1), label='Min accuracy')
        plt.plot(epochs, np.max(fitness_p, axis=1), label='Max accuracy')
        plt.grid()
        plt.legend()
        plt.title('Population accuracy on the validation set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

        plt.plot(np.asarray(data.result_dict.get('Training Accuracy')), label='Training')
        plt.plot(np.asarray(data.result_dict.get('Validation Accuracy')), '--', label='Validation')
        plt.plot(np.asarray(data.result_dict.get('Best')), '*-', label='Best')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylabel('Accuracy[%]')
        plt.grid()
        plt.show()

        plt.plot(epochs, data.result_dict.get('N'))
        plt.title('Number of new individuals in Population')
        plt.xlabel('Epoch')
        plt.ylabel('N')
        plt.grid()
        plt.show()

        plt.plot(epochs, data.result_dict.get('Training Loss'))
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()

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
    res_dict = dict()
    for p in param_list:
        if len(np.unique([c.get(p) for c in config_list if c.get(p) is not None])) > 1:
            for i, c in enumerate(config_list):
                str_list[i] = str_list[i] + ' ' + p + '=' + str(c.get(p))
                if res_dict.get(p) is None:
                    res_dict.update({p: [c.get(p)]})
                else:
                    res_dict.get(p).append(c.get(p))
        elif len(np.unique([c.get(p) for c in config_list if c.get(p) is not None])) == 1:
            if len([c.get(p) for c in config_list if c.get(p) is None]) != 0:
                for i, c in enumerate(config_list):
                    str_list[i] = str_list[i] + ' ' + p + '=' + str(c.get(p))
    if len(res_dict.keys()) == 1:
        param_array = np.asarray(res_dict.get(list(res_dict.keys())[0]))
        res_list = []
        for i, f in enumerate(file_list):
            data = pickle.load(open(os.path.join(f, 'ga_result.pickle'), "rb"))
            res_list.append(np.max(np.asarray(data.result_dict.get('Best'))))
        index = np.argsort(param_array)
        res_list = np.asarray(res_list)[index]
        param_array = param_array[index]
        plt.plot(param_array, res_list)
        plt.grid()
        plt.xlabel(list(res_dict.keys())[0].replace('_', ' '))
        plt.ylabel('Accuracy[%]')
        plt.show()
        print("a")
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
