# Genetic Neural Architecture Search (GNAS) 
The genetic neural architecture search (GNAS) is a neural architecture search method that is based on genetic algorithm which utilized weight sharing accross all candidate network.

Includes code for CIFAR-10 and CIFAR-100 image classification 
 
# Installation
The first is install all the flowing prerequisites using conda:
* pytorch
* graphviz
* install the requirements file

```javascript
    conda install graphviz
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
    pip install -r requirements.txt
```
# Examples Run Search
In this section provide exmaple of how to run architecture search on there dataset CIFAR10 and CIFAR100 
#### CIFAR 10
```javascript
    python main.py --dataset_name CIFAR10
```
#### CIFAR 100
```javascript
    python main.py --dataset_name CIFAR100
```

# Examples Run Final Training
In this section provide exmaple of how to run architecture search on there dataset CIFAR10 and CIFAR100 
#### CIFAR 10
```javascript
    python main.py --dataset_name CIFAR10 --final 1 --serach_dir $LOG_DIR
```
#### CIFAR 100
```javascript
    python main.py --dataset_name CIFAR100 --final 1 --serach_dir $LOG_DIR
```

# Result


## CIFAR10 Counvulation Cell 
![Screenshot](images/search_result_cifar10.png)


## CIFAR100 Counvulation Cell
![Screenshot](images/search_result_cifar100.png)

## Counvulation cell final result
| Dataset | Accuracy[%] |
| --- | --- |
| CIFAR10 | 96% |
| CIFAR100 | 80! |
