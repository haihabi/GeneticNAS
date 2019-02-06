# Genetic Neural Architecture Search (GNAS) 
The genetic neural architecture search (GNAS) is a neural architecture search method that is based on genetic algorithm which utilized weight sharing accross all candidate network.
 
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
In this section provide exmaple of how to run architecture search on there dataset CIFAR10, CIFAR100 and PTB  (Penn Treebank)
#### CIFAR 10
```javascript
    python main.py --dataset_name CIFAR10
```
#### CIFAR 100
```javascript
    python main.py --dataset_name CIFAR100
```
#### Penn Treebank
```javascript
    python main.py --dataset_name PTB
```
# Run Final


# Result
## Counvulation search result
## Counvulation cell final result
| Dataset | Accuracy[%] |
| --- | --- |
| CIFAR10 | 96% |
| CIFAR100 | 80! |