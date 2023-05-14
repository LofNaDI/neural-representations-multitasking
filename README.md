# Neural Representations in Multi-Task Learning guided by Task-Dependent Contexts


## Installation
### Install conda environment
```
conda env create -f multitask/environment.yml
```

### Pip install multitask package
```
pip install multitask/
```

### Train a multi-task neural network
Edit train_tasks.sh with the neural networks to train (individual, parallel,
task-switching) and their parameters:

- `num_runs`: Number of different simulations to run. For statistics.
- `initial_seed`: Main seed to generate the seeds for the different runs.
- `max_seed`: Maximum seed number to generate.
- `num_epochs`: Number of total epochs.
- `num_hidden`: Number of hidden units per layer. The total number of layers is given by the length of the number of hidden units.
- `batch_size`: Batch size for train and test set.
- `num_train`: Number of examples for train.
- `num_test`: Number of examples for test.
- `tasks`: Names of the tasks. Currently, the only tasks supported are `parity`, `value`, `prime`, `fibonacci`, `multiples_3`
- `idxs_contexts`: Indices of the context layers. Only for `task-switching` networks. 

### Generate the figures
Run the corresponding `jupyter notebook` in `figures` to generate the plots.