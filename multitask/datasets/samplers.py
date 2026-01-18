import torch
import numpy as np


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (index for index in self.indices)

    def __len__(self):
        return len(self.indices)


def split_dataset_train_test(tasks_dataset, num_train, num_test):
    num_images = len(tasks_dataset['dataset'])
    list_indices = list(range(num_images))[:num_train + num_test]
    np.random.shuffle(list_indices)
    train_idx, test_idx = list_indices[num_test:], list_indices[:num_test]
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    indices = {'train': train_idx, 'test': test_idx}
    return indices


def split_indices_tasks(tasks_dataset, indices, partition):
    num_tasks = len(tasks_dataset['tasks'])
    train_idxs = indices['train']
    test_idxs  = indices['test']

    new_indices = {}
    new_indices['train'] = {}
    new_indices['test'] = {}

    if partition:
        train_shards = np.array_split(train_idxs, num_tasks)
    else:
        train_shards = [train_idxs] * num_tasks

    for task_name, split in zip(tasks_dataset['tasks'].keys(), train_shards):
        new_indices['train'][task_name] = split
        new_indices['test'][task_name] = test_idxs

    return new_indices