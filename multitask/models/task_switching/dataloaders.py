import torch
import numpy as np


class TaskSwitchingDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, name, mask, indices):
        self.base_dataset = base_dataset
        self.name = name
        self.mask = mask
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        x, digit = self.base_dataset[base_idx]
        return x, self.mask[digit]

    def __repr__(self):
        return f"{self.name}: {len(self)} samples, targets={self.mask}"


def create_dataloaders(tasks_dataset, indices, batch_size):
    train_loaders = {}
    test_loaders  = {}

    for task_name, task_target in tasks_dataset['tasks'].items():
        dataset_train = TaskSwitchingDataset(tasks_dataset['dataset'],
                                             task_name,
                                             task_target,
                                             indices['train'][task_name])
        train_loaders[task_name] = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        dataset_test = TaskSwitchingDataset(tasks_dataset['dataset'],
                                            task_name,
                                            task_target,
                                            indices['test'][task_name])
        test_loaders[task_name] = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

    return train_loaders, test_loaders