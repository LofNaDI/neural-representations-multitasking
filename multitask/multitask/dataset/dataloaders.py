"""
Implements utils for training multitask learning networks.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


class TaskDataLoader:
    def __init__(self, dict_dataloaders):
        self.dict_dataloaders = dict_dataloaders
        self.names_tasks = list(dict_dataloaders.keys())
        self.num_examples = len(dict_dataloaders[self.names_tasks[0]].sampler)
        self.num_tasks = len(self.names_tasks)
        self.blocks = None
        self.lengths = {
            task: len(dataloader)
            for task, dataloader in self.dict_dataloaders.items()
        }

    def __iter__(self):
        self._initialize_blocks()
        iterators = {
            task: iter(dataloader)
            for task, dataloader in self.dict_dataloaders.items()
        }
        for block in self.blocks:
            image, label = next(iterators[block])
            yield block, image, label

    def __str__(self):
        return f"{self.__class__.__name__}\n"

    def __len__(self):
        return sum(self.lengths.values())

    def _initialize_blocks(self):
        raise NotImplementedError


class RandomTaskDataloader(TaskDataLoader):
    def __init__(self, dict_dataloaders):
        super().__init__(dict_dataloaders)

    def _initialize_blocks(self):
        blocks_tmp = []
        for task_name in self.names_tasks:
            blocks_tmp.extend([task_name] * self.lengths[task_name])
        self.blocks = np.array(blocks_tmp)
        np.random.shuffle(self.blocks)


class SequentialTaskDataloader(TaskDataLoader):
    def __init__(self, dict_dataloaders):
        super().__init__(dict_dataloaders)

    def _initialize_blocks(self):
        if self.blocks is None:
            blocks_tmp = []
            for task, dataloader in self.dict_dataloaders.items():
                blocks_tmp.extend([task] * len(dataloader))
            self.blocks = np.array(blocks_tmp)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (index for index in self.indices)

    def __len__(self):
        return len(self.indices)


class MultilabelTasks(torch.utils.data.Dataset):
    def __init__(self, tasks_datasets):
        self.num_tasks = len(tasks_datasets)
        self.names_tasks = list(tasks_datasets.keys())
        self.tasks_datasets = list(tasks_datasets.values())
        self._check_datasets()

    def __getitem__(self, idx):
        image, _ = self.tasks_datasets[0][idx]
        labels = []
        for dataset in self.tasks_datasets:
            _, label = dataset[idx]
            labels.append(label)
        return image, torch.LongTensor(labels)

    def __len__(self):
        return len(self.tasks_datasets[0])

    def _check_datasets(self):
        for i_task in range(self.num_tasks, 1):
            data1 = self.tasks_datasets[i_task - 1].data
            data2 = self.tasks_datasets[i_task].data
            targets1 = self.tasks_datasets[i_task - 1].targets
            targets2 = self.tasks_datasets[i_task].targets
            assert torch.equal(data1, data2)
            assert torch.equal(targets1, targets2)


class CycleTaskScheduler:
    def __init__(self, tasks_trainloaders, tasks_validloaders, num_cycles):
        self.num_cycles = num_cycles
        self.task_dataloaders = tasks_trainloaders
        self.tasks_validloaders = tasks_validloaders

    def __iter__(self):
        for cycle in range(self.num_cycles):
            for task_dataloader in self.task_trainloaders:
                yield next(task_dataloader)
            for task_dataloader in self.tasks_validloaders:
                yield next(task_dataloader)


def get_split_indices(num_train, num_test):
    list_indices = list(range(num_train + num_test))
    np.random.shuffle(list_indices)
    train_idx, valid_idx = list_indices[num_test:], list_indices[:num_test]
    indices = {"train": train_idx, "test": valid_idx}
    return indices


def create_dict_dataloaders(tasks, indices, batch_size):
    train_dataloaders, test_datalaoders = {}, {}
    train_sampler = torch.utils.data.SubsetRandomSampler(indices["train"])
    test_sampler = SequentialSampler(indices["test"])
    for task_name, value in tasks.items():
        trainloader = DataLoader(
            value["data"], batch_size=batch_size, sampler=train_sampler
        )
        testloader = DataLoader(
            value["data"], batch_size=batch_size, sampler=test_sampler
        )
        train_dataloaders[task_name] = trainloader
        test_datalaoders[task_name] = testloader
    return train_dataloaders, test_datalaoders
