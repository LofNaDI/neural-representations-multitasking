import torch

from multitask.datasets.samplers import SequentialSampler
from torch.utils.data import SubsetRandomSampler

class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, tasks_datasets):
        self.dataset = tasks_datasets['dataset']
        self.tasks = tasks_datasets['tasks']
        self.names_tasks = list(self.tasks.keys())

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        task_labels = []
        for task_values in self.tasks.values():
            task_label = task_values[label]
            task_labels.append(task_label)
        return image, torch.LongTensor(task_labels)

    def __len__(self):
        return len(self.dataset)


def create_dataloaders(tasks_dataset, indices, batch_size):
    parallel_tasks = ParallelDataset(tasks_dataset)

    train_sampler = SubsetRandomSampler(indices['train'])
    test_sampler = SequentialSampler(indices['test'])

    trainloader = torch.utils.data.DataLoader(
        parallel_tasks, sampler=train_sampler, batch_size=batch_size
    )
    testloader = torch.utils.data.DataLoader(
        parallel_tasks, sampler=test_sampler, batch_size=batch_size
    )

    return trainloader, testloader