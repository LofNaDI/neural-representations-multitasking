"""
Defines multitasking for MNIST and implements tasks.
"""

import os

import torch
import torchvision


class TaskMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(download=True, train=True, *args, **kwargs)
        self.numbers = self.targets
        self._normalize_data()  # Between 0 and 1
        self.define_task()

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        return image, label

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, __class__.__name__)

    def _normalize_data(self):
        self.data = self.data.float()
        self.data = self.data.div(255)
        self.data = self.data.view(self.data.shape[0], -1)
        self.data = torch.FloatTensor(self.data)

    def define_task():
        raise NotImplementedError


class NumberTask(TaskMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        self.target = self.numbers


class ParityTask(TaskMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        self.targets = [int(not number % 2) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class ValueTask(TaskMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        self.targets = [int(number < 5) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class FibonacciTask(TaskMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        fibonacci = [0, 1, 2, 3, 5, 8]
        self.targets = [int(number in fibonacci) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class PrimeTask(TaskMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        primes = [2, 3, 5, 7]
        self.targets = [int(number in primes) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class Multiples3Task(TaskMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        multiples_3 = [3, 6, 9]
        self.targets = [int(number in multiples_3) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


TASKS = {
    "parity": ParityTask,
    "value": ValueTask,
    "prime": PrimeTask,
    "fibonacci": FibonacciTask,
    "multiples_3": Multiples3Task,
}


def get_tasks_dict(tasks_list, root):
    assert len(tasks_list) == len(set(tasks_list))

    for task_name in tasks_list:
        assert task_name in TASKS

    tasks_dict = {}
    for task_name in tasks_list:
        tasks_dict[task_name] = TASKS[task_name](root)

    return tasks_dict
