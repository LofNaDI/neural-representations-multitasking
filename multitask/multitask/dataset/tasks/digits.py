"""
Defines digit tasks using MNIST.
"""

import os

import torch
import torchvision


class DigitTask(torchvision.datasets.MNIST):
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


class ParityTask(DigitTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        self.targets = [int(not number % 2) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class ValueTask(DigitTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        self.targets = [int(number < 5) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class FibonacciTask(DigitTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        fibonacci = [0, 1, 2, 3, 5, 8]
        self.targets = [int(number in fibonacci) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class PrimeTask(DigitTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        primes = [2, 3, 5, 7]
        self.targets = [int(number in primes) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class Multiples3Task(DigitTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        multiples_3 = [3, 6, 9]
        self.targets = [int(number in multiples_3) for number in self.numbers]
        self.targets = torch.LongTensor(self.targets)


class DigitTaskFactory:
    _tasks = {
        'parity': ParityTask,
        'value': ValueTask,
        'prime': PrimeTask,
        'fibonacci': FibonacciTask,
        'multiples3': Multiples3Task,
    }

    @staticmethod
    def list_tasks():
        return DigitTaskFactory._tasks.keys()

    @staticmethod
    def get_task(task_name):
        DigitTaskClass = DigitTaskFactory._tasks.get(task_name)
        if DigitTaskClass:
            return DigitTaskClass
        raise ValueError(f'{task_name} not in DigitTaskClass')
