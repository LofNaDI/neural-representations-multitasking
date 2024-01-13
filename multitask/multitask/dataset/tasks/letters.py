"""
Defines letter tasks using EMNIST.
"""

import os

import torch
import torchvision


class LetterTask(torchvision.datasets.EMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(download=True, train=True, split='byclass',
                         *args, **kwargs)
        self.samples_per_class = 1896
        self._normalize_data()  # Between 0 and 1
        self._filter_lowercase()
        self._subsample()
        self.letters = self.targets
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

    def _filter_lowercase(self):
        mask_lowercase = self.targets > 35  # 'a' starts at 36
        self.data = self.data[mask_lowercase]
        self.targets = self.targets[mask_lowercase] - 36  # 0 to 25

    def _subsample(self):
        class_indices = \
            {label.item(): [] for label in torch.unique(self.targets)}

        for idx, label in enumerate(self.targets):
            class_indices[label.item()].append(idx)

        balanced_indices = []
        for label, indices in class_indices.items():
            len_indices = len(indices[:self.samples_per_class])
            assert len_indices == self.samples_per_class
            balanced_indices.extend(indices[:self.samples_per_class])

        balanced_indices = sorted(balanced_indices)

        self.data = self.data[balanced_indices]
        self.targets = self.targets[balanced_indices]

    def _normalize_data(self):
        self.data = self.data.float()
        self.data = self.data.div(255)
        self.data = torch.transpose(self.data, 1, 2).contiguous()
        self.data = self.data.view(self.data.shape[0], -1)
        self.data = torch.FloatTensor(self.data)
        self.data = self.data

    def define_task():
        raise NotImplementedError


class VowelTask(LetterTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        vowels = [0, 4, 8, 14, 20]
        self.targets = [int(letter in vowels) for letter in self.letters]
        self.targets = torch.LongTensor(self.targets)


class PositionTask(LetterTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_task()

    def define_task(self):
        self.targets = [int(letter < 13) for letter in self.letters]
        self.targets = torch.LongTensor(self.targets)


class LetterTaskFactory:
    _tasks = {
        'vowel': VowelTask,
        'position': PositionTask
    }

    @staticmethod
    def list_tasks():
        return LetterTaskFactory._tasks.keys()

    @staticmethod
    def get_task(task_name):
        LetterTaskClass = LetterTaskFactory._tasks.get(task_name)
        if LetterTaskClass:
            return LetterTaskClass
        raise ValueError(f'{task_name} not in LetterTaskFactory')
