import torch
import torchvision


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(download=True, train=True, *args, **kwargs)
        self.numbers = self.targets
        self._normalize_data()

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        return image, label

    def __len__(self):
        return len(self.data)

    def _normalize_data(self):
        self.data = self.data.float()
        self.data = self.data.div(255)
        self.data = self.data.view(self.data.shape[0], -1)
        self.data = torch.FloatTensor(self.data)


class EMNIST(torchvision.datasets.EMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(download=True, train=True, split='byclass', *args, **kwargs)
        self.samples_per_class = 1896
        self._normalize_data()
        self._filter_lowercase()
        self._subsample()
        self.letters = self.targets

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        return image, label

    def __len__(self):
        return len(self.data)

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
   

_DATASET_REGISTRY = {
    'MNIST': MNIST,
    'EMNIST': EMNIST,
}


def get_dataset(dataset_name, root, **kwargs):
    key = dataset_name.lower()
    try:
        dataset_class = _DATASET_REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown dataset {dataset_name!r}; "
                         f"valid options are {list(_DATASET_REGISTRY)}")
    return dataset_class(root=root, **kwargs)


def get_tasks_dataset(exp):
    dataset_name = exp.dataset.name
    assert dataset_name in _DATASET_REGISTRY

    dataset = _DATASET_REGISTRY[dataset_name](root=exp.dataset.root)
    unique_labels = set(dataset.targets.numpy())

    task_dict = exp.tasks
    for _, task_values in task_dict.items():
        assert len(task_values) == len(unique_labels)

    tasks_dataset = {
        'dataset': dataset,
        'tasks': task_dict
    }

    return tasks_dataset
