import numpy as np


class TaskScheduler:
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
    

class RandomTaskScheduler(TaskScheduler):
    def __init__(self, dict_dataloaders):
        super().__init__(dict_dataloaders)

    def _initialize_blocks(self):
        blocks_tmp = []
        for task_name in self.names_tasks:
            blocks_tmp.extend([task_name] * self.lengths[task_name])
        self.blocks = np.array(blocks_tmp)
        np.random.shuffle(self.blocks)


class SequentialTaskScheduler(TaskScheduler):
    def __init__(self, dict_dataloaders):
        super().__init__(dict_dataloaders)

    def _initialize_blocks(self):
        if self.blocks is None:
            blocks_tmp = []
            for task, dataloader in self.dict_dataloaders.items():
                blocks_tmp.extend([task] * len(dataloader))
            self.blocks = np.array(blocks_tmp)


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


_SCHEDULER_REGISTRY = {
    'random': RandomTaskScheduler,
    'sequential': SequentialTaskScheduler,
    'cycles': CycleTaskScheduler
}


def get_scheduler(scheduler_name):
    if scheduler_name not in _SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{scheduler_name}' is not registered. Available schedulers: {list(_SCHEDULER_REGISTRY.keys())}")
    return _SCHEDULER_REGISTRY[scheduler_name]