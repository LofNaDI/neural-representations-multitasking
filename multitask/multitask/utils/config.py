from pathlib import Path
from typing import Annotated, Dict, Literal, List, Union

from pydantic import BaseModel, PositiveInt, Field


class DatasetConfig(BaseModel):
    name: Literal['MNIST', 'EMNIST']
    root: str = 'data'


class CommonParameters(BaseModel):
    initial_seed: int
    model: Literal['sequential', 'parallel', 'task_switching']
    num_runs: PositiveInt
    num_epochs: PositiveInt
    batch_size: PositiveInt
    num_train: PositiveInt
    num_test: PositiveInt
    num_hidden: List[int]
    dataset: DatasetConfig
    results: str = 'root'
    tasks: Dict[str, List[int]]

    def pretty(self):
        return "\n".join(f"- {k}: {v}" for k, v in self.model_dump().items()) + '\n'
    
    def to_json(self):
        return self.model_dump_json(indent=4)
    
    def save_json(self, path):
        Path(path).write_text(self.to_json())

    def find_existing_run(self, root):
        root = Path(root)
        for sub in root.iterdir():
            param_file = sub / 'parameters.json'
            if not param_file.exists():
                continue
            try:
                other = self.__class__.model_validate_json(param_file.read_text())
            except Exception:
                continue
            if other == self:
                return sub
        return None


class SequentialParameters(CommonParameters):
    model: Literal['sequential']
    partition: bool


class ParallelParameters(CommonParameters):
    model: Literal['parallel']


class TaskSwitchingParameters(CommonParameters):
    model: Literal['task_switching']
    scheduler: Literal['random', 'sequential', 'cycles']
    idx_context: List[int]
    partition: bool


ExperimentsCfg = Annotated[
    Union[SequentialParameters, ParallelParameters, TaskSwitchingParameters],
    Field(discriminator='model')
]

class ExperimentsConfig(BaseModel):
    experiments: List[ExperimentsCfg]
    