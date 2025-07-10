"""
Definition of the sequential multilayer perceptron.

- SequentiallMLP
- get_model
"""
import torch.nn.functional as F
from torch import nn


class SequentialMLP(nn.Module):
    def __init__(self, layers, output):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)


def get_model(tasks_dict, num_hidden):
    models = {}
    for task_name in tasks_dict['tasks']:
        layers = []
        input_size = 784

        for num_units in num_hidden:
            layers.append(nn.Linear(input_size, num_units))
            input_size = num_units

        output = nn.Linear(input_size, out_features=2)

        model = SequentialMLP(layers, output)
        models[task_name] = model

    return models