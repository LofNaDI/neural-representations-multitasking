"""
Definition of the parallel multilayer perceptron.

- ParallelMLP
- get_model
"""
import torch.nn.functional as F
from torch import nn


class ParallelMLP(nn.Module):
    def __init__(self, layers, outputs):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.outputs = nn.ModuleList(outputs)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        outputs = []
        for output_layer in self.outputs:
            outputs.append(output_layer(x))
        return outputs

    def __str__(self):
        return f"{self.layers}\n{self.outputs}"


def get_model(tasks_dataset, num_hidden):
    layers = []
    input_size = 784
    num_tasks = len(tasks_dataset['tasks'])

    for num_units in num_hidden:
        layers.append(nn.Linear(input_size, num_units))
        input_size = num_units

    outputs = nn.ModuleList([
        nn.Linear(num_hidden[-1], 2) for _ in range(num_tasks)
    ])

    model = ParallelMLP(nn.ModuleList(layers), outputs)
    return model