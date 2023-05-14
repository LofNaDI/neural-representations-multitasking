"""
Definition of the parallel multilayer perceptron.

- ParallelMLP
- get_individualMLP
"""
import torch.nn.functional as F
from torch import nn


class ParallelMLP(nn.Module):
    """
    Implementation of parallel multilayer peceptron for multitask learning.
    """

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


def get_parallel_model(num_binary_tasks, num_hidden, device):
    """
    Instantiates a parallel multilayer perpcetron.

    Args:
        num_binary_tasks (int): Number of binary tasks for the output layer.
        num_hidden (list): Number of hidden units per layer.
        torch.device: Device to run the calculations (CPU or GPU).

    Returns:
        nn.Module: ParallelMLP.
    """
    layers = []
    last_units = None

    for i_layer, num_units in enumerate(num_hidden):
        if i_layer == 0:
            layer = nn.Linear(784, num_units)
        else:
            layer = nn.Linear(last_units, num_units)
        last_units = num_units
        layers.append(layer)

    outputs = []
    for _ in range(num_binary_tasks):
        outputs.append(nn.Linear(last_units, 2).to(device))

    model = ParallelMLP(layers, outputs)
    model = model.to(device)
    return model
