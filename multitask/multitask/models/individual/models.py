"""
Definition of the individual multilayer perceptron.

- IndividualMLP
- get_individual_model
"""
import torch.nn.functional as F
from torch import nn


class IndividualMLP(nn.Module):
    """
    Implementation of individual multilayer peceptron for multitask learning.
    """

    def __init__(self, layers, output):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)


def get_individual_model(num_hidden, device):
    """
    Instantiates an individual multilayer perpcetron.

    Args:
        num_hidden (list): Number of hidden units per layer.
        torch.device: Device to run the calculations (CPU or GPU).

    Returns:
        nn.Module: IndividualMLP.
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
    output = nn.Linear(last_units, 2)

    model = IndividualMLP(layers, output)
    model = model.to(device)
    return model
