"""
Definition of the task-switching multilayer perceptron.

- ContextLayer
- TaskMLP
- get_taskMLP
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class ContextLayer(nn.Module):
    """
    Implementation of a context layer.
    """
    def __init__(self, in_features, out_features, tasks, device):
        super().__init__()
        if tasks is not None:
            self.tasks = tasks
            self.num_tasks = len(tasks)
            for task, values in self.tasks.items():
                assert self.num_tasks == len(
                    values["activations"]
                ), f"Incorrect number of activations for task {task}."
            weight = torch.zeros((out_features, in_features + self.num_tasks))
            self.weight = Parameter(weight)
        else:
            self.tasks = None
            self.num_tasks = None
            self.weight = Parameter(torch.zeros((out_features, in_features)))
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self._initialize_weights()

    def _initialize_weights(self):
        # Equivalent to uniform (-1/sqrt(k), 1/sqrt(k)) by using the default
        # leaky relu gain calculation.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __repr__(self):
        if self.tasks is None:
            return "ContextLayer()"

        tasks_str = ", ".join(
            f'{task} '
            '{self.tasks[task]["activations"]}' for task in self.tasks.keys()
        )
        return f"ContextLayer({tasks_str})"

    def forward(self, x, active_tasks):
        if self.tasks is not None:
            activations = self.tasks[active_tasks]["activations"]
            x_ctxt = torch.tensor(activations, dtype=torch.float32).repeat(
                x.shape[0], 1
            )
            x_ctxt = x_ctxt.to(self.device)
            x = torch.cat((x, x_ctxt), dim=1)
        return F.linear(x, self.weight, bias=None)


class TaskMLP(nn.Module):
    """
    Implementation of task-switching multilayer peceptron for multitask
    learning.
    """

    def __init__(self, layers, output):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output

    def forward(self, x, task):
        for layer in self.layers:
            x = F.relu(layer(x, task))
        return self.output(x, task)


def get_task_model(tasks, num_hidden, idxs_contexts, device, activations={}):
    """
    Instantiates a task-switching multilayer perpcetron.

    Args:
        tasks (dict): Dictionary of tasks.
        num_hidden (list): Number of hidden units per layer.
        idxs_contexts (list): Indices of contexts layers with task biases.
        torch.device: Device to run the calculations (CPU or GPU).
        activations (dict, optional): Activations per layer. Defaults to {}.

    Returns:
        nn.Module: TaskMLP.
    """
    layers = []
    last_units = None

    for i_layer, num_units in enumerate(num_hidden):
        layer_tasks = tasks.copy() if i_layer in idxs_contexts else None

        if activations.get(f"layer{i_layer + 1}", None):
            for task in layer_tasks.keys():
                task_dict = layer_tasks[task].copy()
                task_dict["activations"] = activations[f"layer{i_layer + 1}"]
                layer_tasks[task] = task_dict

        if i_layer == 0:
            layer = ContextLayer(784,
                                 num_units,
                                 tasks=layer_tasks,
                                 device=device)
        else:
            layer = ContextLayer(last_units,
                                 num_units,
                                 tasks=layer_tasks,
                                 device=device)
        last_units = num_units
        layers.append(layer)
    output = ContextLayer(last_units, 2, tasks=None, device=device)

    model = TaskMLP(layers, output)
    model = model.to(device)
    return model
