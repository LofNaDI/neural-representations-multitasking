"""
Definition of the task-switching multilayer perceptron.

- ContextLayer
- TaskMLP
- get_model
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

    

class ContextLayer(nn.Module):
    def __init__(self, in_features, out_features, tasks):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tasks = tasks

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if tasks:
            self.biases = nn.ParameterDict({
                task: Parameter(torch.zeros(out_features))
                for task in tasks
            })

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.tasks:
            for b in self.biases.values():
                nn.init.zeros_(b)
    
    def add_task(self, task, bias=None):
        if task in self.tasks:
            raise ValueError(f"Task '{task}' already exists in the layer.")
        
        self.tasks.append(task)
        self.biases[task] = Parameter(torch.zeros(self.out_features)) if bias is None else Parameter(bias)

    def forward(self, x, active_task):
        if not self.tasks or active_task is None:
            return F.linear(x, self.weight, bias=None)

        if active_task not in self.tasks:
            raise ValueError(f"Task '{active_task}' not found in available tasks: {self.tasks}")
    
        bias = self.biases[active_task]
        return F.linear(x, self.weight, bias=bias)

    def __repr__(self):
        if self.tasks:
            return f"{self.__class__.__name__}(in={self.in_features}, out={self.out_features}, tasks={self.tasks})"
        else:
            return f"{self.__class__.__name__}(in={self.in_features}, out={self.out_features})"


class TaskMLP(nn.Module):
    def __init__(self, layers, output):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output

    def add_task(self, task, bias=None):
        for layer in self.layers:
            layer.add_task(task, bias)

    def forward(self, x, task):
        for layer in self.layers:
            x = F.relu(layer(x, task))
        return self.output(x, task)
    

def get_model(tasks_dataset, num_hidden, idxs_contexts):
    layers = []
    in_features = 784

    for i, hidden_size in enumerate(num_hidden):
        task_names = list(tasks_dataset['tasks'].keys()) if i in idxs_contexts else []
        layer = ContextLayer(
            in_features=in_features,
            out_features=hidden_size,
            tasks=task_names,
        )
        layers.append(layer)
        in_features = hidden_size

    output = ContextLayer(
        in_features=in_features,
        out_features=2,
        tasks=[],
    )

    model = TaskMLP(layers=layers, output=output)
    return model


def load_task_model(model, state_dict):
    model.load_state_dict(state_dict)
    return model
