"""
Contains necessary functions to register hooks for task-switching networks to
retrieve the activation of single units.

- get_hook
- get_layer_activations
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import generate_stat_str

linear_combination = {}


def _get_hook(name):
    """Gets linear combinations of neurons in a layer.

    Args:
        name (str): Layer name to retrieve activations.
    """

    def hook(model, input, output):
        linear_combination[name] = output.detach()

    return hook


def get_layer_activations(model, testloader, criterion, device, disable=False):
    """Gets the activations of a neural network.

    Args:
        model (nn.Module): Neural network model.
        testloader (torch.utils.data.dataloader.DataLoader):
            Iterable dataloader with test data.
        criterion (torch.nn.modules.loss): Loss to optimize model.
        device (torch.device): Device to run the calculations (CPU or GPU).
    """
    names_tasks = testloader.names_tasks
    loss = {task: 0 for task in testloader.names_tasks}
    acc = {task: 0 for task in testloader.names_tasks}
    total_loss = {task: 0 for task in testloader.names_tasks}
    total_acc = {task: 0 for task in testloader.names_tasks}
    total_inputs = {task: 0 for task in testloader.names_tasks}

    # Create hooks to retrieve activations
    layer_activations = {}
    for name_task in testloader.names_tasks:
        layer_activations[name_task] = {}

    num_examples = testloader.num_examples
    for i_layer, layer in enumerate(model.layers, 1):
        layer_name = f"layer{i_layer}"
        layer.register_forward_hook(_get_hook(layer_name))
        num_neurons = layer.out_features

        for name_task in testloader.names_tasks:
            layer_activations[name_task][layer_name] = np.zeros(
                (num_examples, num_neurons)
            )

    current_task = None
    tqdm_bar = tqdm(
        enumerate(testloader, 1),
        total=len(testloader),
        desc="Test",
        disable=disable
    )

    acc_test = {task: np.zeros((num_examples, 1)) for task in names_tasks}

    model.eval()
    with torch.no_grad():
        for _, (task, inputs, labels) in tqdm_bar:
            if task != current_task:
                current_task = task
                # Tasks are sequential! Only reset after prior is finished.
                task_batch = 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs, task)
            probs = F.softmax(logits, dim=1)
            batch_loss = criterion(logits, labels)

            batch_size = len(labels)
            total_inputs[task] += batch_size
            loss[task] += batch_loss.detach().cpu().numpy() * batch_size
            acc[task] += probs.gather(1, labels.view(-1, 1)).sum().item()

            total_loss[task] = loss[task] / total_inputs[task]
            total_acc[task] = acc[task] / total_inputs[task]

            loss_label = generate_stat_str(total_loss)
            acc_label = generate_stat_str(total_acc)

            units_idxs = slice(
                (task_batch - 1) * batch_size,
                (task_batch - 1) * batch_size + labels.shape[0],
            )

            acc_test[task][units_idxs] = (
                probs.gather(1, labels.view(-1, 1)).cpu().numpy()
            )

            for i_layer, layer in enumerate(model.layers, 1):
                layer_name = f"layer{i_layer}"
                layer_outputs = linear_combination[layer_name]
                layer_activations[task][layer_name][units_idxs, :] = (
                    F.relu(layer_outputs).cpu().numpy()
                )

            task_batch += 1
            tqdm_bar.set_postfix_str(f"loss={loss_label}, acc={acc_label}")

    return acc_test, layer_activations
