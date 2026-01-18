"""
Contains necessary functions to register hooks for parallel networks to
retrieve the activation of single units.

- get_hook
- get_layer_activations
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import _generate_stat_str

linear_combination = {}


def get_hook(name):
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
    names_tasks = testloader.dataset.names_tasks
    loss = {task: 0 for task in names_tasks}
    acc = {task: 0 for task in names_tasks}
    total_loss = {task: 0 for task in names_tasks}
    total_acc = {task: 0 for task in names_tasks}
    total_inputs = 0

    # Create hooks to retrieve activations
    layer_activations = {}
    num_examples = len(testloader.sampler)
    for i_layer, layer in enumerate(model.layers, 1):
        layer_name = f"layer{i_layer}"
        layer.register_forward_hook(get_hook(layer_name))
        num_neurons = layer.out_features
        layer_activations[layer_name] = np.zeros((num_examples, num_neurons))

    tqdm_bar = tqdm(
        enumerate(testloader, 1),
        total=len(testloader),
        desc="Test",
        disable=disable
    )

    acc_test = {task: np.zeros((num_examples, 1)) for task in names_tasks}

    model.eval()
    with torch.no_grad():
        for batch, (inputs, labels) in tqdm_bar:
            inputs = inputs.to(device)
            labels = labels.T.to(device)

            logits = model(inputs)
            batch_loss = 0
            batch_size = labels.shape[1]
            total_inputs += batch_size
            for task, logit, label in zip(names_tasks, logits, labels):
                probs = F.softmax(logit, dim=1)
                task_loss = criterion(logit, label)
                loss[task] += task_loss.detach() * batch_size
                acc[task] += probs.gather(1, label.view(-1, 1)).sum().item()
                batch_loss += task_loss
                total_loss[task] = loss[task] / total_inputs
                total_acc[task] = acc[task] / total_inputs

                units_idxs = slice(
                    (batch - 1) * batch_size,
                    (batch - 1) * batch_size + label.shape[0]
                )
                acc_test[task][units_idxs] = (
                    probs.gather(1, label.view(-1, 1)).cpu().numpy()
                )

            for i_layer, layer in enumerate(model.layers, 1):
                layer_name = f"layer{i_layer}"
                layer_outputs = linear_combination[layer_name]
                layer_activations[layer_name][units_idxs, :] = (
                    F.relu(layer_outputs).cpu().numpy()
                )

            loss_label = _generate_stat_str(total_loss)
            acc_label = _generate_stat_str(total_acc)

            tqdm_bar.set_postfix_str(f"loss={loss_label}, acc={acc_label}")

    return acc_test, layer_activations
