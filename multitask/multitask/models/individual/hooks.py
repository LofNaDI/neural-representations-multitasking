"""
Contains necessary functions to register hooks for individual networks to
retrieve the activation of single units.

- _get_layer_hook
- get_activations
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
    loss = 0
    acc = 0
    total_loss = 0
    total_acc = 0
    total_inputs = 0

    # Create hooks to retrieve activations
    layer_activations = {}
    num_examples = len(testloader.sampler)
    for i_layer, layer in enumerate(model.layers, 1):
        layer_name = f"layer{i_layer}"
        layer.register_forward_hook(_get_hook(layer_name))
        num_neurons = layer.out_features
        layer_activations[layer_name] = np.zeros((num_examples, num_neurons))

    tqdm_bar = tqdm(
        enumerate(testloader, 1),
        total=len(testloader),
        desc="Test",
        disable=disable
    )

    acc_test = np.zeros((num_examples, 1))

    model.eval()
    with torch.no_grad():
        for batch, (inputs, labels) in tqdm_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            batch_loss = criterion(logits, labels)

            batch_size = len(labels)
            total_inputs += batch_size
            loss += batch_loss.detach() * batch_size
            acc += probs.gather(1, labels.view(-1, 1)).sum().item()

            total_loss = loss / total_inputs
            total_acc = acc / total_inputs

            units_idxs = slice(
                (batch - 1) * batch_size,
                (batch - 1) * batch_size + labels.shape[0]
            )

            for i_layer, layer in enumerate(model.layers, 1):
                layer_name = f"layer{i_layer}"
                layer_outputs = linear_combination[layer_name]
                layer_activations[layer_name][units_idxs, :] = (
                    F.relu(layer_outputs).cpu().numpy()
                )

            acc_test[units_idxs] = \
                probs.gather(1, labels.view(-1, 1)).cpu().numpy()
            tqdm_str = f"loss={total_loss:.4f}, " f"acc={total_acc:.4f}"
            tqdm_bar.set_postfix_str(tqdm_str)

    return acc_test, layer_activations
