"""
Functions to generate and plot the Representation Dissimilarity Matrix (RDM)
for individual networks.

- get_mean_activations
- calculate_rdm
- plot_rdm
"""
import numpy as np
import seaborn as sns

NUM_DIGITS = 10  # TODO: Add num_classes as parameter instead


def _get_mean_activations(activations, num_hidden, list_numbers):
    """
    Calculates the mean_activation matrix per layer.

    Args:
        activations (dict): Dictionary of np.ndarray containing the matrix
            of activations (N_test x N_hidden_units).
        num_hidden (list): Number of hidden units per layer.
        list_numbers (list): List of numbers of test set (N_test, ).

    Returns:
        dict: Mean of activations per layer.
    """
    mean_activations = {}
    for i_layer, num_units in enumerate(num_hidden, 1):
        layer_name = f"layer{i_layer}"
        mean_activations[layer_name] = np.zeros((NUM_DIGITS, num_units))
        for number in range(NUM_DIGITS):
            activations_number = \
                activations[layer_name][list_numbers == number, :]
            mean_activations[layer_name][number, :] = np.mean(
                activations_number, axis=0
            )

    return mean_activations


def calculate_rdm(activations_tasks, names_tasks, num_hidden, list_numbers):
    """
    Calculates the Representational Dissimilarity Matrix (RDM) per layer.

    Args:
        activations_tasks (list): List of dictionaries of activations per task.
        names_tasks (list): Name of tasks (N_tasks).
        num_hidden (list): Number of hidden units per layer.
        list_numbers (list): List of numbers of test set (N_test, ).

    Returns:
        dict: Dictionary of RDMs per layer.
    """
    mean_activations_tasks = {}

    for name_task, activations, labels_numbers in zip(
        names_tasks, activations_tasks, list_numbers
    ):
        mean_activations_tasks[name_task] = _get_mean_activations(
            activations, num_hidden, labels_numbers
        )

    rdm_dict = {}
    for i_layer, _ in enumerate(num_hidden, 1):
        layer_name = f"layer{i_layer}"

        mean_activations_layer = None
        for name_task in names_tasks:
            if mean_activations_layer is None:
                mean_activations_layer = \
                    mean_activations_tasks[name_task][layer_name]
            else:
                mean_activations_layer = np.vstack(
                    (
                        mean_activations_layer,
                        mean_activations_tasks[name_task][layer_name],
                    )
                )

        correlelation_rdm = np.corrcoef(mean_activations_layer, rowvar=True)
        rdm_dict[i_layer] = (1 - correlelation_rdm) / 2

    return rdm_dict


def plot_rdm(ax, rdm_dict, num_hidden, *args, **kwargs):
    """
    Plots the RDM per layer between two IndividualMLPs.

    Args:
        ax (matplotlib.axes): Axis where the plot is represented.
        rdm_dict (dict): Dictionary of RDMs per layer.
        num_hidden (list): Number of hidden units per layer.
    """
    for i_layer, _ in enumerate(num_hidden, 1):
        rdm_layer = rdm_dict[i_layer]

        sns.heatmap(rdm_layer, ax=ax[i_layer - 1], cbar=False, *args, **kwargs)
        # ax[i_layer-1].set_title(f'Layer {i_layer}', fontsize=12)
        ax[i_layer - 1].set_xticks([])
        ax[i_layer - 1].set_yticks([])
