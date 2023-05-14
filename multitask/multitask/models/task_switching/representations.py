"""
Functions to generate and plot the Representation Dissimilarity Matrix (RDM)
for task-swithing networks.

- get_mean_activations
- calculate_rdm
- plot_rdm
"""
import numpy as np
import seaborn as sns

NUM_DIGITS = 10


def get_mean_activations(activations, names_tasks, num_hidden, list_numbers):
    """
    Calculates the mean_activation matrix per layer.

    Args:
        activations (dict): Dictionary of np.ndarray containing the matrix
            of activations (N_test x N_hidden_units).
        names_tasks (list): Names of tasks.
        num_hidden (list): Number of hidden units per layer.
        list_numbers (list): List of numbers of test set (N_test, ).

    Returns:
        dict: Mean of activations per layer.
    """
    mean_activations = {}
    for name_task in names_tasks:
        mean_activations[name_task] = {}
        for i_layer, num_units in enumerate(num_hidden, 1):
            layer_name = f"layer{i_layer}"
            mean_activations[name_task][layer_name] = \
                np.zeros((NUM_DIGITS, num_units))
            for number in range(NUM_DIGITS):
                idxs_activations = list_numbers == number
                activations_number = activations[name_task][layer_name][
                    idxs_activations, :
                ]
                mean_activations[name_task][layer_name][number, :] = np.mean(
                    activations_number, axis=0
                )

    return mean_activations


def calculate_rdm(activations, test_tasks, num_hidden, list_numbers):
    """
    Calculates the Representational Dissimilarity Matrix (RDM) per layer.

    Args:
        activations_tasks (list): List of dictionaries of activations per task.
        test_tasks (dict): Dictionary of test tasks.
        num_hidden (list): Number of hidden units per layer.
        list_numbers (list): List of numbers of test set (N_test, ).

    Returns:
        dict: Dictionary of RDMs per layer.
    """
    names_tasks = list(test_tasks.keys())
    mean_activations = get_mean_activations(
        activations, names_tasks, num_hidden, list_numbers
    )
    rdm_dict = {}

    for i_layer, _ in enumerate(num_hidden, 1):
        layer_name = f"layer{i_layer}"

        mean_activations_layer = None
        for name_task in names_tasks:
            if mean_activations_layer is None:
                mean_activations_layer = \
                    mean_activations[name_task][layer_name]
            else:
                mean_activations_layer = np.vstack(
                    (mean_activations_layer,
                     mean_activations[name_task][layer_name])
                )

        correlation_dm = np.corrcoef(mean_activations_layer, rowvar=True)
        rdm_dict[i_layer] = (1 - correlation_dm) / 2

    return rdm_dict


def plot_rdm(ax, rdm_dict, num_hidden, idxs_contexts, *args, **kwargs):
    """
    Plots the RDM per layer for a ParallelMLP..

    Args:
        ax (matplotlib.axes): Axis where the plot is represented.
        rdm_dict (dict): Dictionary of RDMs per layer.
        num_hidden (list): Number of hidden units per layer.
        idxs_contexts (list): Indices of contexts layers with task biases.
    """
    for i_layer, _ in enumerate(num_hidden, 1):
        rdm_layer = rdm_dict[i_layer]

        sns.heatmap(rdm_layer, ax=ax[i_layer - 1], cbar=False, *args, **kwargs)

        # if i_layer - 1 in idxs_contexts:
        #     ax[i_layer-1].set_title(f'Layer {i_layer} (C)', fontsize=12)
        # else:
        #     ax[i_layer-1].set_title(f'Layer {i_layer}', fontsize=12)
        ax[i_layer - 1].set_xticks([])
        ax[i_layer - 1].set_yticks([])
