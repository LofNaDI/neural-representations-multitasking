"""
Functions to generate and plot the Representation Dissimilarity Matrix (RDM)
for parallel networks.

- get_mean_activations
- calculate_rdm
- plot_rdm
"""
import numpy as np
import seaborn as sns


def get_mean_activations(activations,
                         num_hidden,
                         list_labels,
                         tasks_names):
    """
    Calculates the mean activation of units by class per layer.

    Args:
        activations (dict): Dictionary of np.ndarray containing the matrix
            of activations (N_test x N_hidden_units).
        num_hidden (list): Number of hidden units per layer.
        list_labels (list): List of numbers of test set (N_test, ).
        tasks_names (list): List of tasks nmes (N_tasks, ).

    Returns:
        dict: Mean of activations of units by class per layer.
    """
    mean_activations = {}

    for name_task, activation_task, labels_task in zip(tasks_names,
                                                       activations,
                                                       list_labels):
        mean_activations[name_task] = {}
        num_classes = len(set(labels_task))

        for i_layer, num_units in enumerate(num_hidden, 1):
            layer_name = f'layer{i_layer}'
            mean_activations[name_task][layer_name] = \
                np.zeros((num_classes, num_units))

            for label in range(num_classes):
                activations_label = \
                    activation_task[layer_name][labels_task == label, :]
                mean_activations[name_task][layer_name][label, :] = \
                    np.mean(activations_label, axis=0)

    return mean_activations


def calculate_rdm(mean_activations,
                  tasks_names):
    """
    Calculates the Representational Dissimilarity Matrix (RDM) per layer.

    Args:
        mean_activations (dict): Mean of activations per layer.
        tasks_names (list): List of tasks nmes (N_tasks, ).

    Returns:
        dict: Dictionary of RDMs per layer.
    """
    rdm_dict = {}
    num_layers = len(mean_activations[tasks_names[0]])

    for i_layer in range(1, num_layers+1):
        layer_name = f"layer{i_layer}"

        mean_activations_layer = None
        for name_task in tasks_names:
            if mean_activations_layer is None:
                mean_activations_layer = \
                    mean_activations[name_task][layer_name]
            else:
                mean_activations_layer = np.vstack(
                    (
                        mean_activations_layer,
                        mean_activations[name_task][layer_name],
                    )
                )

        correlation_rdm = np.corrcoef(mean_activations_layer, rowvar=True)
        rdm_dict[i_layer] = (1 - correlation_rdm) / 2

    return rdm_dict


def plot_rdm(ax, rdm_dict, num_hidden, *args, **kwargs):
    """
    Plots the RDM per layer for a ParallelMLP..

    Args:
        ax (matplotlib.axes): Axis where the plot is represented.
        rdm_dict (dict): Dictionary of RDMs per layer.
        num_hidden (list): Number of hidden units per layer.
    """
    for i_layer, _ in enumerate(num_hidden, 1):
        rdm_layer = rdm_dict[i_layer]

        sns.heatmap(rdm_layer, ax=ax[i_layer - 1], cbar=False, *args, **kwargs)
        ax[i_layer - 1].set_xticks([])
        ax[i_layer - 1].set_yticks([])
