"""
Functions to generate and plot the Similarity Matrix (RDM)
for different multitask networks.

- get_mean_activations
- calculate_sm
- plot_sm
"""
import copy

import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr


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

    activations_tmp = copy.deepcopy(activations)
    list_labels_tmp = copy.deepcopy(list_labels)

    if isinstance(activations_tmp, dict):
        assert list(activations_tmp.keys()) == tasks_names
        activations_tmp = list(activations_tmp.values())

    if isinstance(list_labels_tmp, np.ndarray):
        num_tasks = len(tasks_names)
        list_labels_tmp = [list_labels_tmp for _ in range(num_tasks)]

    for name_task, activation_task, labels_task in zip(tasks_names,
                                                       activations_tmp,
                                                       list_labels_tmp):
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


def calculate_sm(mean_activations,
                 tasks_names,
                 method='cosine'):
    """
    Calculates the Representational Dissimilarity Matrix (RDM) per layer.

    Args:
        mean_activations (dict): Mean of activations per layer.
        tasks_names (list): List of tasks nmes (N_tasks, ).

    Returns:
        dict: Dictionary of RDMs per layer.
    """
    sm_dict = {}
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

        if method == 'cosine':
            similarity = cosine_similarity(mean_activations_layer)
        elif method == 'pearson':
            similarity = np.corrcoef(mean_activations_layer, rowvar=True)
        elif method == 'spearman':
            similarity, _ = spearmanr(mean_activations_layer, axis=1)
        else:
            raise NotImplementedError

        sm_dict[i_layer] = similarity

    return sm_dict


def plot_sm(ax, similarity_dict, num_hidden, cmap='coolwarm_r',
            vmin=-1, vmax=1, *args, **kwargs):
    """
    Plots the similarity matrix. Assumes similarity is bounded in [0, 1].

    Args:
        ax (matplotlib.axes): Axis where the plot is represented.
        rdm_dict (dict): Dictionary of RDMs per layer.
        num_hidden (list): Number of hidden units per layer.
    """
    for i_layer, _ in enumerate(num_hidden, 1):
        rdm_layer = similarity_dict[i_layer]
        sns.heatmap(rdm_layer, ax=ax[i_layer - 1], cbar=False,
                    cmap=cmap, vmin=vmin, vmax=vmax, *args, **kwargs)

        ax[i_layer - 1].set_xticks([])
        ax[i_layer - 1].set_yticks([])
