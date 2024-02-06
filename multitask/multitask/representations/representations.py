"""
Functions to generate and plot the Similarity Matrix (RDM)
for different multitask networks.

- get_mean_activations
- calculate_sm
- calculate_rsa
- calculate_representations
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


def calculate_rdm(mean_activations,
                  tasks_names,
                  method='pearson'):
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
            similarity = 1 - cosine_similarity(mean_activations_layer)
        elif method == 'pearson':
            similarity = 1 - np.corrcoef(mean_activations_layer, rowvar=True)
        elif method == 'spearman':
            similarity = 1 - spearmanr(mean_activations_layer, axis=1)[0]
        else:
            raise NotImplementedError

        sm_dict[i_layer] = similarity

    return sm_dict


def calculate_rsa(first_sm_list, second_sm_list):
    """
    Calculates rsa between similarity matrices.

    Args:
        first_sm_list (list): List of similarity matrices for
                              the first network.
        second_sm_list (list): List of similarity matrices for
                               the second network.

    Returns:
        np.ndarray: Representational similarity matrix (N_seeds x N_layers).
    """
    assert len(first_sm_list) == len(second_sm_list)
    num_seeds = len(first_sm_list)

    assert first_sm_list[0].keys() == second_sm_list[0].keys()
    num_layers = len(first_sm_list[0].keys())
    rsa_matrix = np.zeros((num_seeds, num_layers))

    for i_seed in range(num_seeds):
        for i_layer in range(num_layers):
            first_rdm = first_sm_list[i_seed][i_layer+1].flatten()
            second_rdm = second_sm_list[i_seed][i_layer+1].flatten()
            rsa_matrix[i_seed, i_layer] = \
                np.corrcoef(first_rdm, second_rdm)[0, 1]
    return rsa_matrix


def calculate_rsa_diagonal(first_sm_list,
                           second_sm_list,
                           num_inputs):
    """
    Calculates rsa between similarity matrices.

    Args:
        first_sm_list (list): List of similarity matrices for
                              the first network.
        second_sm_list (list): List of similarity matrices for
                               the second network.

    Returns:
        np.ndarray: Representational similarity matrix (N_seeds x N_layers).
    """
    assert len(first_sm_list) == len(second_sm_list)
    num_seeds = len(first_sm_list)
    num_tasks = first_sm_list[0][1].shape[0] // num_inputs

    assert first_sm_list[0].keys() == second_sm_list[0].keys()
    num_layers = len(first_sm_list[0].keys())
    rsa_matrix = np.zeros((num_seeds, num_layers))

    for i_seed in range(num_seeds):
        for i_layer in range(num_layers):
            first_rdm = []
            second_rdm = []
            for i_task in range(num_tasks):
                start = i_task * num_inputs
                end = start + num_inputs
                first_rdm.append(first_sm_list[i_seed][i_layer+1][start:end, start:end].flatten())
                second_rdm.append(second_sm_list[i_seed][i_layer+1][start:end, start:end].flatten())
            first_rdm = np.concatenate(first_rdm)
            second_rdm = np.concatenate(second_rdm)

            rsa_matrix[i_seed, i_layer] = \
                np.corrcoef(first_rdm, second_rdm)[0, 1]
    return rsa_matrix


def calculate_representations(rdm_dict, num_classes, num_tasks):
    """
    Calculates self and shared representations from a similarity matrix.

    Args:
        rdm_dict (dict): RDM for different layers.
        num_classes (int): Number of different labels in the dataset.
        num_tasks (int): Number of different tasks.
    Returns:
        dict: Dictionary with self and shared representations.
    """
    layers = rdm_dict.keys()
    num_layers = len(layers)
    representations = {
        'self': np.zeros((num_layers, )),
        'shared': np.zeros((num_layers, ))
    }

    num_main_diag = num_tasks
    num_off_diag = (num_tasks * (num_tasks - 1)) / 2
    block_elements = num_classes ** 2

    for i_layer, layer in enumerate(layers):
        sm = 1 - rdm_dict[layer]

        self_repr = _calculate_self_representations(sm,
                                                    num_classes,
                                                    num_tasks)
        shared_repr = _calculate_shared_representations(sm,
                                                        self_repr)

        normalized_self_repr = self_repr / (num_main_diag * block_elements)
        normalized_shared_repr = shared_repr / (num_off_diag * block_elements)

        representations['self'][i_layer] = normalized_self_repr
        representations['shared'][i_layer] = normalized_shared_repr

    return representations


def _calculate_shared_representations(sm, self_repr):
    """
    Calculates the shared representations of a similarity matrix.

    Args:
        sm (np.ndarray): Similarity matrix.
        self_repr (float): Self representations of the similarity matrix.

    Returns:
        float: Shared representations.
    """
    total_shared_repr = np.sum(np.abs(sm)) - self_repr
    return total_shared_repr / 2


def _calculate_self_representations(sm, num_classes, num_tasks):
    """
    Calculates the self representations of a similarity matrix.

    Args:
        sm (np.ndarray): Similarity matrix.
        num_classes (int): Number of different labels in the dataset.
        num_tasks (int): Number of different tasks.

    Returns:
        float: Self representations.
    """
    total_self_repr = 0
    for i in range(num_tasks):
        start = i * num_classes
        end = start + num_classes
        total_self_repr += np.sum(np.abs(sm[start:end, start:end]))

    return total_self_repr


def plot_rdm(ax, similarity_dict, num_hidden, *args, **kwargs):
    """
    Plots the similarity matrix. Assumes similarity is bounded in [0, 1].

    Args:
        ax (matplotlib.axes): Axis where the plot is represented.
        rdm_dict (dict): Dictionary of RDMs per layer.
        num_hidden (list): Number of hidden units per layer.
    """
    for i_layer, _ in enumerate(num_hidden, 1):
        rdm_layer = similarity_dict[i_layer]
        sns.heatmap(rdm_layer, ax=ax[i_layer - 1], cbar=False, *args, **kwargs)
        ax[i_layer - 1].set_xticks([])
        ax[i_layer - 1].set_yticks([])
