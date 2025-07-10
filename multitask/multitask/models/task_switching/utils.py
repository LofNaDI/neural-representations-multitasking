"""
Utils necessary for training and evaluating a model.

- run_epoch
- train
- test
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def generate_stat_str(stat):
    output = ""
    num_stats = len(stat)
    if num_stats == 1:
        for i_metric, (key, value) in enumerate(stat.items()):
            output += f"({value:.4f})"
    else:
        for i_metric, (key, value) in enumerate(stat.items()):
            if i_metric == 0:
                output += f"({value:.4f}, "
            elif i_metric == num_stats - 1:
                output += f"{value:.4f})"
            else:
                output += f"{value:.4f}, "
    return output


def run_epoch(model,
              dataloader,
              criterion,
              optimizer,
              epoch_num,
              phase,
              device,
              disable):
    """
    Runs one epoch of training or validation.

    Args:
        model (nn.Module): Neural network model.
        dataloader (torch.utils.data.dataloader.DataLoader):
            Iterable dataloader.
        criterion (torch.nn.modules.loss): Loss to optimize model.
        optimizer (torch.optim.optimizer.Optimizer): Optimizer algorithm.
        epoch_num (int): Index of current epoch.
        phase (str): Phase (train or valid). Disable gradients in valid.
        device (torch.device): Device to run the calculations (CPU or GPU).

    Returns:
        tuple: Loss and accuracy of the epoch.
    """
    loss = {task: 0 for task in dataloader.names_tasks}
    acc = {task: 0 for task in dataloader.names_tasks}
    total_loss = {task: 0 for task in dataloader.names_tasks}
    total_acc = {task: 0 for task in dataloader.names_tasks}

    total_inputs = {task: 0 for task in dataloader.names_tasks}
    is_train = True if phase == "Train" else False
    tqdm_bar = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        desc=f"Epoch {epoch_num:>2} ({phase})",
        disable=disable,
    )
    model.train(is_train)
    for _, (task, inputs, labels) in tqdm_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(inputs, task)
            probs = F.softmax(logits, dim=1)
            batch_loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
        batch_size = len(labels)
        total_inputs[task] += batch_size
        loss[task] += batch_loss.detach() * batch_size
        acc[task] += probs.gather(1, labels.view(-1, 1)).sum().item()

        total_loss[task] = loss[task] / total_inputs[task]
        total_acc[task] = acc[task] / total_inputs[task]

        loss_label = generate_stat_str(total_loss)
        acc_label = generate_stat_str(total_acc)

        tqdm_bar.set_postfix_str(f"loss={loss_label}, acc={acc_label}")

    return total_loss, total_acc


def train(model,
          trainloader,
          validloader,
          criterion,
          optimizer,
          num_epochs,
          device,
          disable=False):
    """
    Trains a neural network.

    Args:
        model (nn.Module): Neural network model.
        trainloader (torch.utils.data.dataloader.DataLoader):
            Iterable dataloader with train data.
        validloader (torch.utils.data.dataloader.DataLoader):
            Iterable dataloader with validation data.
        criterion (torch.nn.modules.loss): Loss to optimize model.
        optimizer (torch.optim.optimizer.Optimizer): Optimizer algorithm.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the calculations (CPU or GPU).

    Returns:
        dict: Training results.
    """
    names_tasks = validloader.names_tasks
    train_loss = {task: np.zeros((num_epochs,)) for task in names_tasks}
    train_acc = {task: np.zeros((num_epochs,)) for task in names_tasks}
    valid_loss = {task: np.zeros((num_epochs,)) for task in names_tasks}
    valid_acc = {task: np.zeros((num_epochs,)) for task in names_tasks}

    for epoch in range(num_epochs):
        tloss, tacc = run_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            epoch + 1,
            phase="Train",
            device=device,
            disable=disable,
        )
        vloss, vacc = run_epoch(
            model,
            validloader,
            criterion,
            optimizer,
            epoch + 1,
            phase="Valid",
            device=device,
            disable=disable,
        )

        for task in names_tasks:
            train_loss[task][epoch] = tloss[task]
            train_acc[task][epoch] = tacc[task]
            valid_loss[task][epoch] = vloss[task]
            valid_acc[task][epoch] = vacc[task]

    results = {}
    for task in names_tasks:
        results[task] = {
            "train_loss": train_loss[task],
            "train_acc": train_acc[task],
            "valid_loss": valid_loss[task],
            "valid_acc": valid_acc[task],
        }

    return results


def test(model, testloader, criterion, device, disable=False):
    test_loss, test_acc = run_epoch(
        model,
        testloader,
        criterion,
        None,
        0,
        phase="Valid",
        device=device,
        disable=disable,
    )
    return test_loss, test_acc


def train_cycles(model,
                 cycle_trainloaders,
                 validloader,
                 criterion,
                 optimizer,
                 num_epochs,
                 device,
                 num_cycles=1,
                 disable=False):
    """
    Trains a neural network.

    Args:
        model (nn.Module): Neural network model.
        trainloader (torch.utils.data.dataloader.DataLoader):
            Iterable dataloader with train data.
        validloader (torch.utils.data.dataloader.DataLoader):
            Iterable dataloader with validation data.
        criterion (torch.nn.modules.loss): Loss to optimize model.
        optimizer (torch.optim.optimizer.Optimizer): Optimizer algorithm.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run the calculations (CPU or GPU).

    Returns:
        dict: Training results.
    """
    names_tasks = validloader.names_tasks
    tasks_per_cycle = len(cycle_trainloaders)
    num_total_epochs = num_epochs * tasks_per_cycle * num_cycles

    valid_loss = {task: np.zeros((num_total_epochs,)) for task in names_tasks}
    valid_acc = {task: np.zeros((num_total_epochs,)) for task in names_tasks}

    for cycle in range(num_cycles):
        for task_idx in range(tasks_per_cycle):
            for epoch in range(num_epochs):
                _ = run_epoch(
                    model,
                    cycle_trainloaders[task_idx],
                    criterion,
                    optimizer,
                    epoch + 1,
                    phase="Train",
                    device=device,
                    disable=disable,
                )
                vloss, vacc = run_epoch(
                    model,
                    validloader,
                    criterion,
                    optimizer,
                    epoch + 1,
                    phase="Valid",
                    device=device,
                    disable=disable,
                )

                results_idx = \
                    epoch + num_epochs * (task_idx + cycle * tasks_per_cycle)

                for task in names_tasks:
                    valid_loss[task][results_idx] = vloss[task]
                    valid_acc[task][results_idx] = vacc[task]

            if cycle == 0:
                for idx_layer in [0, 1]:
                    model.layers[idx_layer].weight.requires_grad = False

    results = {
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "num_total_epochs": num_total_epochs,
    }

    return results
