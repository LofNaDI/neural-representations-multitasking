"""
Utils necessary for training and evaluating a model.

- generate_stat_str
- run_epoch
- train
- test
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _generate_stat_str(stat):
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


def _run_epoch(model,
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
    names_tasks = dataloader.dataset.names_tasks
    loss = {task: 0 for task in names_tasks}
    acc = {task: 0 for task in names_tasks}
    total_loss = {task: 0 for task in names_tasks}
    total_acc = {task: 0 for task in names_tasks}
    total_inputs = 0

    is_train = phase == "Train"
    tqdm_bar = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        desc=f"Epoch {epoch_num:>2} ({phase})",
        disable=disable,
    )
    model.train(is_train)
    for _, (inputs, labels) in tqdm_bar:
        inputs = inputs.to(device)
        # Here we transpose the labels so that we can iterate later for the
        # same class!
        labels = labels.T.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
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
            if is_train:
                batch_loss.backward()
                optimizer.step()

        loss_label = _generate_stat_str(total_loss)
        acc_label = _generate_stat_str(total_acc)

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
    names_tasks = validloader.dataset.names_tasks
    train_loss = {task: np.zeros((num_epochs,)) for task in names_tasks}
    train_acc = {task: np.zeros((num_epochs,)) for task in names_tasks}
    valid_loss = {task: np.zeros((num_epochs,)) for task in names_tasks}
    valid_acc = {task: np.zeros((num_epochs,)) for task in names_tasks}

    for epoch in range(num_epochs):
        tloss, tacc = _run_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            epoch + 1,
            phase="Train",
            device=device,
            disable=disable,
        )
        vloss, vacc = _run_epoch(
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


def test(model,
         testloader,
         criterion,
         device):
    """
    Tests a neural network.

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

    tqdm_bar = tqdm(
        enumerate(testloader, 1),
        total=len(testloader),
        desc="Test"
    )
    model.eval()
    with torch.no_grad():
        for _, (inputs, labels) in tqdm_bar:
            inputs = inputs.to(device)
            # Here we transpose the labels so that we can iterate later
            # for the same class!
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

            loss_label = _generate_stat_str(total_loss)
            acc_label = _generate_stat_str(total_acc)

            tqdm_bar.set_postfix_str(f"loss={loss_label}, acc={acc_label}")

    return total_loss, total_acc
