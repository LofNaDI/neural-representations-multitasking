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
    loss = 0
    acc = 0
    total_loss = 0
    total_acc = 0
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
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            batch_loss = criterion(logits, labels)
            if is_train:
                batch_loss.backward()
                optimizer.step()
        batch_size = len(labels)
        total_inputs += batch_size
        loss += batch_loss.detach() * batch_size
        acc += probs.gather(1, labels.view(-1, 1)).sum().item()

        total_loss = loss / total_inputs
        total_acc = acc / total_inputs

        tqdm_bar.set_postfix_str(f"loss={total_loss:.4f}, acc={total_acc:.4f}")

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
    train_loss = np.zeros((num_epochs,))
    train_acc = np.zeros_like(train_loss)
    valid_loss = np.zeros_like(train_loss)
    valid_acc = np.zeros_like(valid_loss)

    model.to(device)

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

        train_loss[epoch] = tloss
        train_acc[epoch] = tacc
        valid_loss[epoch] = vloss
        valid_acc[epoch] = vacc

    results = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "num_epochs": num_epochs,
    }
    model.to('cpu')

    return results


def test(model, testloader, criterion, device):
    """
    Tests a neural network.

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

    tqdm_bar = tqdm(
        enumerate(testloader, 1),
        total=len(testloader),
        desc="Test")
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

            tqdm_str = f"loss={total_loss:.4f}, " "acc={total_acc:.4f}"
            tqdm_bar.set_postfix_str(tqdm_str)

    return total_loss, total_acc
