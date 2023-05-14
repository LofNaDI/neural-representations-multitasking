import copy
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import SubsetRandomSampler
from tqdm.auto import tqdm

import multitask.dataset as dataset
from multitask.models.individual import get_individual_model, train
from multitask.utils.argparse import (ExperimentParameters, check_runs,
                                      parse_args)
from multitask.utils.logging import create_logger
from multitask.utils.training import get_device, set_seed


def main(parameters):
    # Retrieve arguments from command line.
    num_runs = parameters.num_runs
    initial_seed = parameters.initial_seed
    max_seed = parameters.max_seed
    num_epochs = parameters.num_epochs
    num_hidden = parameters.num_hidden
    num_train = parameters.num_train
    num_test = parameters.num_test
    batch_size = parameters.batch_size
    tasks_list = parameters.tasks

    # Create output folder
    run_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join("results", "individual")
    out_folder = os.path.join(out_path, run_id)
    out_dict = {}

    # Check if simulation has been run already!
    path_run = check_runs(out_path, parameters)
    if path_run:
        sys.exit(0)

    # Create out folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Create logger
    log_name = "individual"
    log_file = os.path.join(out_folder, "train.log")
    logger = create_logger(log_name, log_file)
    logger.info("\nIndividual\n")
    logger.info("\n" + parameters.__repr__() + "\n")

    # Set seed and get seed for each run
    device = get_device()
    set_seed(initial_seed)
    random_seeds = sorted(np.random.randint(0, max_seed, num_runs))
    assert len(np.unique(random_seeds)) == num_runs

    # Retrieve tasks and define loss
    tasks = dataset.get_tasks_dict(tasks_list, root="data")
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for i_seed, seed in tqdm(enumerate(random_seeds), total=num_runs):
        set_seed(seed)
        log_msg = f"\nSeed: {seed} ({i_seed})\n"
        out_dict[seed] = {}

        indices = dataset.get_split_indices(num_train, num_test)
        out_dict[seed]["indices"] = indices.copy()

        train_sampler = SubsetRandomSampler(indices["train"])
        test_sampler = dataset.SequentialSampler(indices["test"])

        for task_name, task_dataset in tasks.items():
            trainloader = torch.utils.data.DataLoader(
                task_dataset, sampler=train_sampler, batch_size=batch_size
            )
            testloader = torch.utils.data.DataLoader(
                task_dataset, sampler=test_sampler, batch_size=batch_size
            )

            model = get_individual_model(num_hidden, device)
            optimizer = torch.optim.Adam(model.parameters())

            results = train(
                model,
                trainloader,
                testloader,
                criterion,
                optimizer,
                num_epochs,
                device=device,
                disable=True,
            )
            model = model.to('cpu')

            out_dict[seed][task_name] = {}
            out_dict[seed][task_name]["model"] = \
                copy.deepcopy(model.state_dict())
            out_dict[seed][task_name]["results"] = results

            log_msg += (
                f"{task_name}:\n"
                f'\tTrain loss: {results["train_loss"][-1]:.4f}\n'
                f'\tValid loss: {results["valid_loss"][-1]:.4f}\n'
                f'\tTrain acc: {results["train_acc"][-1]:.4f}\n'
                f'\tValid acc: {results["valid_acc"][-1]:.4f}\n'
            )

        logger.info(log_msg + "\n")

    # Save run arguments in json
    out_file_parameters = os.path.join(out_folder, "parameters.json")
    parameters.save(out_file_parameters)

    # Save results in pickle
    out_file_data = os.path.join(out_folder, "data.pickle")
    with open(out_file_data, "wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parse_args()
    parameters = ExperimentParameters(args)
    main(parameters)
