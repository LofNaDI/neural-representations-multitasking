import argparse
import copy
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

import multitask.utils.config as config
import multitask.utils.logging as logging
import multitask.utils.training as training
import multitask.datasets.data as data
import multitask.datasets.samplers as samplers
import multitask.models.parallel as parallel
import multitask.models.sequential as sequential
import multitask.models.task_switching as task_switching


def parse_args():
    parser = argparse.ArgumentParser(description='Runs multiple multitask model experiments.')
    parser.add_argument('config', type=Path, help='Path to the configuration file.')
    return parser.parse_args()


def log_train_results(tasks_dataset, output_dir):
    log_msg = ''

    for task_name in tasks_dataset['tasks'].keys():
        train_loss = output_dir['results'][task_name]['train_loss'][-1]
        valid_loss =  output_dir['results'][task_name]['valid_loss'][-1]
        train_acc =  output_dir['results'][task_name]['train_acc'][-1]
        valid_acc =  output_dir['results'][task_name]['valid_acc'][-1]

        log_msg += (
            f"{task_name}:\n"
            f"\tTrain loss: {train_loss:.4f}\n"
            f"\tValid loss: {valid_loss:.4f}\n"
            f"\tTrain acc: {train_acc:.4f}\n"
            f"\tValid acc: {valid_acc:.4f}\n"
        )
    return log_msg

def run_sequential(models,
                   tasks_dataset,
                   indices,
                   criterion,
                   optimizers,
                   num_epochs,
                   batch_size,
                   partition,
                   device):
    results = {}
    results['indices'] = {}
    results['models'] = {}
    results['results'] = {}

    indices = samplers.split_indices_tasks(tasks_dataset, indices, partition)
    train_dataloader, test_dataloaders = \
        sequential.dataloaders.create_dataloaders(tasks_dataset,
                                                  indices,
                                                  batch_size)
    
    for task_name, model in models.items():
        results['indices'][task_name] = {
            'train': copy.deepcopy(indices['train'][task_name]),
            'test': copy.deepcopy(indices['test'][task_name])
        }
        results['models'][task_name] = copy.deepcopy(model.state_dict())
        results['results'][task_name] = sequential.utils.train(
            model,
            train_dataloader[task_name],
            test_dataloaders[task_name],
            criterion,
            optimizers[task_name],
            num_epochs,
            device=device,
            disable=True,
        )
        
    return results


def run_parallel(model,
                 tasks_dataset,
                 indices,
                 criterion,
                 optimizer,
                 num_epochs,
                 batch_size,
                 device):
    train_dataloaders, test_dataloaders = \
        parallel.dataloaders.create_dataloaders(tasks_dataset,
                                                indices,
                                                batch_size)
    results = parallel.utils.train(
        model,
        train_dataloaders,
        test_dataloaders,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        disable=True,
    )
    return results


def run_task_switching(model,
                       tasks_dataset,
                       indices,
                       criterion,
                       optimizer,
                       num_epochs,
                       batch_size,
                       partition,
                       scheduler,
                       device):
    indices = samplers.split_indices_tasks(tasks_dataset, indices, partition)
    train_dataloaders, test_dataloaders = \
        task_switching.dataloaders.create_dataloaders(tasks_dataset,
                                                      indices,
                                                      batch_size)
    scheduler_train = task_switching.schedulers.get_scheduler(scheduler)
    scheduler_test = task_switching.schedulers.get_scheduler('sequential')
    tasks_trainloader = scheduler_train(train_dataloaders)
    tasks_testloader = scheduler_test(test_dataloaders)

    results = task_switching.utils.train(model,
                                         tasks_trainloader,
                                         tasks_testloader,
                                         criterion,
                                         optimizer,
                                         num_epochs=num_epochs,
                                         device=device,
                                         disable=True)
    
    return results


def run_experiment(exp, base_dir='results'):
    # Check if experiment with same parameters already exists
    model_dir = Path(base_dir) / exp.model
    model_dir.mkdir(parents=True, exist_ok=True)
    exists_run = exp.find_existing_run(model_dir)
    if exists_run is not None:
        print(f'Found existing run with the same parameters: {exists_run}. Skipping this experiment.')
        return
    
    # Create directory for the experiment
    timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = model_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    # Create logger for the experiment
    log_name = exp.model
    log_file = run_dir / 'train.log'
    logger = logging.create_logger(log_name, log_file)
    logger.info(f'\n{log_name}\n')
    logger.info('\n' + exp.pretty() + '\n')

    # Set seed and get seed for each run
    initial_seed = exp.initial_seed
    rng = np.random.default_rng(initial_seed)
    random_seeds = sorted(rng.integers(0, 2**32 - 1, exp.num_runs, dtype=np.uint32))
    
    # Deine loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam
    device = training.get_device()

    # Initialize output results
    output_dir = {}

    for i_seed, seed in tqdm(enumerate(random_seeds), total=exp.num_runs):
        training.set_seed(seed)
        log_msg = f"\nSeed: {seed} ({i_seed})\n"
        
        tasks_dataset = data.get_tasks_dataset(exp)
        indices = samplers.split_dataset_train_test(tasks_dataset, exp.num_train, exp.num_test)
        
        if exp.model == 'sequential':
            models = sequential.models.get_model(tasks_dataset, num_hidden=exp.num_hidden)
            optimizers = {task_name: optimizer_cls(model.parameters()) for task_name, model in models.items()}
            results = run_sequential(models, tasks_dataset, indices, criterion, optimizers,
                                              num_epochs=exp.num_epochs, batch_size=exp.batch_size,
                                              partition=exp.partition, device=device)
            output_dir[seed] = results
        else:
            if exp.model == 'parallel':
                model = parallel.models.get_model(tasks_dataset, num_hidden=exp.num_hidden)
                optimizer = optimizer_cls(model.parameters())
                results = run_parallel(model, tasks_dataset, indices, criterion, optimizer,
                                       num_epochs=exp.num_epochs, batch_size=exp.batch_size,
                                       device=device)
            elif exp.model == 'task_switching':
                model = task_switching.models.get_model(tasks_dataset, num_hidden=exp.num_hidden, idxs_contexts=exp.idx_context)
                optimizer = optimizer_cls(model.parameters())
                results = run_task_switching(model, tasks_dataset, indices, criterion, optimizer,
                                            num_epochs=exp.num_epochs, batch_size=exp.batch_size,
                                            partition=exp.partition, scheduler=exp.scheduler, device=device)
                        
            else:
                raise ValueError(f"Unknown model type: {exp.model}. Supported models are 'sequential', 'parallel', and 'task_switching'.")

            output_dir[seed] = {}
            output_dir[seed]['model'] = copy.deepcopy(model.state_dict())
            output_dir[seed]['indices'] = copy.deepcopy(indices)
            output_dir[seed]['results'] = results
        
        
        log_msg += log_train_results(tasks_dataset, output_dir[seed])
        logger.info(log_msg + "\n")
                
    # Save parameters in json
    exp.save_json(run_dir / 'parameters.json')

    # Save results
    out_file_data = run_dir / 'data.pickle'
    with open(out_file_data, "wb") as f:
        pickle.dump(output_dir, f, protocol=pickle.HIGHEST_PROTOCOL)



def main():
    args = parse_args()
    config_path = args.config

    try:
        raw = yaml.safe_load(config_path.read_text())
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    # Validate parameters in configuration file
    cfg_file = config.ExperimentsConfig.model_validate(raw)
    num_experiments = len(cfg_file.experiments)
    print(f'Running {num_experiments} experiments from the configuration file: {config_path}.\n')

    # Run each experiment
    for i_exp, exp in enumerate(cfg_file.experiments):
        print(f'Running experiment {i_exp + 1}/{num_experiments} with parameters:')
        print(exp.pretty())
        run_experiment(exp)


if __name__ == "__main__":
    main()