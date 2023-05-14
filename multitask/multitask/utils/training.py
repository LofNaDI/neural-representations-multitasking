import os
import random

import numpy as np
import torch


def get_device():
    """Gets available device in current machine.

    Returns:
       torch.device: Device to run the calculations (CPU or GPU).
    """
    if torch.cuda.is_available():
        print("Running on GPU.")
        device = torch.device("cuda")
    else:
        print("Running on CPU.")
        device = torch.device("cpu")
    return device


def set_seed(seed):
    """Sets rng using seed for reproducibility.

    Args:
        seed (int): Number to initialize the rng.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
