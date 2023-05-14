import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_runs",
        help="Total number of models trained",
        type=int
    )
    parser.add_argument(
        "--initial_seed",
        help="Initial seed to generate seeds for each run",
        type=int
    )
    parser.add_argument(
        "--max_seed",
        help="Maximum seed value. Minimum is 0.",
        type=lambda x: int(float(x)),
        default=10e5,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs per run",
        type=int
    )
    parser.add_argument(
        "--num_hidden",
        help="Number of hidden units per layer",
        nargs="+",
        type=int
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for training and test",
        type=int
    )
    parser.add_argument(
        "--num_train",
        help="Number of training examples",
        type=int
    )
    parser.add_argument(
        "--num_test",
        help="Number of testing examples",
        type=int
    )
    parser.add_argument(
        "--tasks",
        help="Learning tasks list.",
        nargs="+",
        type=str
    )
    parser.add_argument(
        "--idxs_contexts",
        help="Layers with context units (only for TS models)",
        nargs="+",
        type=int,
        default=None,
    )
    return parser.parse_args()


class ExperimentParameters:
    def __init__(self, args):
        self._args = args

    def __repr__(self):
        return "\n".join(
            f"{key}: {value}" for key, value in self._args.__dict__.items()
        )

    def __getattr__(self, item):
        return getattr(self._args, item)

    def get_dict(self):
        return self._args.__dict__

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self._args.__dict__, f, indent=4)


def _parse_json(folder):
    parameters_json = os.path.join(folder, "parameters.json")
    if os.path.exists(parameters_json):
        with open(parameters_json, "r") as f:
            parameters = json.load(f)
        return parameters
    return {}


def check_runs(out_path, parameters):
    list_folders = sorted(glob.glob(os.path.join(out_path, "*")))

    if not isinstance(parameters, dict):
        dict_params = parameters.get_dict()
    else:
        dict_params = parameters

    for folder in list_folders:
        folder_id = os.path.basename(folder)
        folder_params = _parse_json(folder)
        if dict_params == folder_params:
            print(
                f"Found simulation in {out_path} "
                f"with the same parameters ({folder_id})"
            )
            return folder

    return None
