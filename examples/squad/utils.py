import os
import random
import yaml

import torch
import numpy as np
from attrdict import AttrDict


def read_config(config_file):
    # Source: https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Catched the following YAMLError:\n{exc}")

    # Convert to AttrDict to allow acessing by dot e.g. config.seed
    config = AttrDict(config)

    return config


def save_config(config_file, output_file):
    config_file = dict(config_file)
    with open(output_file, 'w') as yaml_file:
        yaml.dump(config_file, yaml_file, default_flow_style=True)


def create_unique_dir(path, config, timestamp):
    new_dir = os.path.join(path, timestamp)

    for name in [config.model.model_name_or_path, config.optimizer.learning_rate_schedule]:
        new_dir += f'_{name}'

    if config.optimizer.fp16:
        new_dir += f'_fp16_{config.optimizer.fp16_opt_level}'

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    return new_dir


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()