import os
import random
import yaml

import warnings
import torch
from torch._six import inf
import torch.nn.functional as F
from torch.distributions import Categorical

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


def create_unique_dir(path, config, args, timestamp):
    new_dir = os.path.join(path, timestamp)

    for name in [config.model.model_name_or_path, config.model.classifier_init_std,
                 config.optimizer.learning_rate_schedule, config.optimizer.learning_rate,
                 config.model.pooler_activation, config.training.num_train_epochs, config.optimizer.name,
                 config.optimizer.correct_bias,
                 config.optimizer.adam_epsilon, config.optimizer.local_normalization, config.optimizer.max_grad_norm,
                 ]:
        new_dir += f'_{name}'

    new_dir += f'_{config.training.per_gpu_train_batch_size * config.training.gradient_accumulation_steps}'

    try:
        new_dir += f'_hdp_{config.model.hidden_dropout_prob}'
        new_dir += f'_adp_{config.model.attention_probs_dropout_prob}'
        new_dir += f'_cdp_{config.model.classifier_dropout_prob}'
    except AttributeError:
        pass

    try:
        new_dir += f'_attnorm_{config.model.attention_norm}'
    except AttributeError:
        pass

    if config.optimizer.weight_decay >= 0.0:
        new_dir += f'_wd_{config.optimizer.weight_decay}'

    if config.optimizer.warmup_steps > 0:
        new_dir += f'_wup_{config.optimizer.warmup_steps}'

    # if config.optimizer.adam_epsilon == 1e-08:
    #     new_dir += f'_hf'  # hugging face default, also Dodge et al. 2020 default

    if config.model.re_init_pooler:
        new_dir += f'_{config.model.distribution}'

        if config.model.distribution == 'uniform':
            new_dir += f'_{config.model.bound}'

        if config.model.distribution == 'normal':
            new_dir += f'_{config.model.std}'

    if config.model.pooler_dropout:
        new_dir += f'_pdrop'

    if config.model.pooler_layer_norm:
        new_dir += f'_pln'

    if config.optimizer.fp16:
        new_dir += f'_fp16_{config.optimizer.fp16_opt_level}'

    new_dir += f'_{args.seed}'

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    return new_dir


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # make torch deterministic: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def tanh_saturation(activations, lower_bound=-.99, upper_bound=.99):
    # activations is of shape (bsz, hidden_dim)

    pos_saturated = torch.where(activations.view(-1) >= upper_bound)[0]
    neg_saturated = torch.where(activations.view(-1) <= lower_bound)[0]

    return (len(pos_saturated) + len(neg_saturated)) / len(activations.view(-1))


def relu_saturation(activations):
    # activations is of shape (bsz, hidden_dim)

    saturated = torch.where(activations.view(-1) == 0)[0]

    return len(saturated) / len(activations.view(-1))


def gelu_saturation(activations):
    # activations is of shape (bsz, hidden_dim)

    saturated = torch.where(activations.view(-1) <= 0)[0]

    return len(saturated) / len(activations.view(-1))


def softmax_saturation(logits, lower_bound=-.01, upper_bound=.99):
    activations = F.softmax(logits)

    pos_saturated = torch.where(activations.view(-1) >= upper_bound)[0]
    neg_saturated = torch.where(activations.view(-1) <= lower_bound)[0]

    return (len(pos_saturated) + len(neg_saturated)) / len(activations.view(-1))


def softmax_entropy(logits):
    # logits are of shape: (bsz, n_classes)
    activations = F.softmax(logits, dim=1)
    entropy = Categorical(probs=activations).entropy()  # entropy is of shape (bsz,)
    mean_entropy = entropy.mean()  # take the mean over batches

    return mean_entropy.item()


def clip_grad_sign_(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data = torch.sign(p.grad.data)


def compute_total_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm
