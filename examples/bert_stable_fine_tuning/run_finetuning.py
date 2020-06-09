# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import copy
import glob
import json
import logging
import os
import random
from datetime import datetime
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from examples.bert_stable_fine_tuning.glue_metrics import glue_compute_metrics as compute_metrics
from examples.bert_stable_fine_tuning.glue import glue_convert_examples_to_features as convert_examples_to_features
from examples.bert_stable_fine_tuning.glue import glue_output_modes as output_modes
from examples.bert_stable_fine_tuning.glue import glue_processors as processors

from examples.bert_stable_fine_tuning.utils import read_config, save_config, create_unique_dir, set_seed, tanh_saturation, relu_saturation, gelu_saturation, softmax_saturation, softmax_entropy, clip_grad_sign_, compute_total_grad_norm

from examples.bert_stable_fine_tuning.pooling_bert import BertForSequenceClassificationWithPooler
from examples.bert_stable_fine_tuning.pooling_roberta import RobertaForSequenceClassificationWithPooler
from examples.bert_stable_fine_tuning.pooling_albert import AlbertForSequenceClassificationWithPooler

from examples.bert_stable_fine_tuning.adamw import mAdamW

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import wandb

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            RobertaConfig,
            AlbertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "pooler-bert": (BertConfig, BertForSequenceClassificationWithPooler, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "pooler-roberta": (RobertaConfig, RobertaForSequenceClassificationWithPooler, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "pooler-albert": (AlbertConfig, AlbertForSequenceClassificationWithPooler, AlbertTokenizer),
}


def train(args, config, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        if config.tensorboard.enable:
            logger.info(f"Saving tensorboard logs to: {config.tensorboard.log_dir}")
            tb_writer = SummaryWriter(log_dir=config.tensorboard.log_dir, flush_secs=30)

            # Write args and config file to tensorboard
            tb_writer.add_text('args', str(args))
            tb_writer.add_text('config', str(config))

        # Create logfile for training progress
        train_log_file = os.path.join(config.output.log_dir, "train_results.tsv")
        with open(train_log_file, "a") as writer:
            writer.write(f"global_step\tloss\tlearning_rate\n")  # write header

    config['training']['train_batch_size'] = config.training.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.training.train_batch_size)

    if config.training.max_steps > 0:
        t_total = config.training.max_steps
        # overwrite epochs
        config['training']['num_train_epochs'] = config.training.max_steps // (len(train_dataloader) // config.training.gradient_accumulation_steps) + 1
        steps_per_epoch = t_total / config.training.num_train_epochs
    else:
        t_total = len(train_dataloader) // config.training.gradient_accumulation_steps * config.training.num_train_epochs
        steps_per_epoch = t_total / config.training.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]  # no weight decay for these params

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
            "weight_decay": config.optimizer.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
            "weight_decay": 0.0
        },
    ]

    # We also keep track of the names
    optimizer_grouped_parameters_names = [
        {
            "params": [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
            "weight_decay": config.optimizer.weight_decay,
        },
        {
            "params": [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],  # keep only params that require a gradient
            "weight_decay": 0.0
        },
    ]

    if config.optimizer.name == 'adamW':
        # Create AdamW optimizer
        optimizer = mAdamW(optimizer_grouped_parameters, lr=config.optimizer.learning_rate,
                           betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
                           eps=config.optimizer.adam_epsilon, correct_bias=config.optimizer.correct_bias,
                           local_normalization=config.optimizer.local_normalization, max_grad_norm=config.optimizer.max_grad_norm
                           )
    else:
        raise KeyError('Unknown optimizer:', config.optimizer.name)

    # Create learning rate schedule
    warmup_steps = int(t_total * config.optimizer.warmup_steps)
    if config.optimizer.learning_rate_schedule == 'warmup-linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    elif config.optimizer.learning_rate_schedule == 'warmup-constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
    elif config.optimizer.learning_rate_schedule == 'warmup-cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total, num_cycles=0.5
        )
    else:
        raise NotImplementedError(f"Unkown learning_rate_schedule {config.optimizer.learning_rate_schedule}")

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(config.model.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(config.model.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(config.model.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(config.model.model_name_or_path, "scheduler.pt")))

    if config.optimizer.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.optimizer.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.training.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", config.training.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        config.training.train_batch_size
        * config.training.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", config.training.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Optimization steps per epoch = %d", steps_per_epoch)
    logger.info("  Warmup steps = %d", warmup_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(config.model.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(config.model.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0

        epochs_trained = global_step // (len(train_dataloader) // config.training.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // config.training.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    last_eval_metric = 0.0
    best_step = 0
    stop_training = False
    model.zero_grad()

    # Create training iterator
    train_iterator = trange(
        epochs_trained, int(config.training.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    # Evaluate and log once before training starts/resumes
    if args.local_rank in [-1, 0]:
        # Only evaluate when single GPU and not torch.distributed otherwise metrics may not average well
        if args.local_rank == -1 and config.training.evaluate_during_training:
            logs = {}
            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar

            results = evaluate(args, config, model, tokenizer, global_step=global_step)
            last_eval_metric = results[config.training.early_stopping_metric]
            best_step = global_step

            if config.training.early_stopping:
                # Save only the best model
                logger.info("   Saving new best model checkpoint to %s", config.output.checkpoint_dir)
                # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(config.output.checkpoint_dir)
                tokenizer.save_pretrained(config.output.checkpoint_dir)

                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(config.output.checkpoint_dir, "training_args.bin"))

                # Good practice: save your training config file together with the trained model
                save_config(config, os.path.join(config.output.checkpoint_dir, "config.yaml"))

            for key, value in results.items():
                eval_key = "eval_{}".format(key)
                logs[eval_key] = value

            for key, value in logs.items():
                if config.tensorboard.enable:
                    tb_writer.add_scalar(key, value, global_step)
                if config.wandb.enable:
                    wandb.log({key: value}, step=global_step)

            print(json.dumps({**logs, **{"step": global_step}}))

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        # Train for one epoch
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            # Preprocess batch
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if config.model.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if config.model.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            # Run foward pass
            outputs = model(**inputs)  # loss, logits, pooled_activation_dropout, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor, (hidden_states), (attentions)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits, pooled_activation_dropout, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor = outputs[1:7]
            hidden_states = outputs[7]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if config.training.gradient_accumulation_steps > 1:
                loss = loss / config.training.gradient_accumulation_steps

            if config.optimizer.fp16:  # scale loss when using fp16
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()  # backprop to compute gradients
            else:
                loss.backward()  # backprop to compute gradients

            tr_loss += loss.item()  # accumulate loss across batches. Needed for gradient accumulation

            if (step + 1) % config.training.gradient_accumulation_steps == 0:  # perform an update, else continue with the next batch

                if args.local_rank in [-1, 0] and config.training.train_logging_steps > 0 and ((global_step % config.training.train_logging_steps == 0) or global_step == 0):
                    # Log gradients before clipping them
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        grads = param.grad
                        grads = grads.view(-1)
                        grads_norm = torch.norm(grads, p=2, dim=0)
                        weight_norm = torch.norm(param.view(-1), p=2, dim=0)

                        # compute ratio ||w|| / ||grad L(w)||
                        grad_weight_ratio = weight_norm / grads_norm

                        if config.tensorboard.enable:
                            if config.tensorboard.log_histograms and (config.tensorboard.stop_after is None or global_step < config.tensorboard.stop_after):  # we log histograms only during the first stop_after steps
                                # logging histograms can result in large log files
                                tb_writer.add_histogram(name, param, global_step)  # log param histograms
                                tb_writer.add_histogram(name + '_grad', grads, global_step)  # log gradients histograms

                            tb_writer.add_scalar(name + '_grad_norm', grads_norm, global_step)  # log param gradient norm

                        if config.wandb.enable:
                            if config.wandb.log_histograms and (config.wandb.stop_after is None or global_step < config.wandb.stop_after):  # we log histograms only during the first stop_after steps
                                # param_histo = param.detach().cpu().numpy().reshape(-1)
                                # param_histo = wandb.Histogram(sequence=param_histo)
                                # wandb.log({f"{name}_grad_histo": param_histo}, step=global_step)
                                pass

                            wandb.log({f"{name}_grad_norm": grads_norm}, step=global_step)
                            # wandb.log({f"{name}_norm": weight_norm}, step=global_step)

                            # Log mean and std for gradients
                            # mean = torch.mean(grads)
                            # std = torch.std(grads)
                            # wandb.log({f"{name}_grad_mean": mean}, step=global_step)
                            # wandb.log({f"{name}_grad_std": std}, step=global_step)

                            # Log gradient weights ratio
                            # wandb.log({f"{name}_weight_grads_ratio": grad_weight_ratio}, step=global_step)

                # Clip gradients
                if config.optimizer.max_grad_norm > 0 and not config.optimizer.local_normalization:
                    if config.optimizer.fp16:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.optimizer.max_grad_norm)
                        total_grad_norm_after_clipping = compute_total_grad_norm(amp.master_params(optimizer))  # compute total gradient norm after clipping

                    else:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                        total_grad_norm_after_clipping = compute_total_grad_norm(model.parameters())  # compute total gradient norm after clipping
                else:
                    total_grad_norm = compute_total_grad_norm(model.parameters())  # compute total gradient norm
                    total_grad_norm_after_clipping = total_grad_norm

                # Log training progress (before performing the update)
                if args.local_rank in [-1, 0] and config.training.train_logging_steps > 0 and ((global_step % config.training.train_logging_steps == 0) or global_step == 0):
                    logs = {}

                    # Log gradients again after clipping them
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        grads = param.grad
                        grads = grads.view(-1)
                        grads_norm = torch.norm(grads, p=2, dim=0)

                        # compute ratio ||w|| / ||grad L(w)||
                        grad_weight_ratio = torch.norm(param.view(-1), p=2, dim=0) / grads_norm

                        if config.tensorboard.enable:
                            if config.tensorboard.log_histograms and (config.tensorboard.stop_after is None or global_step < config.tensorboard.stop_after):  # we log histograms only during the first stop_after steps
                                # logging histograms can result in large log files
                                tb_writer.add_histogram(name, param, global_step)  # log param histograms
                                tb_writer.add_histogram(name + '_grad_after_clipping', grads, global_step)  # log gradients histograms

                            # tb_writer.add_scalar(name + '_grad_norm_after_clipping', grads_norm, global_step)  # log param gradient norm

                        if config.wandb.enable:
                            if config.wandb.log_histograms and (config.wandb.stop_after is None or global_step < config.wandb.stop_after):  # we log histograms only during the first stop_after steps
                                # param_histo = param.detach().cpu().numpy().reshape(-1)
                                # param_histo = wandb.Histogram(sequence=param_histo)
                                # wandb.log({f"{name}_grad_histo": param_histo}, step=global_step)
                                pass
                            # wandb.log({f"{name}_grad_norm_after_clipping": grads_norm}, step=global_step)

                            # Log mean and std for gradients
                            # mean = torch.mean(grads)
                            # std = torch.std(grads)
                            # wandb.log({f"{name}_grad_after_clipping_mean": mean}, step=global_step)
                            # wandb.log({f"{name}_grad_after_clipping_std": std}, step=global_step)

                            # Log gradient weights ratio
                            # wandb.log({f"{name}_weight_grads_ratio_after_clipping": grad_weight_ratio}, step=global_step)

                    # Log training progress
                    if global_step == 0:
                        loss_scalar = (tr_loss - logging_loss)
                    else:
                        loss_scalar = (tr_loss - logging_loss) / config.training.train_logging_steps

                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["train_loss"] = loss_scalar
                    logging_loss = tr_loss

                    # log mean and std for hidden states
                    # for layer, hidden_state in enumerate(hidden_states + (pooled_activation_dropout, )):
                    # hidden_state-shape = (bsz, seq_len, hidden_dim)
                    # mean = torch.mean(hidden_state)
                    # std = torch.std(hidden_state)
                    # logs[f"hidden_state_{layer}_mean"] = mean
                    # logs[f"hidden_state_{layer}_std"] = std

                    # log stats on the labels and predictions
                    logs[f"labels_per_batch_mean"] = torch.mean(inputs['labels'].float())
                    preds_ = torch.argmax(logits, dim=1)
                    logs[f"predictions_per_batch_mean"] = torch.mean(preds_.float())
                    logs[f"train_loss_per_batch"] = loss
                    logs[f"train_acc_per_batch"] = torch.mean((inputs['labels'].view(-1) == preds_.view(-1)).float())

                    if config.model.pooler_activation == 'tanh':
                        # log fraction of saturated tanh units of the pooling layer
                        saturated_units = tanh_saturation(pooled_activation_dropout, lower_bound=-.99, upper_bound=.99)
                        logs["saturated_tanh_units_99"] = saturated_units

                        # we additionally log the fraction of units that is "almost" saturated to get a better idea of what is going on
                        saturated_units = tanh_saturation(pooled_activation_dropout, lower_bound=-.95, upper_bound=.95)
                        logs["saturated_tanh_units_95"] = saturated_units

                        saturated_units = tanh_saturation(pooled_activation_dropout, lower_bound=-.90, upper_bound=.90)
                        logs["saturated_tanh_units_90"] = saturated_units

                        saturated_units = tanh_saturation(pooled_activation_dropout, lower_bound=-.85, upper_bound=.85)
                        logs["saturated_tanh_units_85"] = saturated_units

                    elif config.model.pooler_activation == 'relu':
                        saturated_units = relu_saturation(pooled_activation_dropout)  # fraction of units = 0.0
                        logs["saturated_relu_units"] = saturated_units

                    elif config.model.pooler_activation == 'gelu':
                        saturated_units = gelu_saturation(pooled_activation_dropout)  # fraction of units = 0.0
                        logs["saturated_gelu_units"] = saturated_units

                    # log also entropy of the  classifier softmax
                    softmax_ent = softmax_entropy(logits)
                    logs["softmax_entropy_ln"] = softmax_ent

                    # Log total gradient norm (after clipping)
                    if config.tensorboard.enable:
                        tb_writer.add_scalar('total_grad_norm', total_grad_norm, global_step)

                        # Log pooler related tensors
                        if config.model.pooler_activation == 'tanh':
                            if config.tensorboard.log_histograms and (config.tensorboard.stop_after is None or global_step < config.tensorboard.stop_after):  # we log histograms only during the first stop_after steps
                                tb_writer.add_histogram('logits', logits, global_step)
                                tb_writer.add_histogram('pooled_activation_dropout', pooled_activation_dropout, global_step)
                                tb_writer.add_histogram('pooled_activation', pooled_activation, global_step)
                                tb_writer.add_histogram('normalized_pooled_linear_transform', normalized_pooled_linear_transform, global_step)
                                tb_writer.add_histogram('pooled_linear_transform', pooled_linear_transform, global_step)
                                tb_writer.add_histogram('token_tensor', token_tensor, global_step)

                            if config.wandb.log_histograms and (config.wandb.stop_after is None or global_step < config.wandb.stop_after):  # we log histograms only during the first stop_after steps
                                histo = pooled_activation_dropout.detach().cpu().numpy().reshape(-1)
                                wandb.log({f"{name}_grad_histo": wandb.Histogram(sequence=histo, num_bins=64)})

                    if config.wandb.enable:
                        wandb.log({'total_grad_norm': total_grad_norm}, step=global_step)
                        wandb.log({'total_grad_norm_after_clipping': total_grad_norm_after_clipping}, step=global_step)

                    for key, value in logs.items():
                        if config.tensorboard.enable:
                            tb_writer.add_scalar(key, value, global_step)
                        if config.wandb.enable:
                            wandb.log({key: value}, step=global_step)

                    # print(json.dumps({**logs, **{"step": global_step}}))

                    # Write to logfile
                    with open(train_log_file, "a") as writer:
                        writer.write(f"{global_step}\t{loss_scalar}\t{learning_rate_scalar}\n")

                # Evaluate during training
                if args.local_rank in [-1, 0] and config.training.eval_logging_steps > 0 and global_step % config.training.eval_logging_steps == 0 and global_step > 0 and step < steps_per_epoch - 1:
                    logs = {}
                    if (
                        args.local_rank == -1 and config.training.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, config, model, tokenizer, global_step=global_step)
                        eval_metric = results[config.training.early_stopping_metric]

                        if eval_metric > last_eval_metric:
                            last_eval_metric = eval_metric
                            best_step = global_step

                            if config.training.early_stopping:
                                # Save only the best model
                                logger.info("   Saving new best model checkpoint to %s", config.output.checkpoint_dir)
                                # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
                                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                # They can then be reloaded using `from_pretrained()`
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(config.output.checkpoint_dir)
                                tokenizer.save_pretrained(config.output.checkpoint_dir)

                                # Good practice: save your training arguments together with the trained model
                                torch.save(args, os.path.join(config.output.checkpoint_dir, "training_args.bin"))

                                # Good practice: save your training config file together with the trained model
                                save_config(config, os.path.join(config.output.checkpoint_dir, "config.yaml"))

                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    for key, value in logs.items():
                        if config.tensorboard.enable:
                            tb_writer.add_scalar(key, value, global_step)
                        if config.wandb.enable:
                            wandb.log({key: value}, step=global_step)

                    print(json.dumps({**logs, **{"step": global_step}}))

                _, updates = optimizer.step()  # Update weights

                # Log adam updates
                # if config.wandb.project_name == 'adam-updates':
                #     if args.local_rank in [-1, 0] and config.training.train_logging_steps > 0 and ((global_step % config.training.train_logging_steps == 0) or global_step == 0):
                #         # map updates to parameter names
                #         for group_idx, group_updates in enumerate(updates):
                #             group_param_names = optimizer_grouped_parameters_names[group_idx]['params']

                #             for param_idx, (exp_avg, denom) in enumerate(group_updates):
                #                 param_name = group_param_names[param_idx]
                #                 adam_update = exp_avg / denom  # compute adam update

                #                 if config.wandb.enable:
                #                     adam_update = adam_update.view(-1)  # flatten
                #                     mean = torch.mean(adam_update)
                #                     std = torch.std(adam_update)
                #                     wandb.log({f"{param_name}_update_mean": mean}, step=global_step)
                #                     wandb.log({f"{param_name}_update_std": std}, step=global_step)

                scheduler.step()  # Update learning rate schedule
                model.zero_grad()  # Reset gradients to zero
                global_step += 1  # update step count

                # Save model checkpoint during training (based on save_steps)
                if args.local_rank in [-1, 0] and config.training.save_steps > 0 and global_step % config.training.save_steps == 0:
                    output_dir = os.path.join(config.output.checkpoint_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint and tokenizer to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if (config.training.max_steps > 0 and global_step > config.training.max_steps):
                logger.info(f"  Stopping training early after {global_step} steps")
                epoch_iterator.close()
                break

        # Evaluate at the end of the epoch
        if args.local_rank in [-1, 0]:
            logs = {}
            if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, config, model, tokenizer, global_step=global_step)
                eval_metric = results[config.training.early_stopping_metric]

                if eval_metric > last_eval_metric:
                    last_eval_metric = eval_metric
                    best_step = global_step

                    if config.training.early_stopping:
                        # Save only the best model
                        logger.info("   Saving new best model checkpoint to %s", config.output.checkpoint_dir)
                        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
                        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                        # They can then be reloaded using `from_pretrained()`
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(config.output.checkpoint_dir)
                        tokenizer.save_pretrained(config.output.checkpoint_dir)

                        # Good practice: save your training arguments together with the trained model
                        torch.save(args, os.path.join(config.output.checkpoint_dir, "training_args.bin"))

                        # Good practice: save your training config file together with the trained model
                        save_config(config, os.path.join(config.output.checkpoint_dir, "config.yaml"))

                for key, value in results.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value

                for key, value in logs.items():
                    if config.tensorboard.enable:
                        tb_writer.add_scalar(key, value, global_step)
                    if config.wandb.enable:
                        wandb.log({key: value}, step=global_step)

                print(json.dumps({**logs, **{"step": global_step}}))

        if (config.training.max_steps > 0 and global_step > config.training.max_steps):
            logger.info(f"  Stopping training early after {global_step} steps")
            train_iterator.close()
            break

    if args.local_rank in [-1, 0] and config.tensorboard.enable:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_step, last_eval_metric


def evaluate(args, config, model, tokenizer, prefix="", global_step=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched). This is MNLI specific.
    eval_task_names = ("mnli", "mnli-mm") if config.input.task_name == "mnli" else (config.input.task_name,)
    eval_outputs_dirs = (config.output.log_dir, config.output.log_dir + "-MM") if config.input.task_name == "mnli" else (config.output.log_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # Load evaluation data
        eval_dataset = load_and_cache_examples(args, config, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            # This is only neccessary for MNLI
            os.makedirs(eval_output_dir)

        config['eval']['eval_batch_size'] = config.eval.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.eval.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", config.eval.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                # Preprocess batch
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if config.model.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if config.model.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                # Forward pass
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()  # accumulate loss across batches

            nb_eval_steps += 1

            # Collect predictions and ground-truth
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # Average loss
        eval_loss = eval_loss / nb_eval_steps

        # Preprocess predictions based on output mode
        if config.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif config.output_mode == "regression":
            preds = np.squeeze(preds)

        # Compute metrics
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        results['loss'] = eval_loss

        # Save eval results to output file as well
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.tsv")

        if not os.path.exists(output_eval_file):  # file does not exist yet. write header first
            with open(output_eval_file, "a") as writer:
                writer.write("global_step\t" + "\t".join(result.keys()) + "\n")  # write header

        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
            line = [str(global_step)] + [str(r) for r in result.values()]
            writer.write("\t".join(line) + "\n")

    return results


def load_and_cache_examples(args, config, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    task = config.input.task_name
    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        config.input.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, config.model.model_name_or_path.split("/"))).pop(),
            str(config.model.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not config.input.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", config.input.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and config.model.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(config.input.data_dir) if evaluate else processor.get_train_examples(config.input.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=config.model.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(config.model.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if config.model.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="A .yaml config file specifying data, model, training, and evaluation arguments."
    )

    parser.add_argument(
        "--bound",
        type=float,
        default=None,
        help="Bound for the Uniform initialization. If specified, overwrites the config file."
    )

    parser.add_argument(
        "--std",
        type=float,
        default=None,
        help="Std for the Gaussian initialization. If specified, overwrites the config file."
    )

    parser.add_argument("--do_train", action="store_true", help="Run training.")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation **after** training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank. Will be set automatically.")
    parser.add_argument("--no_cuda", action="store_true", help="Run on CPU.")

    args = parser.parse_args()

    # Parse config file
    config = read_config(args.config)

    # Overwrite (some) config arguments based on args
    if args.bound is not None:
        config['model']['bound'] = args.bound
        print('Overwriting model.bound with:', args.bound)

    if args.std is not None:
        config['model']['std'] = args.std
        print('Overwriting model.std with:', args.std)

    # Get timestamp
    CURRENT_TIME = datetime.now().strftime('%m-%d-%H-%M-%S')

    # Create unique output dirs based on timestamp and config file
    # Create output directory if needed
    if args.local_rank in [-1, 0]:
        config['output']['log_dir'] = create_unique_dir(config.output.log_dir, config, args, CURRENT_TIME)
        config['output']['checkpoint_dir'] = create_unique_dir(config.output.checkpoint_dir, config, args, CURRENT_TIME)

        if config.tensorboard.enable:
            config['tensorboard']['log_dir'] = create_unique_dir(config.tensorboard.log_dir, config, args, CURRENT_TIME)

        if config.wandb.enable:
            # Initialize wandb
            config['wandb']['log_dir'] = create_unique_dir(config.wandb.log_dir, config, args, CURRENT_TIME)
            run_name = config.output.log_dir.split('/')[-1]
            wand_config = {**config, **vars(args)}  # combine args and config into single dictionary
            wandb.init(name=run_name, config=wand_config, dir=config.wandb.log_dir, project=config.wandb.project_name)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        config.optimizer.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE training task
    config.input.task_name = config.input.task_name.lower()
    if config.input.task_name not in processors:
        raise ValueError("Task not found: %s" % (config.input.task_name))

    processor = processors[config.input.task_name]()
    config['output_mode'] = output_modes[config.input.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Make sure do_lower_case is used correctly
    config.model.model_type = config.model.model_type.lower()
    if 'albert' in config.model.model_type:
        assert config.model.do_lower_case
    if 'roberta' in config.model.model_type:
        assert not config.model.do_lower_case
    if '-cased' in config.model.model_name_or_path:
        assert not config.model.do_lower_case

    config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model.model_type]

    # Load model config of pre-trained model
    model_config = config_class.from_pretrained(
        config.model.config_name if config.model.config_name else config.model.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=config.input.task_name,
        cache_dir=config.model.cache_dir if config.model.cache_dir else None,
    )

    # Modify model config
    model_config.output_hidden_states = True
    # model_config.output_attentions = True

    # Set pooler a pooler model
    if 'pooler-' in config.model.model_type:
        model_config.pooler = config.model.pooler
        model_config.pooler_dropout = config.model.pooler_dropout
        model_config.distribution = config.model.distribution
        model_config.distribution_bound = config.model.bound
        model_config.distribution_std = config.model.std
        model_config.pooler_activation = config.model.pooler_activation
        model_config.pooler_layer_norm = config.model.pooler_layer_norm

    # Set dropout rate for the encoder
    try:
        model_config.hidden_dropout_prob = config.model.hidden_dropout_prob
        model_config.attention_probs_dropout_prob = config.model.attention_probs_dropout_prob
        model_config.classifier_dropout_prob = config.model.classifier_dropout_prob
    except AttributeError:
        pass  # keep defaults

    # fix layer norm eps
    try:
        model_config.layer_norm_eps = config.model.layer_norm_eps
    except AttributeError:
        pass  # keep defaults

    tokenizer = tokenizer_class.from_pretrained(
        config.model.tokenizer_name if config.model.tokenizer_name else config.model.model_name_or_path,
        do_lower_case=config.model.do_lower_case,
        cache_dir=config.model.cache_dir if config.model.cache_dir else None,
    )

    model = model_class.from_pretrained(
        config.model.model_name_or_path,
        from_tf=bool(".ckpt" in config.model.model_name_or_path),
        config=model_config,
        cache_dir=config.model.cache_dir if config.model.cache_dir else None,
    )

    # Init weights of linear classifier and maybe pooler
    if 'pooler-' in config.model.model_type:
        model.manually_init_weights(config.model.classifier_init_std)  # re-init the linear classifier

        if config.model.re_init_pooler:  # re-init pooler weights
            assert config.model.distribution in ['normal', 'uniform']

            if config.model.model_type == 'pooler-bert':
                model.bert.pooler.reset_parameters()
            elif config.model.model_type == 'pooler-roberta':
                model.classifier.reset_pooler_parameters()
            elif config.model.model_type == 'pooler-albert':
                model.albert.reset_pooler_parameters()
            else:
                raise KeyError(f'Unknown model type: {config.model.model_type}')
        else:
            assert config.model.distribution == 'none'

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)  # Put model on device

    logger.info("Training/evaluation arguments %s", args)
    logger.info("Training/evaluation config file %s", config)

    # Training
    if args.do_train:
        # Loading training dataset
        train_dataset = load_and_cache_examples(args, config, tokenizer, evaluate=False)
        # Start training
        global_step, tr_loss, best_step, last_eval_metric = train(args, config, train_dataset, model, tokenizer)
        logger.info("End of training: global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info("Best model on dev set saved at step: %s with metric = %s", best_step, last_eval_metric)

    # Save model obtained at the end of training (only if we don't do early stopping)
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not config.training.early_stopping:
        logger.info("Saving last model checkpoint to %s", config.output.checkpoint_dir)
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(config.output.checkpoint_dir)
        tokenizer.save_pretrained(config.output.checkpoint_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(config.output.checkpoint_dir, "training_args.bin"))

        # Good practice: save your training config file together with the trained model
        save_config(config, os.path.join(config.output.checkpoint_dir, "config.yaml"))

    # Evaluate after training
    results = {}

    if args.do_eval and args.local_rank in [-1, 0]:
        # Load saved tokenizer
        tokenizer = tokenizer_class.from_pretrained(config.output.checkpoint_dir, do_lower_case=config.model.do_lower_case)

        # Collect all checkpoints
        checkpoints = [config.output.checkpoint_dir]  # checkpoint that was saved after training

        # Collect checkpoints saved during training (checkpoint_dir/checkpoint-globalstep)
        if config.eval.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(config.output.checkpoint_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else "-1"
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint-") != -1 else ""

            # Restore model from checkpoint
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)  # Put model on device

            # Run evaluation
            result = evaluate(args, config, model, tokenizer, prefix=prefix, global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
