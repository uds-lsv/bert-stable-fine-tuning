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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit
from datetime import datetime
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    CamembertConfig,
    CamembertForQuestionAnswering,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

from examples.squad.utils import read_config, save_config, create_unique_dir, set_seed, to_list

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import wandb

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, CamembertConfig, RobertaConfig, XLNetConfig, XLMConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "camembert": (CamembertConfig, CamembertForQuestionAnswering, CamembertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
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
        config.training.num_train_epochs = config.training.max_steps // (len(train_dataloader) // config.training.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // config.training.gradient_accumulation_steps * config.training.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.optimizer.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.optimizer.learning_rate, eps=config.optimizer.adam_epsilon)

    # Create learning rate schedule
    if config.optimizer.learning_rate_schedule == 'warmup-linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.optimizer.warmup_steps, num_training_steps=t_total
        )
    elif config.optimizer.learning_rate_schedule == 'warmup-constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=config.optimizer.warmup_steps
        )
    elif config.optimizer.learning_rate_schedule == 'warmup-cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config.optimizer.warmup_steps, num_training_steps=t_total, num_cycles=0.5
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
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
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

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(config.model.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = config.model.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // config.training.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // config.training.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(config.training.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    # Added here for reproductibility
    set_seed(args)

    # Evaluate and log once before training starts/resumes
    if args.local_rank in [-1, 0]:
        # Only evaluate when single GPU and not torch.distributed otherwise metrics may not average well
        if args.local_rank == -1 and config.training.evaluate_during_training:
            logs = {}
            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar

            results = evaluate(args, config, model, tokenizer, global_step=global_step)
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
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if config.model.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if config.model.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if config.input.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * config.input.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if config.training.gradient_accumulation_steps > 1:
                loss = loss / config.training.gradient_accumulation_steps

            if config.optimizer.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                global_step += 1  # moved here because of logging

                if args.local_rank in [-1, 0] and config.training.train_logging_steps > 0 and global_step % config.training.train_logging_steps == 0:
                    # Log gradients before clipping them
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        grads = param.grad.view(-1)
                        grads_norm = torch.norm(grads, p=2, dim=0)
                        if config.tensorboard.enable:
                            if config.tensorboard.log_histograms:
                                # logging histograms can result in large log files
                                tb_writer.add_histogram(name, param, global_step)  # log param histograms
                            tb_writer.add_scalar(name + '_grad_norm', grads_norm, global_step)  # log param gradient norm
                        if config.wandb.enable:
                            wandb.log({f"{name}_grad_norm": grads_norm}, step=global_step)

                # clip gradients
                if config.optimizer.fp16:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.optimizer.max_grad_norm)
                else:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                # Log training progress, we will log once after the very first iteration
                if args.local_rank in [-1, 0] and config.training.train_logging_steps > 0 and (global_step % config.training.train_logging_steps == 0 or global_step == 1):
                    logs = {}

                    # Log training progress
                    loss_scalar = (tr_loss - logging_loss) / config.training.train_logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["train_loss"] = loss_scalar
                    logging_loss = tr_loss

                    # Log total gradient norm (after clipping)
                    if config.tensorboard.enable:
                        tb_writer.add_scalar('total_grad_norm', total_grad_norm, global_step)
                    if config.wandb.enable:
                        wandb.log({'total_grad_norm': total_grad_norm}, step=global_step)

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
                if args.local_rank in [-1, 0] and config.training.eval_logging_steps > 0 and global_step % config.training.eval_logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and config.training.evaluate_during_training:
                        results = evaluate(args, config, model, tokenizer, global_step=global_step)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                        for key, value in logs.items():
                            if config.tensorboard.enable:
                                tb_writer.add_scalar(key, value, global_step)
                            if config.wandb.enable:
                                wandb.log({key: value}, step=global_step)

                        print(json.dumps({**logs, **{"step": global_step}}))

                # Save model checkpoint
                if args.local_rank in [-1, 0] and config.training.save_steps > 0 and global_step % config.training.save_steps == 0:
                    output_dir = os.path.join(config.output.checkpoint_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint and tokenizer to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    save_config(config, os.path.join(output_dir, "config.yaml"))

                    logger.info("Saving args and config to %s", output_dir)

            if config.training.max_steps > 0 and global_step > config.training.max_steps:
                epoch_iterator.close()
                break

        if config.training.max_steps > 0 and global_step > config.training.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0] and config.tensorboard.enable:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, config, model, tokenizer, prefix="", global_step=0):
    dataset, examples, features = load_and_cache_examples(args, config, tokenizer, evaluate=True, output_examples=True)

    config['eval']['eval_batch_size'] = config.eval.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=config.eval.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", config.eval.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if config.model.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if config.model.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * config.input.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(config.output.log_dir, f"predictions_{prefix}.json")
    output_nbest_file = os.path.join(config.output.log_dir, f"nbest_{config.model.n_best_size}_predictions_{prefix}.json")

    if config.input.version_2_with_negative:
        output_null_log_odds_file = os.path.join(config.output.log_dir, f"null_odds_{prefix}.json")
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if config.model.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            config.model.n_best_size,
            config.model.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            config.input.version_2_with_negative,
            tokenizer,
            config.output.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            config.model.n_best_size,
            config.model.max_answer_length,
            config.model.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            config.output.verbose_logging,
            config.input.version_2_with_negative,
            config.model.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)

    # Save eval results to output file as well
    if prefix == "-1":
        # evaluate at the end of training, store in the log_dir directly
        output_eval_file = os.path.join(config.output.log_dir, "eval_results.tsv")
    else:
        # there is a 'prefix' subfolder
        output_eval_file = os.path.join(config.output.log_dir, prefix, "eval_results.tsv")

    if not os.path.exists(output_eval_file):  # file does not exist yet. write header first
        with open(output_eval_file, "a") as writer:
            writer.write("global_step\t" + "\t".join(results.keys()) + "\n")  # write header

    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        line = [str(global_step)] + [str(r) for r in results.values()]
        writer.write("\t".join(line) + "\n")

    return results


def load_and_cache_examples(args, config, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = config.input.data_dir if config.input.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, config.model.model_name_or_path.split("/"))).pop(),
            str(config.model.max_seq_length),
            str(config.model.max_query_length),
            str(config.input.doc_stride),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not config.input.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not config.input.data_dir and ((evaluate and not config.input.predict_file) or (not evaluate and not config.input.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if config.input.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if config.input.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(config.input.data_dir, filename=config.input.predict_file)
            else:
                examples = processor.get_train_examples(config.input.data_dir, filename=config.input.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=config.model.max_seq_length,
            doc_stride=config.input.doc_stride,
            max_query_length=config.model.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=config.input.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
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
        required=True,
        help="A .yaml config file specifying data, model, training, and evaluation arguments."
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    args = parser.parse_args()

    # Parse config file
    config = read_config(args.config)

    # Get timestamp
    CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')

    if config.input.doc_stride >= config.model.max_seq_length - config.model.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    # Create unique output dirs based on timestamp and config file
    # Create output directory if needed
    if args.local_rank in [-1, 0]:
        config['output']['log_dir'] = create_unique_dir(config.output.log_dir, config, CURRENT_TIME)
        config['output']['checkpoint_dir'] = create_unique_dir(config.output.checkpoint_dir, config, CURRENT_TIME)

        if config.tensorboard.enable:
            config['tensorboard']['log_dir'] = create_unique_dir(config.tensorboard.log_dir, config, CURRENT_TIME)

        if config.wandb.enable:
            # Initialize wandb
            config['wandb']['log_dir'] = create_unique_dir(config.wandb.log_dir, config, CURRENT_TIME)
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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config.model.model_type = config.model.model_type.lower()

    # Make sure do_lower_case is used correctly
    config.model.model_type = config.model.model_type.lower()
    if 'albert' in config.model.model_type:
        assert config.model.do_lower_case
    if 'roberta' in config.model.model_type:
        assert not config.model.do_lower_case
    if '-cased' in config.model.model_name_or_path:
        assert not config.model.do_lower_case

    config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model.model_type]

    model_config = config_class.from_pretrained(
        config.model.config_name if config.model.config_name else config.model.model_name_or_path,
        cache_dir=config.model.cache_dir if config.model.cache_dir else None,
    )
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

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation arguments %s", args)
    logger.info("Training/evaluation config file %s", config)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if config.optimizer.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if config.optimizer.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, config, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, config, train_dataset, model, tokenizer)
        logger.info("End of training: global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", config.output.checkpoint_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(config.output.checkpoint_dir)
        tokenizer.save_pretrained(config.output.checkpoint_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(config.output.checkpoint_dir, "training_args.bin"))

        # Good practice: save your training config file together with the trained model
        save_config(config, os.path.join(config.output.checkpoint_dir, "config.yaml"))

        # # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(config.output.checkpoint_dir)  # , force_download=True)
        # tokenizer = tokenizer_class.from_pretrained(config.output.checkpoint_dir, do_lower_case=config.model.do_lower_case)
        # model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # Load saved tokenizer
        tokenizer = tokenizer_class.from_pretrained(config.output.checkpoint_dir, do_lower_case=config.model.do_lower_case)

        # Collect all checkpoints
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [config.output.checkpoint_dir]
            if config.eval.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(config.output.checkpoint_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", config.model.model_name_or_path)
            checkpoints = [config.model.model_name_or_path]  # save a pre-trained checkpoint

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else "-1"
            model = model_class.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, config, model, tokenizer, prefix=global_step)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
