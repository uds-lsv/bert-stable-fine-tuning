#!/usr/bin/env bash
export PYTHONPATH="$PYTHONPATH:/transformers"

# Passed as arguements
CONFIG_FILE=$1
SEED=$2
export CUDA_VISIBLE_DEVICES=$3

# Setup weights & biases environment variables
# Comment lines below if you don't want to use wandb
export WANDB_API_KEY=your-key
export WANDB_USERNAME="your-username"
export WANDB_ENTITY="your-entity"

python /transformers/examples/squad/run_finetuning.py \
    --config ${CONFIG_FILE} \
    --do_train \
    --do_eval \
    --seed ${SEED}