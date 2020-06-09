#!/usr/bin/env bash
export PYTHONPATH="$PYTHONPATH:/transformers"

# Passed as arguements
CONFIG_FILE=$1
SEED=$2
N_GPUS=$3
export CUDA_VISIBLE_DEVICES=$4

# Setup weights & biases environment variables
# Comment lines below if you don't want to use wandb
export WANDB_API_KEY=your-key
export WANDB_USERNAME="your-username"
export WANDB_ENTITY="your-entity"

python -m torch.distributed.launch --nproc_per_node ${N_GPUS} /transformers/examples/squad/run_finetuning.py \
    --config ${CONFIG_FILE} \
    --do_train   \
    --do_eval   \
    --seed ${SEED}