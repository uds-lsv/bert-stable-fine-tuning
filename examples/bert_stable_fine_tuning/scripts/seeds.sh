#!/usr/bin/env bash
export PYTHONPATH="$PYTHONPATH:/transformers"

# Passed as arguements
CONFIG_FILE=$1
FIRST_SEED=$2
LAST_SEED=$3
export CUDA_VISIBLE_DEVICES=$4

# Setup weights & biases environment variables
# Comment lines below if you don't want to use wandb
export WANDB_API_KEY=your-key
export WANDB_USERNAME="your-username"
export WANDB_ENTITY="your-entity"

# Train the same model on the same dataset with different random seeds
for SEED in $(seq $FIRST_SEED $LAST_SEED);
do
    python /transformers/examples/bert_stable_fine_tuning/run_finetuning.py \
        --config ${CONFIG_FILE} \
        --do_train \
        --do_eval \
        --seed ${SEED}
done