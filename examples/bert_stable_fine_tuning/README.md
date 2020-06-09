## Installing & Getting Started

1. Clone the repository.

````
git clone git@github.com:uds-lsv/bert-stable-fine-tuning.git
cd bert-stable-fine-tuning/examples/bert_stable_fine_tuning
````

2. [Download datasets](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py) from the GLUE benchmark.

3. Setup a Docker image and start a container. 

````
docker build -f .Dockerfile --build-arg USER_UID=$UID --build-arg USER_NAME=$(id -un) -t bert-stable-fine-tuning:latest .

docker run -it --rm --runtime=nvidia --pid=host --ipc=host \
    -v /path/to/bert-stable-fine-tuning:/transformers \
    -v /path/to/pre-trained-transformers:/pre-trained-transformers \
    -v /path/to/datasets:/datasets \
    -v /path/to/bert-stable-fine-tuning/logfiles:/logfiles \
    -v /path/to/bert-stable-fine-tuning/checkpoints:/checkpoints \
    -v /path/to/bert-stable-fine-tuning/tb-logs:/tb-logs \
    -v /path/to/bert-stable-fine-tuning/wandb-logs:/wandb-logs \
    bert-stable-fine-tuning:latest
````

Add `--user=<username>` to the `docker run` command above in order to run the container as your user. Use `--gpus=all instead` instead of `--runtime=nvidia` for more recent Docker versions (starting from 19.03). More information on Docker can be found here: `/bert_stable_fine_tuning/run_docker.txt`

4. Install huggingface transformers in editable mode **inside** the container.

````
python3 -m pip install -e . --user
````

5. Fine-tune BERT-large-uncased on RTE. (You might want to check `./bert_stable_fine_tuning/scripts/seeds.sh` first.) 

````
bash /transformers/examples/bert_stable_fine_tuning/scripts/seeds.sh /transformers/examples/bert_stable_fine_tuning/configs/rte/pooler-bert-large-uncased_bsz_16_lr_2e-05_adamW_bias-correct_early-stopping_20.yaml 1 1 0  
````

6. Additional config files for RTE, MRPC, and CoLA can be found here: `./bert_stable_fine_tuning/configs`. Bash commands for every config file can be found here: `./bert_stable_fine_tuning/scripts/run_scripts.sh`

**Happy stable fine-tuning!** :rocket: :metal: 

