#!/bin/bash


algo=$1
p_s=$2
p_e=$3
data=$4
wandb_project=$5
wandb_key=$6
seed=$7

# Start training
python src/main/run_osaka_benchmark.py --dataset $data --prob_statio $p_s --prob_env_switch $p_e --model_name_impv $algo --output "output"  --pretrain --pretrain_model "models/${data}_maml.model" -v --wandb $wandb_project  --wandb_key $wandb_key --seed $seed



