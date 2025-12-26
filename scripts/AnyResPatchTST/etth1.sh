#!/bin/bash

# AnyResPatchTST Training Script for ETTh1 Dataset
# Multi-scale patch configuration experiment

# Model: AnyResPatchTST with default multi-scale patches (8, 16, 32)
# Dataset: ETTh1
# Prediction lengths: 96, 192, 336, 720

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/AnyResPatchTST" ]; then
    mkdir ./logs/AnyResPatchTST
fi

model_name=AnyResPatchTST
root_path_name=./data/ETT/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021

# Common hyperparameters
seq_len=512
e_layers=3
n_heads=4
d_model=128
d_ff=256
dropout=0.2
fc_dropout=0.2
head_dropout=0

# AnyRes-specific parameters
patch_scales="16,32,64"
use_global_token=1
cross_scale_layers=2
cross_scale_fusion=attention
share_encoder=1

for pred_len in 96 192 336 720
do
    echo "Training AnyResPatchTST on ETTh1 with pred_len=${pred_len}"
    
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers $e_layers \
      --n_heads $n_heads \
      --d_model $d_model \
      --d_ff $d_ff \
      --dropout $dropout \
      --fc_dropout $fc_dropout \
      --head_dropout $head_dropout \
      --patch_scales $patch_scales \
      --use_global_token $use_global_token \
      --cross_scale_layers $cross_scale_layers \
      --cross_scale_fusion $cross_scale_fusion \
      --share_encoder $share_encoder \
      --train_epochs 100 \
      --patience 20 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --des 'Exp' \
      --lradj 'TST' \
      --pct_start 0.4 \
      >logs/AnyResPatchTST/${model_id_name}_${seq_len}_${pred_len}.log 2>&1
      
    echo "Completed training for pred_len=${pred_len}"
done

echo "All experiments completed!"
