#!/bin/bash

# Training script for Timer-XL Peru Rainfall Prediction
# Transfer learning from pre-trained Timer-XL (260B time points)

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0

# Model Configuration
model_name=timer_xl_classifier
token_len=96          # Timer-XL token length
token_num=15          # Number of tokens (15 * 96 * 12h = 180 days context)
seq_len=$[$token_num*$token_len]  # Total sequence length

# For longer context experiments, try:
# token_num=30  # 360 days (1 year)
# token_num=60  # 720 days (2 years)

# Data Configuration
data_path=peru_rainfall.csv
root_path=./datasets/processed/

# Training Configuration
batch_size=256        # Adjust based on GPU memory
learning_rate=1e-5    # Low LR for fine-tuning
train_epochs=50
patience=10

# Transfer Learning
pretrain_path=./checkpoints/timer_xl/checkpoint.pth

echo "========================================"
echo "Timer-XL Peru Rainfall Training"
echo "========================================"
echo "Model: $model_name"
echo "Context length: $seq_len timesteps ($(($seq_len / 2)) days)"
echo "Batch size: $batch_size"
echo "Learning rate: $learning_rate"
echo "========================================"

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id peru_rainfall_transfer_learning \
  --model $model_name \
  --data PeruRainfall \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 2 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --dropout 0.1 \
  --activation relu \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --gpu 0 \
  --cosine \
  --tmax $train_epochs \
  --use_norm \
  --adaptation \
  --pretrain_model_path $pretrain_path \
  --loss CE \
  --checkpoints ./results/peru_rainfall/ \
  --use_focal_loss \
  --itr 1

echo "========================================"
echo "Training complete!"
echo "Results saved to: ./results/peru_rainfall/"
echo "========================================"
