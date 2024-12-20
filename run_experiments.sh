#!/bin/bash

# Define the values for epochs_per_lr_decay and learning_rate
epochs_per_lr_decay_values=(4)
learning_rate_values=(0.0001)

# Loop over each combination of epochs_per_lr_decay and learning_rate
for epochs_per_lr_decay in "${epochs_per_lr_decay_values[@]}"; do
  for learning_rate in "${learning_rate_values[@]}"; do
    echo "Running training with epochs_per_lr_decay=${epochs_per_lr_decay} and learning_rate=${learning_rate}"
    python train.py --epochs_per_lr_decay ${epochs_per_lr_decay} --learning_rate ${learning_rate} --batch_size 256 --epochs 15
  done
done