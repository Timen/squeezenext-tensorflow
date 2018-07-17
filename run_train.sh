#!/usr/bin/env bash
DATE=`date '+%Y-%m-%d_%H-%M'`
TRAIN_PATH="/usr/local/share/models/"
TRAIN_DIR=$TRAIN_PATH$DATE
DATA_DIR="/usr/local/share/Datasets/Imagenet/"
PYTHONPATH="./" python train.py \
--model_dir $TRAIN_DIR \
--batch_size 256 \
--num_epochs 30 \
--training_file_pattern $DATA_DIR"train-*" \
--validation_file_pattern $DATA_DIR"validation-*"