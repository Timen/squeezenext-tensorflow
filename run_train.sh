#!/usr/bin/env bash
DATE=`date '+%Y-%m-%d_%H-%M'`
TRAIN_PATH="/usr/local/share/models/"
TRAIN_DIR=$TRAIN_PATH$DATE

PYTHONPATH="./" python train.py \
--model_dir $TRAIN_DIR \
--training_file_pattern "/usr/local/share/Datasets/Imagenet/validation-*" \
--validation_file_pattern "/usr/local/share/Datasets/Imagenet/validation-*"