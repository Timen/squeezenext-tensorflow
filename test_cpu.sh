#!/usr/bin/env bash
DATE=`date '+%Y-%m-%d_%H-%M'`
TRAIN_PATH="/usr/local/share/models/"
TRAIN_DIR=$TRAIN_PATH$DATE

PYTHONPATH="./" python train.py \
--model_dir $TRAIN_DIR \
--batch_size 2 \
--configuration "v_1_0_G_SqNxt_23" \
--num_epochs 30 \
--training_file_pattern "/usr/local/share/Datasets/Imagenet/train-*" \
--validation_file_pattern "/usr/local/share/Datasets/Imagenet/validation-*"