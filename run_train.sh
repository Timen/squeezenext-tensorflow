#!/usr/bin/env bash
DATE=`date '+%Y-%m-%d_%H-%M'`
TRAIN_PATH="/usr/local/share/models/"
TRAIN_DIR=$TRAIN_PATH$DATE

if [[ ! -e $DATA_DIR ]]; then
    echo "Data dir $DATA_DIR does not exists." 1>&2
    exit 1
fi
if [[ ! -e $TRAIN_DIR ]]; then
    mkdir $TRAIN_DIR
elif [[ ! -d $TRAIN_DIR ]]; then
    echo "Model dir $TRAIN_DIR already exists but is not a directory" 1>&2
fi

PYTHONPATH="./" python train.py \
--model_dir $TRAIN_DIR \
--configuration "v_1_0_SqNxt_23" \
--batch_size 256 \
--num_epochs 120 \
--training_file_pattern $DATA_DIR"tf-records/train-*" \
--validation_file_pattern $DATA_DIR"tf-records/validation-*"
