#!/usr/bin/env bash
DATA_DIR="/usr/local/share/Datasets/Imagenet/"

PYTHONPATH=$PWD bash datasets/process_downloaded_imagenet.sh $DATA_DIR
