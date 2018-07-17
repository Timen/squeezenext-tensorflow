#!/usr/bin/env bash

MODELS="/usr/local/share/models/"
DATASETS="/usr/local/share/Datasets/"

docker run -it -p 6006:6006  \
-v $(pwd):/usr/local/src/ \
-v $MODELS:$MODELS \
-v $DATASETS:$DATASETS \
tensorflow/tensorflow python -m tensorboard.main --logdir=$MODELS
