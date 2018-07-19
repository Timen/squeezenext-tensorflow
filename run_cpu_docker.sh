#!/usr/bin/env bash

MODELS="/usr/local/share/models/"

docker run -it  \
-v $(pwd):/usr/local/src/ \
-v $MODELS:$MODELS \
-v $DATA_DIR:$DATA_DIR \
tensorflow/tensorflow \
bash -c "export DATA_DIR="$DATA_DIR"; bash"

