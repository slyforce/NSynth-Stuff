#!/bin/bash

MODELDIR="./models/small-model"
mkdir -p $MODELDIR

python main_classification.py \
--command train  \
--gpu=true \
--model_directory $MODELDIR \
--training_data /tmp/nsynth-train \
--valid_data /tmp/nsynth-valid \
--test_data /tmp/nsynth-test \
--model_hidden_dim 64 \
--batch_size 128
